"""Build a runnable DK-VLM from a released checkpoint and answer questions with it.

A checkpoint directory contains three things:

``language_model_lora/``   LoRA adapter for the Qwen2.5-VL or Qwen3-VL text tower
``geo_branch_state.pt``    geometry-branch gates and projections
``dkvlm_config.json``      which layers carry the branch, and its slice widths (optional;
                           sensible defaults are used when absent)

The legacy filename ``p2vlm_config.json`` is also accepted for older checkpoints.

Base Qwen2.5-VL or Qwen3-VL weights are downloaded separately from the upstream model hub, and
VGGT-Omega weights come from its own release. Nothing here fetches them for you.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from .frames import (
    SampledFrames,
    image_content_elements,
    load_image_frames,
    processor_visual_grid,
    sample_video_frames,
)
from .geometry import VGGTOmegaConfig, VGGTOmegaEngine
from .qwen3_geo import DKVLMGeoWrapper, install_geo_attention
from .tokens import DEFAULT_METRIC_GLOBAL_SCALE, build_geo_coords

DEFAULT_GEO_LAYERS = (6, 12, 18, 27, 33)
DEFAULT_TOKEN_SE3_DIM = 48
DEFAULT_POSE_GTA_DIM = 24
DEFAULT_LAMBDA_MIN = 0.25
DEFAULT_LAMBDA_MAX = 4.0

DEFAULT_NFRAMES = 32
DEFAULT_MAX_PIXELS = 786432
DEFAULT_MAX_NEW_TOKENS = 128


def _load_vl_base(path: Path, attn_implementation: str, dtype: torch.dtype):
    """Load Qwen3-VL or Qwen2.5-VL from ``path`` according to its config."""
    model_type = str(getattr(AutoConfig.from_pretrained(str(path)), "model_type", ""))
    if model_type == "qwen3_vl":
        cls = Qwen3VLForConditionalGeneration
    elif model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        cls = Qwen2_5_VLForConditionalGeneration
    else:
        raise ValueError(
            f"unsupported model_type={model_type!r} at {path}; "
            "DK-VLM supports qwen3_vl and qwen2_5_vl"
        )
    return cls.from_pretrained(str(path), attn_implementation=attn_implementation, dtype=dtype)


def _layers_from_geo_state(state: Dict[str, torch.Tensor]) -> List[int]:
    return sorted(
        {
            int(key.split(".")[1])
            for key in state
            if key.startswith("geo_gate.") and key.count(".") == 1
        }
    )


def resolve_attn_implementation(preferred: str) -> str:
    """Prefer Flash Attention 2, then SDPA. ``eager`` is only used if requested."""
    wanted = (preferred or "flash_attention_2").strip().lower()
    if wanted == "eager":
        return "eager"
    if wanted == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except Exception:
            return "sdpa"
    if wanted == "sdpa":
        return "sdpa"
    raise ValueError(f"unknown attn_implementation={preferred!r}")

PRE_PROMPT = "These are frames of a video."
MCA_POST_PROMPT = "Answer with the option's letter from the given choices directly."
NA_POST_PROMPT = "Please answer the question using a single word or phrase."


def build_prompt(question: str, options: Optional[Sequence[str]] = None) -> str:
    """Compose the question the way the model was trained to read it."""
    if options:
        return "\n".join(
            [PRE_PROMPT, question, "Options:\n" + "\n".join(options), MCA_POST_PROMPT]
        )
    return PRE_PROMPT + "\n" + question + "\n" + NA_POST_PROMPT


@dataclass
class GeoBranchConfig:
    """Shape of the geometry branch. Must match the checkpoint being loaded."""

    target_layers: List[int] = field(default_factory=lambda: list(DEFAULT_GEO_LAYERS))
    geo_branch_mrope: bool = False
    token_se3_dim: int = DEFAULT_TOKEN_SE3_DIM
    pose_gta_dim: int = DEFAULT_POSE_GTA_DIM
    lambda_min: float = DEFAULT_LAMBDA_MIN
    lambda_max: float = DEFAULT_LAMBDA_MAX
    metric_global_scale: float = DEFAULT_METRIC_GLOBAL_SCALE
    normalize_mode: str = "metric"

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: Path) -> "GeoBranchConfig":
        path = Path(checkpoint_dir) / "dkvlm_config.json"
        if not path.is_file():
            path = Path(checkpoint_dir) / "p2vlm_config.json"
        if not path.is_file():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        cfg = cls()
        if "target_layers" in raw:
            cfg.target_layers = [int(x) for x in raw["target_layers"]]
        if "geo_branch_mrope" in raw:
            cfg.geo_branch_mrope = bool(raw["geo_branch_mrope"])
        slice_pair = raw.get("token_se3_slice")
        if slice_pair:
            cfg.token_se3_dim = int(slice_pair[1]) - int(slice_pair[0])
        slice_pair = raw.get("pose_gta_slice")
        if slice_pair:
            cfg.pose_gta_dim = int(slice_pair[1]) - int(slice_pair[0])
        lambdas = raw.get("token_se3_lambdas")
        if lambdas:
            cfg.lambda_min = float(min(lambdas))
            cfg.lambda_max = float(max(lambdas))
        for key in ("metric_global_scale", "normalize_mode"):
            if key in raw:
                setattr(cfg, key, type(getattr(cfg, key))(raw[key]))
        return cfg


class DKVLM:
    """A loaded DK-VLM: Qwen-VL with the geometry branch, plus the geometry front-end."""

    def __init__(
        self,
        model: DKVLMGeoWrapper,
        processor: AutoProcessor,
        geo_cfg: GeoBranchConfig,
        geometry_engine: Optional[VGGTOmegaEngine] = None,
        *,
        nframes: int = DEFAULT_NFRAMES,
        max_pixels: Optional[int] = DEFAULT_MAX_PIXELS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ):
        self.model = model
        self.processor = processor
        self.geo_cfg = geo_cfg
        self.geometry_engine = geometry_engine
        self.nframes = int(nframes)
        self.max_pixels = max_pixels
        self.max_new_tokens = int(max_new_tokens)
        self.patch_size, self.merge_size, self.cell_size = processor_visual_grid(processor)

    @property
    def device(self) -> torch.device:
        return next(self.model.base.parameters()).device


    def compute_geometry(self, sampled: SampledFrames, sample_id: str = ""):
        """Run VGGT-Omega over the sampled frames."""
        if self.geometry_engine is None:
            raise RuntimeError(
                "no geometry engine configured; construct DKVLM with geometry_ckpt=... "
                "or pass use_geometry=False to run DK-VLM without the geometric branch"
            )
        return self.geometry_engine.infer(
            pil_images=sampled.frames,
            source_frame_names=sampled.frame_names,
            sample_id=sample_id,
        )

    def build_inputs(
        self,
        sampled: SampledFrames,
        prompt: str,
        *,
        geometry=None,
        sample_id: str = "",
    ) -> Dict[str, Any]:
        """Tokenize frames plus prompt, and attach per-token geometry features."""
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": image_content_elements(sampled.frames, max_pixels=self.max_pixels)
                + [{"type": "text", "text": prompt}],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(
            messages, image_patch_size=self.patch_size
        )
        inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = self.device
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        if geometry is None:
            return inputs

        cfg = self.model.base.config
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is None or int(image_grid_thw.numel()) == 0:
            raise ValueError(
                "expected image_grid_thw; DK-VLM sends frames as independent images"
            )
        if int(image_grid_thw.shape[0]) != len(sampled):
            raise ValueError(
                f"image grid rows {int(image_grid_thw.shape[0])} != sampled frames {len(sampled)}"
            )

        geo_coords, geo_mask, stats = build_geo_coords(
            geometry,
            inputs["input_ids"][0].cpu(),
            image_grid_thw.cpu(),
            spatial_merge_size=self.merge_size,
            patch_size=self.patch_size,
            image_token_id=int(cfg.image_token_id),
            video_token_id=int(cfg.video_token_id),
            normalize_mode=self.geo_cfg.normalize_mode,
            metric_global_scale=self.geo_cfg.metric_global_scale,
            sample_id=sample_id,
        )
        inputs = dict(inputs)
        inputs["geo_coords"] = geo_coords.unsqueeze(0).to(device=device, dtype=torch.float32)
        inputs["geo_coords_mask"] = geo_mask.unsqueeze(0).to(device=device, dtype=torch.bool)
        self.last_geometry_stats = stats
        return inputs


    @torch.inference_mode()
    def generate(self, inputs: Dict[str, Any], **gen_kwargs) -> str:
        """Greedy-decode an answer and return the newly generated text."""
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
        }
        kwargs.update(gen_kwargs)
        generated = self.model.generate(**inputs, **kwargs)
        trimmed = generated[0][len(inputs["input_ids"][0]) :]
        return self.processor.decode(trimmed, skip_special_tokens=True)

    def answer(
        self,
        source: Union[str, Path, SampledFrames, Sequence[Image.Image]],
        question: str,
        *,
        options: Optional[Sequence[str]] = None,
        use_geometry: bool = True,
        sample_id: str = "",
        **gen_kwargs,
    ) -> str:
        """Answer a question about a video, a directory of frames, or decoded frames."""
        sampled = self.resolve_frames(source)
        geometry = self.compute_geometry(sampled, sample_id) if use_geometry else None
        inputs = self.build_inputs(
            sampled, build_prompt(question, options), geometry=geometry, sample_id=sample_id
        )
        return self.generate(inputs, **gen_kwargs)

    def resolve_frames(
        self, source: Union[str, Path, SampledFrames, Sequence[Image.Image]]
    ) -> SampledFrames:
        """Normalize any accepted input into :class:`SampledFrames`."""
        if isinstance(source, SampledFrames):
            return source
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                files = sorted(
                    p
                    for p in path.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                )
                if not files:
                    raise FileNotFoundError(f"no image files in {path}")
                return load_image_frames(files)
            if not path.is_file():
                raise FileNotFoundError(f"video not found: {path}")
            return sample_video_frames(path, self.nframes)
        frames = list(source)
        if not frames:
            raise ValueError("empty frame sequence")
        return SampledFrames(
            frames=[f.convert("RGB") for f in frames],
            frame_indices=list(range(len(frames))),
            frame_names=[f"frame_{i:06d}.png" for i in range(len(frames))],
        )


def load_dkvlm(
    checkpoint_dir: Union[str, Path],
    *,
    base_model_path: Union[str, Path],
    geometry_ckpt: str = "",
    device: Optional[Union[str, torch.device]] = None,
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: str = "flash_attention_2",
    geo_attention_backend: str = "sdpa",
    nframes: int = DEFAULT_NFRAMES,
    max_pixels: Optional[int] = DEFAULT_MAX_PIXELS,
    use_geometry: bool = True,
    verbose: bool = True,
) -> DKVLM:
    """Assemble base model, LoRA adapter, geometry branch and geometry front-end."""
    ckpt = Path(checkpoint_dir).resolve()
    if not ckpt.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {ckpt}")

    lora_dir = ckpt / "language_model_lora"
    if not (lora_dir / "adapter_config.json").is_file():
        parent = ckpt.parent / "language_model_lora"
        lora_dir = parent if (parent / "adapter_config.json").is_file() else lora_dir
    if not (lora_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"missing LoRA adapter_config.json in {lora_dir}; "
            "DK-VLM requires its released adapter and cannot fall back to base weights"
        )
    if not any((lora_dir / name).is_file() for name in (
        "adapter_model.safetensors", "adapter_model.bin",
        "adapter_model.safetensors.index.json", "adapter_model.bin.index.json",
    )):
        raise FileNotFoundError(f"missing LoRA adapter weights in {lora_dir}")

    dev = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch_dtype or (torch.bfloat16 if dev.type == "cuda" else torch.float32)
    attn_impl = resolve_attn_implementation(attn_implementation)
    if verbose and attn_impl != attn_implementation:
        print(f"[dkvlm] attn_implementation {attn_implementation} -> {attn_impl}")

    base = _load_vl_base(Path(base_model_path), attn_impl, dtype).to(dev)
    base.eval()

    from peft import PeftModel

    base.model.language_model = PeftModel.from_pretrained(
        base.model.language_model, str(lora_dir), is_trainable=False
    )
    if verbose:
        print(f"[dkvlm] loaded LoRA adapter from {lora_dir}")

    proc_src = ckpt if (ckpt / "preprocessor_config.json").is_file() else Path(base_model_path)
    processor = AutoProcessor.from_pretrained(str(proc_src), use_fast=False)
    if verbose:
        print(f"[dkvlm] processor from {proc_src}")

    geo_state_path = ckpt / "geo_branch_state.pt"
    if not geo_state_path.is_file():
        parent = ckpt.parent / "geo_branch_state.pt"
        if not parent.is_file():
            raise FileNotFoundError(f"missing geo_branch_state.pt in {ckpt}")
        geo_state_path = parent
    geo_state = torch.load(str(geo_state_path), map_location="cpu", weights_only=True)

    geo_cfg = GeoBranchConfig.from_checkpoint(ckpt)
    if not any((ckpt / name).is_file() for name in ("dkvlm_config.json", "p2vlm_config.json")):
        inferred = _layers_from_geo_state(geo_state)
        if inferred:
            geo_cfg.target_layers = inferred
    wrapper = DKVLMGeoWrapper(
        base,
        target_layers=geo_cfg.target_layers,
        use_geo_branch=True,
        geo_attention_backend=geo_attention_backend,
        geo_branch_mrope=geo_cfg.geo_branch_mrope,
        geo_token_se3_dim=geo_cfg.token_se3_dim,
        geo_pose_gta_dim=geo_cfg.pose_gta_dim,
        geo_token_se3_lambda_min=geo_cfg.lambda_min,
        geo_token_se3_lambda_max=geo_cfg.lambda_max,
    ).to(dev)
    install_geo_attention(base, wrapper)

    report = wrapper.load_geo_state(geo_state)
    wrapper.eval()
    if verbose:
        print(
            f"[dkvlm] geometry branch on layers {geo_cfg.target_layers}: "
            f"{report['loaded_qkv_tensors']} projection tensors, "
            f"{len(report['loaded_gates'])} gates"
        )
        if report["ignored_keys"]:
            print(
                f"[dkvlm] ignored {len(report['ignored_keys'])} non-geometry entries "
                f"(e.g. {report['ignored_keys'][0]})"
            )

    engine = None
    if use_geometry:
        engine = VGGTOmegaEngine(
            VGGTOmegaConfig(ckpt=geometry_ckpt, device=str(dev), max_frames=max(nframes, 32))
        )

    return DKVLM(
        wrapper,
        processor,
        geo_cfg,
        engine,
        nframes=nframes,
        max_pixels=max_pixels,
    )
