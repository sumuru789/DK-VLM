# VGGT-Omega preprocessing portions adapted from vggt_omega/utils/load_fn.py.
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Those portions retain the FAIR Noncommercial Research License; see
# ../../../licenses/VGGT-Omega.txt and ../../../THIRD_PARTY.md.
"""Geometry front-end: video frames -> per-pixel 3D points + camera poses.

DK-VLM reads geometry from VGGT-Omega, a feed-forward 3D reconstruction model. This
module owns the whole "pixels in, geometry out" stage:

    PIL frames -> crop/resize/pad (CPU) -> VGGT-Omega forward (GPU)
               -> unproject depth to a world point map + confidence mask
               -> camera-to-world poses

The output :class:`GeometryBundle` carries everything ``dkvlm.tokens`` needs to place
geometry on the visual tokens of the language model, including the per-frame
``frame_transforms`` that record exactly how each frame was cropped and resized.

VGGT-Omega itself is an upstream dependency and is not vendored here. Clone it and
point ``VGGT_OMEGA_ROOT`` at the checkout (see README).
"""

from __future__ import annotations

import os
import sys
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF

DEFAULT_IMAGE_RESOLUTION = 512
DEFAULT_VGGT_PATCH_SIZE = 16
DEFAULT_CONF_PERCENTILE = 20.0


def vggt_omega_root() -> Path:
    """Locate the upstream VGGT-Omega checkout and put it on ``sys.path``."""
    raw = os.environ.get("VGGT_OMEGA_ROOT", "").strip()
    if not raw:
        raise RuntimeError(
            "VGGT_OMEGA_ROOT is not set. Clone VGGT-Omega and export "
            "VGGT_OMEGA_ROOT=/path/to/vggt-omega (see README)."
        )
    root = Path(raw).expanduser().resolve()
    if not (root / "vggt_omega").is_dir():
        raise RuntimeError(
            f"VGGT_OMEGA_ROOT={root} does not look like a VGGT-Omega checkout "
            "(no vggt_omega/ package inside)."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _crop_to_supported_aspect_ratio_with_box(
    image: Image.Image,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    width, height = image.size
    aspect_ratio = height / max(width, 1)
    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        box = (left, 0, left + crop_width, height)
        return image.crop(box), box
    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        box = (0, top, width, top + crop_height)
        return image.crop(box), box
    box = (0, 0, width, height)
    return image, box


def _round_to_patch_multiple(value: float, patch_size: int) -> int:
    return max(patch_size, int(np.round(float(value) / patch_size)) * patch_size)


def _balanced_target_shape(
    aspect_ratio: float, image_resolution: int, patch_size: int
) -> Tuple[int, int]:
    token_number = (image_resolution // patch_size) ** 2
    w_patches = np.sqrt(token_number / aspect_ratio)
    h_patches = token_number / w_patches
    w_patches = max(1, int(np.round(w_patches)))
    h_patches = max(1, int(np.round(h_patches)))
    return h_patches * patch_size, w_patches * patch_size


def _max_size_target_shape(
    aspect_ratio: float, image_resolution: int, patch_size: int
) -> Tuple[int, int]:
    if aspect_ratio >= 1.0:
        height = image_resolution
        width = _round_to_patch_multiple(image_resolution / aspect_ratio, patch_size)
    else:
        width = image_resolution
        height = _round_to_patch_multiple(image_resolution * aspect_ratio, patch_size)
    return height, width


def _pad_images_to_common_size_with_meta(
    images: Sequence[torch.Tensor], shapes
) -> Tuple[List[torch.Tensor], List[Dict[str, int]]]:
    max_height = max(shape[0] for shape in shapes)
    max_width = max(shape[1] for shape in shapes)
    padded_images: List[torch.Tensor] = []
    pad_meta: List[Dict[str, int]] = []
    for image in images:
        h_padding = max_height - image.shape[1]
        w_padding = max_width - image.shape[2]
        pad_top = h_padding // 2
        pad_bottom = h_padding - pad_top
        pad_left = w_padding // 2
        pad_right = w_padding - pad_left
        if h_padding > 0 or w_padding > 0:
            image = torch.nn.functional.pad(
                image,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=1.0,
            )
        padded_images.append(image)
        pad_meta.append(
            {
                "pad_top": int(pad_top),
                "pad_left": int(pad_left),
                "pad_bottom": int(pad_bottom),
                "pad_right": int(pad_right),
                "final_height": int(max_height),
                "final_width": int(max_width),
            }
        )
    return padded_images, pad_meta


def preprocess_images_cpu(
    images: Sequence[Union[str, Image.Image]],
    *,
    mode: str = "balanced",
    image_resolution: int = DEFAULT_IMAGE_RESOLUTION,
    patch_size: int = DEFAULT_VGGT_PATCH_SIZE,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Crop/resize/pad frames on CPU, returning the batch plus per-frame transforms.

    Pixel path matches official ``vggt_omega.utils.load_fn.load_and_preprocess_images``.
    Official VGGT-Omega does not return crop/pad boxes, so the invert metadata lives
    here; that is what lets us sample geometry at the right pixel for each visual token.
    """
    if len(images) == 0:
        raise ValueError("At least 1 image is required")
    if mode not in ("balanced", "max_size"):
        raise ValueError("Mode must be either 'balanced' or 'max_size'")
    if image_resolution <= 0 or patch_size <= 0:
        raise ValueError("image_resolution/patch_size must be positive")
    if image_resolution % patch_size != 0:
        raise ValueError("image_resolution must be divisible by patch_size")

    tensors: List[torch.Tensor] = []
    transforms: List[Dict[str, Any]] = []
    shapes = set()
    to_tensor = TF.ToTensor()

    for item in images:
        if isinstance(item, Image.Image):
            rgb = item.convert("RGB")
        elif isinstance(item, str):
            with Image.open(item) as image:
                if image.mode == "RGBA":
                    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                    image = Image.alpha_composite(background, image)
                rgb = image.convert("RGB")
        else:
            raise TypeError(f"unsupported image type: {type(item)}")

        orig_w, orig_h = rgb.size
        cropped, crop_box = _crop_to_supported_aspect_ratio_with_box(rgb)
        crop_w, crop_h = cropped.size
        aspect_ratio = crop_h / max(crop_w, 1)

        if mode == "balanced":
            target_h, target_w = _balanced_target_shape(
                aspect_ratio, image_resolution, patch_size
            )
        else:
            target_h, target_w = _max_size_target_shape(
                aspect_ratio, image_resolution, patch_size
            )

        resized = cropped.resize((target_w, target_h), Image.Resampling.BICUBIC)
        tensor = to_tensor(resized)
        shapes.add((tensor.shape[1], tensor.shape[2]))
        tensors.append(tensor)
        transforms.append(
            {
                "orig_height": int(orig_h),
                "orig_width": int(orig_w),
                "crop_left": int(crop_box[0]),
                "crop_top": int(crop_box[1]),
                "crop_right": int(crop_box[2]),
                "crop_bottom": int(crop_box[3]),
                "crop_height": int(crop_h),
                "crop_width": int(crop_w),
                "resized_height": int(target_h),
                "resized_width": int(target_w),
                "pad_top": 0,
                "pad_left": 0,
                "pad_bottom": 0,
                "pad_right": 0,
                "final_height": int(target_h),
                "final_width": int(target_w),
                "preprocess_mode": str(mode),
                "image_resolution": int(image_resolution),
                "vggt_patch_size": int(patch_size),
            }
        )

    if len(shapes) > 1:
        warnings.warn(
            f"Found images with different shapes: {shapes}; padding to a common size.",
            stacklevel=2,
        )
        tensors, pad_meta = _pad_images_to_common_size_with_meta(tensors, shapes)
        for tfm, pad in zip(transforms, pad_meta):
            tfm["pad_top"] = int(pad["pad_top"])
            tfm["pad_left"] = int(pad["pad_left"])
            tfm["pad_bottom"] = int(pad["pad_bottom"])
            tfm["pad_right"] = int(pad["pad_right"])
            tfm["final_height"] = int(pad["final_height"])
            tfm["final_width"] = int(pad["final_width"])

    return torch.stack(tensors).contiguous().cpu(), transforms


@torch.inference_mode()
def unproject_depth_to_points_torch(
    depth_map: torch.Tensor,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
) -> torch.Tensor:
    """Unproject depth (S,H,W,1) with w2c extrinsic (S,3,4) and K (S,3,3) -> (S,H,W,3)."""
    depth = depth_map[..., 0]
    s, h, w = depth.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    cam = torch.stack(
        [(xx[None] - cx) / fx * depth, (yy[None] - cy) / fy * depth, depth],
        dim=-1,
    )
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3]
    return torch.einsum(
        "sij,shwj->shwi",
        rot.transpose(1, 2),
        cam - trans[:, None, None, :],
    )


def linear_quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    """``torch.quantile`` without its 2**24 element cap.

    Above that size aten refuses the input, which a 64-frame confidence map
    exceeds, so fall back to a sort and reproduce the linear interpolation.
    """
    flat = values.reshape(-1)
    if flat.numel() <= 2**24:
        return torch.quantile(flat, q)
    ordered = flat.sort().values
    pos = q * (ordered.numel() - 1)
    lower = int(pos)
    upper = min(lower + 1, ordered.numel() - 1)
    frac = pos - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * frac


@torch.inference_mode()
def build_valid_mask_torch(
    depth_conf: torch.Tensor,
    conf_percentile: float,
) -> torch.Tensor:
    """Drop the lowest-confidence ``conf_percentile`` percent of depths."""
    conf = depth_conf.float()
    mask = torch.isfinite(conf)
    if conf_percentile > 0 and bool(mask.any()):
        th = linear_quantile(conf[mask].float(), float(conf_percentile) / 100.0)
        mask = mask & (conf >= th)
    mask = mask & (conf > 1e-5)
    return mask.to(torch.uint8)


def unproject_depth_to_points_numpy(
    depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray
) -> np.ndarray:
    """CPU twin of :func:`unproject_depth_to_points_torch`."""
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))
    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    camera_points = np.stack(
        [(x - cx) / fx * depth, (y - cy) / fy * depth, depth],
        axis=-1,
    )
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def build_valid_mask_numpy(depth_conf: np.ndarray, conf_percentile: float) -> np.ndarray:
    """CPU twin of :func:`build_valid_mask_torch`."""
    conf = depth_conf.astype(np.float32)
    mask = np.isfinite(conf)
    if conf_percentile > 0 and np.any(mask):
        th = np.percentile(conf[mask], conf_percentile)
        mask &= conf >= th
    mask &= conf > 1e-5
    return mask.astype(np.uint8)


def w2c_to_c2w(extrinsic_w2c: np.ndarray) -> np.ndarray:
    """Invert world-to-camera (S,3,4) into camera-to-world (S,4,4)."""
    vggt_omega_root()
    from vggt_omega.utils.geometry import closed_form_inverse_se3

    if extrinsic_w2c.shape[-2:] != (3, 4):
        raise ValueError(f"unexpected extrinsic shape: {extrinsic_w2c.shape}")
    return closed_form_inverse_se3(extrinsic_w2c.astype(np.float64)).astype(np.float32)


@dataclass
class VGGTOmegaConfig:
    """Settings for the geometry front-end. Defaults match the released checkpoints."""

    ckpt: str = ""
    device: str = "cuda"
    variant: str = "512"
    max_frames: int = 32
    preprocess_mode: str = "balanced"
    conf_percentile: float = DEFAULT_CONF_PERCENTILE
    gpu_postprocess: bool = True

    def resolved_ckpt(self) -> Path:
        raw = str(self.ckpt or os.environ.get("VGGT_OMEGA_CKPT", "")).strip()
        if not raw:
            raise RuntimeError(
                "VGGT-Omega checkpoint not set. Pass ckpt=... or export VGGT_OMEGA_CKPT."
            )
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {path}")
        return path


@dataclass
class GeometryBundle:
    """In-memory geometry for one clip, ready for token alignment."""

    points: np.ndarray
    valid_mask: np.ndarray
    camera_poses: np.ndarray
    frame_transforms: List[Dict[str, Any]]
    source_frame_names: List[str]
    sample_id: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


def _variant_spec(variant: str, ckpt: Path) -> Tuple[str, int, bool]:
    """Resolve (name, image_resolution, enable_alignment) from the checkpoint name."""
    name = ckpt.name.lower()
    if "256" in name and "text" in name:
        return "256_text", 256, True
    if "512" in name:
        return "512", 512, False
    if variant == "256_text":
        return "256_text", 256, True
    if variant in ("", "512"):
        return "512", 512, False
    raise ValueError(f"cannot infer VGGT-Omega variant from {ckpt}; pass variant=...")


class VGGTOmegaEngine:
    """Loads VGGT-Omega once, then produces a :class:`GeometryBundle` per clip."""

    def __init__(self, cfg: Optional[VGGTOmegaConfig] = None):
        self.cfg = cfg or VGGTOmegaConfig()
        self._lock = threading.Lock()
        self._model = None
        self._variant = ""
        self._ckpt_path: Optional[Path] = None
        self._image_resolution = DEFAULT_IMAGE_RESOLUTION
        self._device = torch.device(self.cfg.device)

    def _lazy_init(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            vggt_omega_root()
            from vggt_omega.models import VGGTOmega

            ckpt = self.cfg.resolved_ckpt()
            var_name, image_resolution, enable_alignment = _variant_spec(
                self.cfg.variant, ckpt
            )
            model = VGGTOmega(enable_alignment=enable_alignment).to(self._device).eval()
            model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
            for p in model.parameters():
                p.requires_grad_(False)
            self._model = model
            self._variant = var_name
            self._ckpt_path = ckpt
            self._image_resolution = int(image_resolution)

    @torch.inference_mode()
    def infer(
        self,
        *,
        pil_images: Optional[Sequence[Image.Image]] = None,
        image_paths: Optional[Sequence[str]] = None,
        images: Optional[torch.Tensor] = None,
        frame_transforms: Optional[Sequence[Dict[str, Any]]] = None,
        source_frame_names: Optional[Sequence[str]] = None,
        sample_id: str = "",
    ) -> GeometryBundle:
        """Run VGGT-Omega on one clip.

        Supply either decoded ``pil_images`` / ``image_paths`` (this method then does the
        CPU preprocess), or a preprocessed ``images`` batch together with its
        ``frame_transforms``.
        """
        self._lazy_init()
        assert self._model is not None

        vggt_omega_root()
        from vggt_omega.utils.pose_enc import encoding_to_camera

        if images is None:
            if pil_images is not None:
                source = list(pil_images)
                default_names = [f"frame_{i:04d}.jpg" for i in range(len(source))]
            elif image_paths is not None:
                source = [str(p) for p in image_paths]
                default_names = [Path(p).name for p in source]
            else:
                raise ValueError("infer() needs pil_images, image_paths, or images")
            images, frame_transforms = preprocess_images_cpu(
                source,
                mode=str(self.cfg.preprocess_mode),
                image_resolution=int(self._image_resolution),
            )
            if source_frame_names is None:
                source_frame_names = default_names
        elif frame_transforms is None:
            raise ValueError("frame_transforms is required when passing images directly")

        n_frames = int(images.shape[0])
        if source_frame_names is None:
            source_frame_names = [f"frame_{i:04d}.jpg" for i in range(n_frames)]
        source_frame_names = [str(n) for n in source_frame_names]
        if len(source_frame_names) != n_frames:
            raise ValueError(f"source_frame_names={len(source_frame_names)} != T={n_frames}")
        if len(frame_transforms) != n_frames:
            raise ValueError(f"frame_transforms={len(frame_transforms)} != T={n_frames}")
        if n_frames > int(self.cfg.max_frames):
            raise ValueError(f"T={n_frames} exceeds max_frames={self.cfg.max_frames}")

        images_dev = images.to(self._device, non_blocking=False)
        predictions = self._model(images_dev)
        extrinsic, intrinsic = encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:]
        )
        depth_t = predictions["depth"].detach().float()
        depth_conf_t = predictions["depth_conf"].detach().float()
        extrinsic_t = extrinsic.detach().float()
        intrinsic_t = intrinsic.detach().float()
        if extrinsic_t.shape[0] == 1:
            extrinsic_t = extrinsic_t[0]
            intrinsic_t = intrinsic_t[0]
            depth_t = depth_t[0]
            depth_conf_t = depth_conf_t[0]

        if self.cfg.gpu_postprocess and self._device.type == "cuda":
            pts_t = unproject_depth_to_points_torch(depth_t, extrinsic_t, intrinsic_t)
            mask_t = build_valid_mask_torch(depth_conf_t, float(self.cfg.conf_percentile))
            points = pts_t.half().cpu().numpy()
            valid_mask = mask_t.cpu().numpy()
            extrinsic_np = extrinsic_t.cpu().numpy()
            post_backend = "gpu_torch"
        else:
            extrinsic_np = extrinsic_t.cpu().numpy()
            points = unproject_depth_to_points_numpy(
                depth_t.cpu().numpy(), extrinsic_np, intrinsic_t.cpu().numpy()
            ).astype(np.float16)
            valid_mask = build_valid_mask_numpy(
                depth_conf_t.cpu().numpy(), float(self.cfg.conf_percentile)
            )
            post_backend = "cpu_numpy"

        camera_poses = w2c_to_c2w(extrinsic_np).astype(np.float32)
        del images_dev, predictions

        return GeometryBundle(
            points=points,
            valid_mask=valid_mask,
            camera_poses=camera_poses,
            frame_transforms=list(frame_transforms),
            source_frame_names=list(source_frame_names),
            sample_id=str(sample_id or ""),
            meta={
                "variant": self._variant,
                "ckpt": str(self._ckpt_path or ""),
                "device": str(self._device),
                "post_backend": post_backend,
            },
        )
