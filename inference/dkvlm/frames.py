"""Video decoding, frame sampling, and the visual-grid facts geometry alignment needs.

DK-VLM feeds a clip to the language model as N *independent images* rather than as one
video, because that is the layout the geometry aligner expects: one ``image_grid_thw`` row
per frame, one camera pose per row. The sampler here is the single source of truth for
which frames are chosen, and the same frames go to both Qwen3-VL and VGGT-Omega.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SampledFrames:
    """Decoded frames plus the indices and stable names they were drawn from."""

    frames: List[Image.Image]
    frame_indices: List[int]
    frame_names: List[str]

    def __len__(self) -> int:
        return len(self.frames)


def uniform_frame_indices(total: int, max_frames: int) -> List[int]:
    """Evenly spaced frame indices, truncating rather than rounding."""
    if total <= 0:
        return []
    k = int(max_frames)
    if k <= 0:
        raise ValueError(f"max_frames must be > 0, got {max_frames}")
    if total < k:
        return list(range(total))
    return np.linspace(0, total - 1, k).astype(int).tolist()


def frame_name_for_index(idx: int) -> str:
    return f"frame_{int(idx):06d}.png"


def sample_video_frames(video_path: str | Path, max_frames: int) -> SampledFrames:
    """Decode a video with decord and return uniformly sampled RGB frames."""
    from decord import VideoReader, cpu

    vr = VideoReader(str(Path(video_path)), ctx=cpu(0))
    indices = uniform_frame_indices(len(vr), max_frames)
    frames = [Image.fromarray(vr[int(i)].asnumpy()).convert("RGB") for i in indices]
    return SampledFrames(
        frames=frames,
        frame_indices=indices,
        frame_names=[frame_name_for_index(i) for i in indices],
    )


def load_image_frames(image_paths: Sequence[str | Path]) -> SampledFrames:
    """Build a :class:`SampledFrames` from an ordered list of image files."""
    paths = [Path(p) for p in image_paths]
    if not paths:
        raise ValueError("image_paths must not be empty")
    frames = [Image.open(p).convert("RGB") for p in paths]
    return SampledFrames(
        frames=frames,
        frame_indices=list(range(len(frames))),
        frame_names=[p.name for p in paths],
    )


def processor_visual_grid(processor) -> Tuple[int, int, int]:
    """Return ``(patch_size, merge_size, cell_size)`` from the image processor.

    Geometry alignment has to know the exact pixel footprint of one visual token, so these
    are read from the processor rather than assumed.
    """
    ip = getattr(processor, "image_processor", None)
    patch = getattr(ip, "patch_size", None)
    merge = getattr(ip, "merge_size", None)
    if patch is None or merge is None:
        raise ValueError(
            "processor.image_processor must expose patch_size and merge_size; "
            "DK-VLM will not guess the visual grid"
        )
    patch, merge = int(patch), int(merge)
    if patch <= 0 or merge <= 0:
        raise ValueError(f"invalid visual grid patch={patch}, merge={merge}")
    return patch, merge, patch * merge


def image_content_elements(
    frames: Sequence[Image.Image],
    *,
    max_pixels: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Wrap frames as independent image contents for the chat template."""
    out: List[Dict[str, Any]] = []
    for frame in frames:
        ele: Dict[str, Any] = {"type": "image", "image": frame}
        if max_pixels is not None:
            ele["max_pixels"] = int(max_pixels)
        out.append(ele)
    return out
