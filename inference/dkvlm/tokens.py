"""Place geometry onto the language model's visual tokens.

The geometry front-end gives us a world point map and camera poses at VGGT-Omega's own
resolution. The language model, meanwhile, sees each frame as a grid of merge cells at
Qwen3-VL's resolution. This module bridges the two and produces, for every visual token,
a 13-dimensional geometry feature:

    geo_coords[..., 0:3]  xyz        world position of the token's pixel
    geo_coords[..., 3:6]  t          camera translation relative to frame 0
    geo_coords[..., 6:12] rot6d      camera rotation relative to frame 0
    geo_coords[..., 12]   frame_frac i / (T - 1)

plus a boolean mask marking tokens whose geometry is trustworthy.

The mapping runs Qwen merge-cell center -> original frame pixel -> VGGT pixel, using the
crop/resize metadata recorded during preprocessing, then bilinearly samples the point
map. Getting this chain right is what keeps geometry and semantics on the same pixel.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

XYZ_DIM = 3
POSE_DIM = 10
FEATURE_DIM = XYZ_DIM + POSE_DIM
XYZ_SLICE = slice(0, 3)
POSE_SLICE = slice(3, 13)

DEFAULT_METRIC_GLOBAL_SCALE = 5.0


def token_runs(input_ids_1d: torch.Tensor, token_id: int) -> List[Tuple[int, int]]:
    """Return ``[start, end)`` spans of each contiguous run of ``token_id``."""
    positions = (input_ids_1d == token_id).nonzero(as_tuple=False).squeeze(-1).tolist()
    if not positions:
        return []
    runs: List[Tuple[int, int]] = []
    start = prev = positions[0]
    for pos in positions[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        runs.append((start, prev + 1))
        start = prev = pos
    runs.append((start, prev + 1))
    return runs


def count_expected_tokens_from_grid_thw(
    grid_thw: torch.Tensor, spatial_merge_size: int
) -> int:
    """Qwen3-VL emits ``(t*h*w) // merge**2`` placeholders for a ``(t,h,w)`` grid."""
    t, h, w = (int(grid_thw[i].item()) for i in range(3))
    merge = int(spatial_merge_size)
    if merge <= 0:
        raise ValueError(f"merge_size must be positive, got {merge}")
    if h % merge != 0 or w % merge != 0:
        raise ValueError(f"grid_h/w must be divisible by merge_size={merge}, got {(h, w)}")
    return int(t * h * w) // (merge**2)


def qwen_merge_cell_centers(
    grid_h: int,
    grid_w: int,
    *,
    spatial_merge_size: int,
    patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Row-major merge-cell centers in resized-image pixel coordinates.

    Token ``(r, c)`` covers patches ``[r*merge:(r+1)*merge, c*merge:(c+1)*merge]``, i.e. a
    square of ``merge * patch`` pixels whose center sits at ``(r + 0.5) * cell``.
    """
    merge = int(spatial_merge_size)
    patch = int(patch_size)
    if merge <= 0 or patch <= 0:
        raise ValueError(f"patch_size and merge_size must be positive, got {(patch, merge)}")
    if grid_h % merge != 0 or grid_w % merge != 0:
        raise ValueError(f"grid_h/w must be divisible by merge_size={merge}, got {(grid_h, grid_w)}")
    lh = grid_h // merge
    lw = grid_w // merge
    cell = merge * patch
    rows = torch.arange(lh, dtype=torch.float32)
    cols = torch.arange(lw, dtype=torch.float32)
    cy = ((rows + 0.5) * cell).unsqueeze(1).expand(lh, lw)
    cx = ((cols + 0.5) * cell).unsqueeze(0).expand(lh, lw)
    return cy.reshape(-1), cx.reshape(-1), lh, lw


def map_qwen_centers_to_original(
    cy_qwen: torch.Tensor,
    cx_qwen: torch.Tensor,
    *,
    qwen_h: int,
    qwen_w: int,
    orig_h: int,
    orig_w: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Invert Qwen's smart_resize, which scales the whole frame uniformly."""
    if qwen_h <= 0 or qwen_w <= 0:
        raise ValueError(f"invalid qwen size {(qwen_h, qwen_w)}")
    return (
        cy_qwen * (float(orig_h) / float(qwen_h)),
        cx_qwen * (float(orig_w) / float(qwen_w)),
    )


def map_original_to_vggt(
    oy: torch.Tensor,
    ox: torch.Tensor,
    transform: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay VGGT's crop/resize/pad on original-frame coords. Returns ``(gy, gx, inside)``."""
    crop_left = float(transform["crop_left"])
    crop_top = float(transform["crop_top"])
    crop_right = float(transform["crop_right"])
    crop_bottom = float(transform["crop_bottom"])
    crop_w = float(transform.get("crop_width", crop_right - crop_left))
    crop_h = float(transform.get("crop_height", crop_bottom - crop_top))
    resized_h = float(transform["resized_height"])
    resized_w = float(transform["resized_width"])
    pad_top = float(transform.get("pad_top", 0))
    pad_left = float(transform.get("pad_left", 0))
    final_h = float(transform["final_height"])
    final_w = float(transform["final_width"])

    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"invalid crop size in transform: {transform}")

    cy = oy - crop_top
    cx = ox - crop_left
    inside = (oy >= crop_top) & (oy < crop_bottom) & (ox >= crop_left) & (ox < crop_right)
    gy = cy * (resized_h / crop_h) + pad_top
    gx = cx * (resized_w / crop_w) + pad_left
    inside = inside & (gy >= 0) & (gy < final_h) & (gx >= 0) & (gx < final_w)
    return gy, gx, inside


def sample_points_at_coords(
    points_hw3: np.ndarray,
    valid_hw: np.ndarray,
    gy: torch.Tensor,
    gx: torch.Tensor,
    inside: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly sample XYZ (confidence-weighted) and take the mask at the nearest pixel."""
    h, w, _ = points_hw3.shape
    p = torch.from_numpy(np.asarray(points_hw3)).to(dtype=torch.float32)
    m = torch.from_numpy(np.asarray(valid_hw)).to(dtype=torch.float32)
    n = int(gy.numel())
    out = torch.zeros(n, 3, dtype=torch.float32)
    if n == 0:
        return out, torch.zeros(n, dtype=torch.bool)

    x = gx.clone()
    y = gy.clone()
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x - x0.float()) * (y1.float() - y)
    wc = (x1.float() - x) * (y - y0.float())
    wd = (x - x0.float()) * (y - y0.float())

    def _gather(yy: torch.Tensor, xx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        yy_c = yy.clamp(0, h - 1)
        xx_c = xx.clamp(0, w - 1)
        vals = p[yy_c, xx_c]
        valid = m[yy_c, xx_c] > 0.5
        valid = valid & (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
        return vals, valid

    pa, va = _gather(y0, x0)
    pb, vb = _gather(y0, x1)
    pc, vc = _gather(y1, x0)
    pd, vd = _gather(y1, x1)

    weight_sum = wa * va.float() + wb * vb.float() + wc * vc.float() + wd * vd.float()
    xyz = (
        pa * wa.unsqueeze(-1) * va.float().unsqueeze(-1)
        + pb * wb.unsqueeze(-1) * vb.float().unsqueeze(-1)
        + pc * wc.unsqueeze(-1) * vc.float().unsqueeze(-1)
        + pd * wd.unsqueeze(-1) * vd.float().unsqueeze(-1)
    )
    good = inside & (weight_sum > 1e-6)
    xyz = torch.where(
        weight_sum.unsqueeze(-1) > 1e-6,
        xyz / weight_sum.clamp_min(1e-6).unsqueeze(-1),
        torch.zeros_like(xyz),
    )

    yn = torch.round(y).long()
    xn = torch.round(x).long()
    mask_nearest = (
        (m[yn.clamp(0, h - 1), xn.clamp(0, w - 1)] > 0.5)
        & (yn >= 0)
        & (yn < h)
        & (xn >= 0)
        & (xn < w)
    )

    out[good] = xyz[good]
    out_m = good & mask_nearest
    out[~out_m] = 0
    return out, out_m


def sample_frame_tokens(
    points_hw3: np.ndarray,
    valid_hw: np.ndarray,
    transform: Dict[str, Any],
    *,
    grid_h: int,
    grid_w: int,
    spatial_merge_size: int,
    patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample one frame's merge-cell tokens through the Qwen -> original -> VGGT chain."""
    cy, cx, _lh, _lw = qwen_merge_cell_centers(
        grid_h, grid_w, spatial_merge_size=spatial_merge_size, patch_size=patch_size
    )
    oy, ox = map_qwen_centers_to_original(
        cy,
        cx,
        qwen_h=int(grid_h) * int(patch_size),
        qwen_w=int(grid_w) * int(patch_size),
        orig_h=int(transform["orig_height"]),
        orig_w=int(transform["orig_width"]),
    )
    gy, gx, inside = map_original_to_vggt(oy, ox, transform)
    if int(transform["final_height"]) != int(points_hw3.shape[0]) or int(
        transform["final_width"]
    ) != int(points_hw3.shape[1]):
        raise ValueError(
            f"transform final HxW={transform['final_height']}x{transform['final_width']} "
            f"!= points HxW={points_hw3.shape[0]}x{points_hw3.shape[1]}"
        )
    return sample_points_at_coords(points_hw3, valid_hw, gy, gx, inside)


def normalize_coords(
    coords: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    mode: str = "metric",
    clamp: Optional[float] = None,
    metric_global_scale: float = DEFAULT_METRIC_GLOBAL_SCALE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Center coordinates on their median, then rescale.

    ``metric`` divides by a fixed global scale so absolute room size survives; ``robust``
    divides by the per-clip P95 radius, which discards it. The released checkpoints were
    trained and evaluated with ``metric``.
    """
    mode = (mode or "metric").strip().lower()
    n = int(coords.shape[0])
    if n == 0:
        return torch.zeros(0, 3, device=coords.device, dtype=coords.dtype), valid_mask

    m = valid_mask & torch.isfinite(coords).all(dim=-1)
    if m.sum() < 3:
        return torch.zeros_like(coords), torch.zeros(n, device=coords.device, dtype=torch.bool)
    c = coords[m].to(dtype=torch.float32)
    center = c.median(dim=0).values

    if mode in ("metric", "metric_global", "absolute"):
        limit = 10.0 if clamp is None else float(clamp)
        scale = max(float(metric_global_scale), 1e-6)
    elif mode in ("robust", "legacy", "p95"):
        limit = 2.0 if clamp is None else float(clamp)
        scale = float(torch.quantile((c - center).norm(dim=-1), 0.95))
        if scale < 1e-6:
            return torch.zeros_like(coords), torch.zeros(n, device=coords.device, dtype=torch.bool)
    else:
        raise ValueError(f"unknown geometry normalize mode: {mode!r}")

    out = (coords - center) / scale
    if limit > 0:
        out = out.clamp(-limit, limit)
    out = torch.where(m.unsqueeze(-1), out, torch.zeros_like(out))
    return out.to(dtype=coords.dtype), m


def build_xyz_sequence(
    points: np.ndarray,
    valid_mask: np.ndarray,
    input_ids_1d: torch.Tensor,
    image_grid_thw: torch.Tensor,
    frame_transforms: Sequence[Dict[str, Any]],
    *,
    spatial_merge_size: int,
    image_token_id: int,
    patch_size: int,
    normalize_mode: str = "metric",
    metric_global_scale: float = DEFAULT_METRIC_GLOBAL_SCALE,
    sample_id: str = "",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build ``(L, 3)`` world coordinates aligned to image tokens, plus a validity mask."""
    seq_len = int(input_ids_1d.shape[0])
    geo_coords = torch.zeros(seq_len, XYZ_DIM, dtype=torch.float32)
    geo_mask = torch.zeros(seq_len, dtype=torch.bool)

    points = np.asarray(points)
    valid_mask = np.asarray(valid_mask)
    if points.ndim != 4 or valid_mask.ndim != 3:
        raise ValueError(f"unexpected points/mask shape: {points.shape}, {valid_mask.shape}")
    if patch_size <= 0 or spatial_merge_size <= 0:
        raise ValueError(f"invalid visual grid patch={patch_size}, merge={spatial_merge_size}")

    runs = token_runs(input_ids_1d, image_token_id)
    n_img = int(image_grid_thw.shape[0])
    if len(runs) != n_img:
        raise ValueError(
            f"image token runs {len(runs)} != image_grid rows {n_img} (sample={sample_id})"
        )
    if len(runs) != int(points.shape[0]):
        raise ValueError(
            f"image token runs {len(runs)} != geometry frames {points.shape[0]} "
            f"(sample={sample_id}); refusing silent pad/clip alignment"
        )
    if len(frame_transforms) != int(points.shape[0]):
        raise ValueError(
            f"frame_transforms={len(frame_transforms)} != geometry frames={points.shape[0]} "
            f"(sample={sample_id})"
        )

    per_run_coords: List[torch.Tensor] = []
    per_run_masks: List[torch.Tensor] = []
    for i, (s, e) in enumerate(runs):
        ig = image_grid_thw[i]
        expected = count_expected_tokens_from_grid_thw(ig, spatial_merge_size)
        if (e - s) != expected:
            raise ValueError(
                f"image token count {e - s} != expected {expected} (run {i}, sample={sample_id})"
            )
        t, h, w = (int(ig[j].item()) for j in range(3))
        if t != 1:
            raise ValueError(
                f"the 32-image protocol expects per-image grid t=1, got t={t} "
                f"(run {i}, sample={sample_id})"
            )
        coords_i, mask_i = sample_frame_tokens(
            points[i],
            valid_mask[i],
            dict(frame_transforms[i]),
            grid_h=h,
            grid_w=w,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
        )
        if coords_i.shape[0] != expected:
            raise ValueError(
                f"sampled {coords_i.shape[0]} tokens != expected {expected} "
                f"(run {i}, sample={sample_id})"
            )
        per_run_coords.append(coords_i)
        per_run_masks.append(mask_i)

    all_norm, all_ok = normalize_coords(
        torch.cat(per_run_coords, dim=0),
        torch.cat(per_run_masks, dim=0),
        mode=normalize_mode,
        metric_global_scale=metric_global_scale,
    )
    offset = 0
    for (s, e), coords_i in zip(runs, per_run_coords):
        n = int(coords_i.shape[0])
        geo_coords[s:e, :] = all_norm[offset : offset + n].to(geo_coords.dtype)
        geo_mask[s:e] = all_ok[offset : offset + n]
        offset += n
    return geo_coords, geo_mask


def camera_poses_to_relative_features(
    poses: np.ndarray,
    ref_idx: int = 0,
    metric_global_scale: float = 1.0,
) -> np.ndarray:
    """Express camera-to-world poses ``(T,4,4)`` relative to a reference frame.

    Returns ``(T, 10)``: translation, the first two columns of the rotation matrix
    (a 6-D rotation representation), and the frame's fractional position in the clip.
    """
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"expected poses (T,4,4), got {poses.shape}")
    metric_global_scale = float(metric_global_scale)
    if metric_global_scale <= 0:
        raise ValueError(f"metric_global_scale must be positive, got {metric_global_scale}")
    T = int(poses.shape[0])
    ref_idx = int(np.clip(ref_idx, 0, max(T - 1, 0)))
    R0 = poses[ref_idx, :3, :3]
    t0 = poses[ref_idx, :3, 3]
    feats = np.zeros((T, POSE_DIM), dtype=np.float32)
    denom = float(max(T - 1, 1))
    for i in range(T):
        R_rel = R0.T @ poses[i, :3, :3]
        feats[i, 0:3] = (R0.T @ (poses[i, :3, 3] - t0)) / metric_global_scale
        feats[i, 3:9] = R_rel[:, :2].reshape(-1).astype(np.float32)
        feats[i, 9] = float(i) / denom
    return feats


def build_pose_sequence(
    camera_poses: np.ndarray,
    input_ids_1d: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
    image_token_id: int,
    metric_global_scale: float = DEFAULT_METRIC_GLOBAL_SCALE,
    sample_id: str = "",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Broadcast each frame's pose feature across that frame's visual tokens."""
    seq_len = int(input_ids_1d.shape[0])
    geo_coords = torch.zeros(seq_len, POSE_DIM, dtype=torch.float32)
    geo_mask = torch.zeros(seq_len, dtype=torch.bool)

    runs = token_runs(input_ids_1d, image_token_id)
    n_img = int(image_grid_thw.shape[0])
    if len(runs) != n_img:
        raise ValueError(f"image runs {len(runs)} != grid rows {n_img} (sample={sample_id})")
    poses = np.asarray(camera_poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"expected poses (T,4,4), got {poses.shape}")
    if int(poses.shape[0]) != n_img:
        raise ValueError(
            f"poses T={poses.shape[0]} != image runs={n_img} (sample={sample_id}); "
            "geometry frames and sampled frames must correspond one to one"
        )

    frame_feats = camera_poses_to_relative_features(
        poses, metric_global_scale=metric_global_scale
    )
    for i, (s, e) in enumerate(runs):
        ig = image_grid_thw[i]
        expected = count_expected_tokens_from_grid_thw(ig, spatial_merge_size)
        if (e - s) != expected:
            raise ValueError(
                f"image tokens {e - s} != expected {expected} (run {i}, sample={sample_id})"
            )
        if int(ig[0].item()) != 1:
            raise ValueError(
                f"the 32-image protocol expects per-image grid t=1, got t={int(ig[0].item())} "
                f"(run {i}, sample={sample_id})"
            )
        geo_coords[s:e, :] = torch.from_numpy(frame_feats[i])
        geo_mask[s:e] = True
    return geo_coords, geo_mask


def combine_xyz_pose(
    xyz_coords: torch.Tensor,
    xyz_mask: torch.Tensor,
    pose_coords: torch.Tensor,
    pose_mask: torch.Tensor,
    *,
    visual_mask: torch.Tensor,
    sample_id: str = "",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Concatenate XYZ and pose into ``(L, 13)`` and intersect their validity masks."""
    if xyz_coords.shape[-1] != XYZ_DIM or pose_coords.shape[-1] != POSE_DIM:
        raise ValueError(
            f"expected xyz (L,{XYZ_DIM}) and pose (L,{POSE_DIM}), got "
            f"{tuple(xyz_coords.shape)} and {tuple(pose_coords.shape)} (sample={sample_id})"
        )
    if xyz_coords.shape[0] != pose_coords.shape[0]:
        raise ValueError(
            f"length mismatch: xyz L={xyz_coords.shape[0]} pose L={pose_coords.shape[0]} "
            f"(sample={sample_id})"
        )

    L = int(xyz_coords.shape[0])
    xyz_b = xyz_mask.bool()
    pose_b = pose_mask.bool()
    visual_b = visual_mask.bool()

    if bool((xyz_b & ~visual_b).any().item()) or bool((pose_b & ~visual_b).any().item()):
        raise ValueError(f"geometry mask is set outside visual tokens (sample={sample_id})")
    if bool(visual_b.any().item()) and not bool(pose_b[visual_b].all().item()):
        n_miss = int((visual_b & ~pose_b).sum().item())
        raise ValueError(
            f"pose must cover every visual token; missing={n_miss} (sample={sample_id})"
        )

    joint = xyz_b & pose_b
    out = torch.zeros(L, FEATURE_DIM, dtype=torch.float32)
    out[:, XYZ_SLICE] = xyz_coords.to(dtype=torch.float32)
    out[:, POSE_SLICE] = pose_coords.to(dtype=torch.float32)

    visual_n = int(visual_b.sum().item())
    joint_n = int(joint.sum().item())
    stats = {
        "visual": visual_n,
        "xyz_valid": int(xyz_b.sum().item()),
        "pose_valid": int(pose_b.sum().item()),
        "joint_valid": joint_n,
        "joint_invalid": visual_n - joint_n,
        "valid_ratio": (joint_n / visual_n) if visual_n else 0.0,
    }
    return out, joint, stats


def build_geo_coords(
    bundle,
    input_ids_1d: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
    patch_size: int,
    image_token_id: int,
    video_token_id: int,
    normalize_mode: str = "metric",
    metric_global_scale: float = DEFAULT_METRIC_GLOBAL_SCALE,
    sample_id: str = "",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Turn a :class:`dkvlm.geometry.GeometryBundle` into ``(L,13)`` features and a mask."""
    if image_grid_thw is None or int(image_grid_thw.numel()) == 0:
        raise ValueError(
            f"expected image_grid_thw from the 32-image protocol (sample={sample_id}); "
            "DK-VLM inference does not use the single-video token layout"
        )

    xyz_coords, xyz_mask = build_xyz_sequence(
        bundle.points,
        bundle.valid_mask,
        input_ids_1d,
        image_grid_thw,
        bundle.frame_transforms,
        spatial_merge_size=spatial_merge_size,
        image_token_id=image_token_id,
        patch_size=patch_size,
        normalize_mode=normalize_mode,
        metric_global_scale=metric_global_scale,
        sample_id=sample_id,
    )
    pose_coords, pose_mask = build_pose_sequence(
        bundle.camera_poses,
        input_ids_1d,
        image_grid_thw,
        spatial_merge_size=spatial_merge_size,
        image_token_id=image_token_id,
        metric_global_scale=metric_global_scale,
        sample_id=sample_id,
    )
    visual = (input_ids_1d == image_token_id) | (input_ids_1d == video_token_id)
    return combine_xyz_pose(
        xyz_coords,
        xyz_mask,
        pose_coords,
        pose_mask,
        visual_mask=visual,
        sample_id=sample_id,
    )
