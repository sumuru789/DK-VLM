"""Building blocks of the geometry branch.

Three pieces live here:

``GeoBranchRuntime``
    A per-forward slot holding the geometry tensors. The wrapper fills it before calling
    the base model; the patched attention layers read it. Passing geometry this way keeps
    the stock Qwen3-VL forward signature untouched.

``GeoBranchQKV``
    Per-layer Q/K/V/O projections for the geometry branch, cloned at construction time
    from the corresponding semantic attention so the branch starts as a copy of it.

``TokenSE3PoseGTA``
    The geometric transform itself. It applies two independent SE(3) actions to disjoint
    slices of each attention head, so that attention logits depend on *relative* geometry
    between tokens rather than absolute coordinates.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

GEO_FEATURE_DIM = 13


@dataclass
class GeoBranchRuntime:
    """Geometry for the forward pass currently in flight."""

    geo_coords: Optional[torch.Tensor] = None
    geo_coords_mask: Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None
    visual_token_mask: Optional[torch.Tensor] = None

    def clear(self) -> None:
        self.geo_coords = None
        self.geo_coords_mask = None
        self.input_ids = None
        self.visual_token_mask = None


_RUNTIME = GeoBranchRuntime()


def geo_runtime_get() -> GeoBranchRuntime:
    return _RUNTIME


def geo_runtime_set(
    *,
    geo_coords: Optional[torch.Tensor],
    geo_coords_mask: Optional[torch.Tensor],
    input_ids: Optional[torch.Tensor],
    image_token_id: int,
    video_token_id: int,
) -> None:
    _RUNTIME.geo_coords = geo_coords
    _RUNTIME.geo_coords_mask = geo_coords_mask
    _RUNTIME.input_ids = input_ids
    if input_ids is not None:
        _RUNTIME.visual_token_mask = (input_ids == image_token_id) | (
            input_ids == video_token_id
        )
    else:
        _RUNTIME.visual_token_mask = None


def geo_runtime_clear() -> None:
    _RUNTIME.clear()


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Recover a rotation matrix from its first two columns via Gram-Schmidt."""
    x = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
    a1, a2 = x[..., 0], x[..., 1]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a batch of 4x4 rigid transforms."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def camera_to_world_from_pose_feats(feats: torch.Tensor) -> torch.Tensor:
    """Build camera-to-world 4x4 matrices from ``(t, rot6d, ...)`` pose features."""
    eye = torch.zeros(*feats.shape[:-1], 4, 4, device=feats.device, dtype=feats.dtype)
    eye[..., :3, :3] = rot6d_to_matrix(feats[..., 3:9])
    eye[..., :3, 3] = feats[..., :3]
    eye[..., 3, 3] = 1.0
    return eye


def world_to_camera_from_pose_feats(feats: torch.Tensor) -> torch.Tensor:
    return invert_se3(camera_to_world_from_pose_feats(feats))


def token_frame_ids(pose_feats: torch.Tensor) -> torch.Tensor:
    """Group tokens into contiguous frame ids using the ``frame_frac`` channel."""
    n = pose_feats.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.long, device=pose_feats.device)
    keys = pose_feats[:, 9] if pose_feats.shape[-1] >= 10 else pose_feats[:, :9].sum(dim=-1)
    id_map: dict = {}
    out = torch.empty(n, dtype=torch.long, device=pose_feats.device)
    for i in range(n):
        k = float(keys[i].item())
        if k not in id_map:
            id_map[k] = len(id_map)
        out[i] = id_map[k]
    return out


def unique_frame_pose_and_ids(
    pose_feats: torch.Tensor,
    token_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-frame world-to-camera matrices and each selected token's frame id."""
    sel = pose_feats[token_index]
    frame_ids = token_frame_ids(sel)
    t_max = int(frame_ids.max().item()) + 1 if frame_ids.numel() else 0
    uniq_feats = []
    for t in range(t_max):
        idx = (frame_ids == t).nonzero(as_tuple=False)[0, 0]
        uniq_feats.append(sel[idx])
        same = sel[frame_ids == t]
        if not torch.allclose(same[:, :9], same[:1, :9], atol=1e-4, rtol=1e-3):
            raise ValueError(f"inconsistent pose within frame_id={t}")
    P = world_to_camera_from_pose_feats(torch.stack(uniq_feats, 0).to(dtype=torch.float32))
    return P, frame_ids


def _clone_linear(src: nn.Module) -> nn.Linear:
    base = src
    if hasattr(src, "base_layer"):
        base = src.base_layer
    elif hasattr(src, "orig_module"):
        base = src.orig_module
    if not isinstance(base, nn.Linear):
        raise TypeError(f"expected nn.Linear for geo branch init, got {type(src)!r}")
    dst = nn.Linear(
        base.in_features,
        base.out_features,
        bias=base.bias is not None,
        device=base.weight.device,
        dtype=base.weight.dtype,
    )
    dst.load_state_dict(base.state_dict())
    return dst


def _clone_optional(attn_module: nn.Module, name: str) -> Optional[nn.Module]:
    src = getattr(attn_module, name, None)
    return None if src is None else copy.deepcopy(src)


class GeoBranchQKV(nn.Module):
    """Independent Q/K/V/O projections for the geometry branch of one layer."""

    def __init__(self, attn_module: nn.Module, use_geo_o_proj: bool = True):
        super().__init__()
        self.head_dim = int(attn_module.head_dim)
        self.use_geo_o_proj = bool(use_geo_o_proj)
        self.q_proj = _clone_linear(attn_module.q_proj)
        self.k_proj = _clone_linear(attn_module.k_proj)
        self.v_proj = _clone_linear(attn_module.v_proj)
        self.q_norm = _clone_optional(attn_module, "q_norm")
        self.k_norm = _clone_optional(attn_module, "k_norm")
        if self.use_geo_o_proj and hasattr(attn_module, "o_proj"):
            self.o_proj = _clone_linear(attn_module.o_proj)
        else:
            self.o_proj = None

    def project(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return query/key ``(B, H, L, D)`` and value ``(B, H_kv, L, D)``."""
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
        value_states = self.v_proj(hidden_states)
        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
        if self.k_norm is not None:
            key_states = self.k_norm(key_states)
        return (
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2),
        )


def geometric_lambdas(
    n_blocks: int,
    lam_min: float,
    lam_max: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Geometric series covering ``[lam_min, lam_max]`` inclusive."""
    if n_blocks <= 0:
        raise ValueError(f"n_blocks must be positive, got {n_blocks}")
    if lam_min <= 0 or lam_max <= 0:
        raise ValueError(f"lambda range must be positive, got [{lam_min}, {lam_max}]")
    if lam_max < lam_min:
        raise ValueError(f"lambda_max {lam_max} < lambda_min {lam_min}")
    if n_blocks == 1:
        return torch.tensor([float(lam_min)], dtype=dtype, device=device)
    ratio = (float(lam_max) / float(lam_min)) ** (1.0 / float(n_blocks - 1))
    return float(lam_min) * (ratio ** torch.arange(n_blocks, dtype=dtype, device=device))


def apply_tiled_4x4_band(
    feats: torch.Tensor,
    mats: torch.Tensor,
    dim_start: int,
    dim_end: int,
) -> torch.Tensor:
    """Treat ``feats[..., dim_start:dim_end]`` as homogeneous 4-vectors and transform them.

    ``feats`` is ``[B, H, N, D]``. ``mats`` is either ``[N, 4, 4]`` (shared across blocks)
    or ``[N, nblocks, 4, 4]`` (one matrix per 4-D block).
    """
    if dim_end < dim_start:
        raise ValueError(f"empty/negative band [{dim_start}:{dim_end}]")
    band = dim_end - dim_start
    if band == 0:
        return feats
    if band % 4 != 0:
        raise ValueError(f"band width must be %4==0, got {band} for [{dim_start}:{dim_end}]")
    if feats.shape[-1] < dim_end:
        raise ValueError(f"feat dim {feats.shape[-1]} < dim_end {dim_end}")
    b, h, n, _ = feats.shape
    nblocks = band // 4
    x = feats[..., dim_start:dim_end].reshape(b, h, n, nblocks, 4)
    if mats.dim() == 3:
        if mats.shape != (n, 4, 4):
            raise ValueError(f"shared mats shape {tuple(mats.shape)} != ({n},4,4)")
        y = torch.einsum("nij,bhnkj->bhnki", mats.to(dtype=x.dtype), x)
    elif mats.dim() == 4:
        if mats.shape != (n, nblocks, 4, 4):
            raise ValueError(f"per-block mats shape {tuple(mats.shape)} != ({n},{nblocks},4,4)")
        y = torch.einsum("nkij,bhnkj->bhnki", mats.to(dtype=x.dtype), x)
    else:
        raise ValueError(f"mats must be 3-D or 4-D, got {tuple(mats.shape)}")
    out = feats.clone()
    out[..., dim_start:dim_end] = y.reshape(b, h, n, band)
    return out


def token_se3_from_coords(
    geo_coords: torch.Tensor,
    lambdas: torch.Tensor,
) -> torch.Tensor:
    """Per-token, per-block transform ``T = [R_cam | lambda_b * xyz]``.

    ``geo_coords`` is ``[N, 13]``; the result is ``[N, K, 4, 4]`` for ``K`` lambdas. Each
    lambda gives the block a different spatial wavelength, the same way RoPE gives each
    pair of channels a different frequency.
    """
    if geo_coords.ndim != 2 or geo_coords.shape[-1] < GEO_FEATURE_DIM:
        raise ValueError(
            f"token_se3_from_coords expects [N,{GEO_FEATURE_DIM}], got {tuple(geo_coords.shape)}"
        )
    xyz = geo_coords[..., 0:3].to(dtype=torch.float32)
    R = rot6d_to_matrix(geo_coords[..., 6:12].to(dtype=torch.float32))
    lam = lambdas.to(device=xyz.device, dtype=xyz.dtype).reshape(-1)
    n, k = xyz.shape[0], int(lam.numel())
    out = xyz.new_zeros(n, k, 4, 4)
    out[..., :3, :3] = R.unsqueeze(1).expand(n, k, 3, 3)
    out[..., :3, 3] = xyz.unsqueeze(1) * lam.view(1, k, 1)
    out[..., 3, 3] = 1.0
    return out


def _scatter_xform(
    src: torch.Tensor,
    bi: int,
    sel: torch.Tensor,
    mats: torch.Tensor,
    dim_start: int,
    dim_end: int,
) -> torch.Tensor:
    compact = src[bi : bi + 1, :, sel, :]
    out_c = apply_tiled_4x4_band(compact, mats, dim_start, dim_end)
    src = src.clone()
    src[bi, :, sel, :] = out_c[0]
    return src


class TokenSE3PoseGTA(nn.Module):
    """Two-band geometric transform attention over disjoint head-dimension slices.

    Token band (default ``[0:48]``, 12 blocks) carries per-token SE(3):
    ``T_i = [R_cam(i) | lambda_b * xyz_i]``, so a query-key pair sees
    ``T_i^{-1} T_j = [R_i^T R_j | lambda_b R_i^T (x_j - x_i)]`` -- purely relative geometry.

    Pose band (default ``[48:72]``, 6 blocks) carries frame-level relative camera pose
    without lambda scaling, which is where camera translation lives.

    Q, K, V and the attention output are transformed asymmetrically (``P^T`` on queries,
    ``P^{-1}`` on keys and values, ``P`` on the output, with ``P = T^{-1}``) so the
    composition leaves non-geometric channels untouched. This module holds no learnable
    parameters.
    """

    feature_dim = GEO_FEATURE_DIM

    def __init__(
        self,
        head_dim: int,
        token_dim: int = 48,
        pose_dim: int = 24,
        token_offset: int = 0,
        pose_offset: Optional[int] = None,
        lambda_min: float = 0.25,
        lambda_max: float = 4.0,
        lambdas: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        if token_dim % 4 != 0 or token_dim <= 0:
            raise ValueError(f"token_dim must be positive multiple of 4, got {token_dim}")
        if pose_dim % 4 != 0 or pose_dim <= 0:
            raise ValueError(f"pose_dim must be positive multiple of 4, got {pose_dim}")
        if pose_offset is None:
            pose_offset = token_offset + token_dim
        token_end = token_offset + token_dim
        pose_end = pose_offset + pose_dim
        if token_offset < 0 or pose_offset < 0:
            raise ValueError("slice offsets must be non-negative")
        if token_end > head_dim or pose_end > head_dim:
            raise ValueError(
                f"slice overflow: token[{token_offset}:{token_end}] "
                f"pose[{pose_offset}:{pose_end}] vs head_dim={head_dim}"
            )
        if not (token_end <= pose_offset or pose_end <= token_offset):
            raise ValueError(
                f"slice overlap: token[{token_offset}:{token_end}] "
                f"vs pose[{pose_offset}:{pose_end}]"
            )
        self.head_dim = int(head_dim)
        self.token_dim = int(token_dim)
        self.pose_dim = int(pose_dim)
        self.token_slice = (int(token_offset), int(token_end))
        self.pose_slice = (int(pose_offset), int(pose_end))

        n_token_blocks = self.token_dim // 4
        if lambdas is not None:
            lam = torch.tensor([float(x) for x in lambdas], dtype=torch.float32)
            if lam.numel() != n_token_blocks:
                raise ValueError(
                    f"lambdas length {lam.numel()} != n_token_blocks {n_token_blocks}"
                )
        else:
            lam = geometric_lambdas(n_token_blocks, float(lambda_min), float(lambda_max))
        self.lambda_min = float(lam.min().item())
        self.lambda_max = float(lam.max().item())
        self.register_buffer("lambdas", lam, persistent=False)
        self.token_se3_disable = False
        self.pose_gta_disable = False

    def apply_qkvo(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        geo_coords: torch.Tensor,
        geo_mask: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        """Transform Q/K/V in place of geometry-valid visual tokens.

        Returns the transformed tensors plus a closure that applies the matching inverse
        transform to the attention output.
        """
        if geo_coords.shape[-1] < self.feature_dim:
            raise ValueError(
                f"TokenSE3PoseGTA expects geo_coords[..., {self.feature_dim}], "
                f"got {tuple(geo_coords.shape)}"
            )
        if query_states.shape[-1] < self.head_dim:
            raise ValueError(f"head_dim {query_states.shape[-1]} < configured {self.head_dim}")

        bsz = query_states.shape[0]
        q, k, v = query_states, key_states, value_states
        mask = geo_mask.bool()
        if visual_mask is not None:
            mask = mask & visual_mask.bool()

        token_o: List[Optional[torch.Tensor]] = []
        pose_o: List[Optional[torch.Tensor]] = []
        token_on = not self.token_se3_disable
        pose_on = not self.pose_gta_disable
        t0, t1 = self.token_slice
        p0, p1 = self.pose_slice

        for bi in range(bsz):
            sel = mask[bi].nonzero(as_tuple=False).squeeze(-1)
            if sel.numel() == 0:
                token_o.append(None)
                pose_o.append(None)
                continue

            if token_on:
                T = token_se3_from_coords(geo_coords[bi, sel], self.lambdas)
                P = invert_se3(T)
                q = _scatter_xform(q, bi, sel, P.transpose(-1, -2), t0, t1)
                k = _scatter_xform(k, bi, sel, T, t0, t1)
                v = _scatter_xform(v, bi, sel, T, t0, t1)
                token_o.append(P)
            else:
                token_o.append(None)

            if pose_on:
                P_u, frame_ids = unique_frame_pose_and_ids(geo_coords[bi, ..., 3:13], sel)
                P_tok = P_u.to(device=q.device, dtype=torch.float32)[frame_ids]
                Pinv = invert_se3(P_tok)
                q = _scatter_xform(q, bi, sel, P_tok.transpose(-1, -2), p0, p1)
                k = _scatter_xform(k, bi, sel, Pinv, p0, p1)
                v = _scatter_xform(v, bi, sel, Pinv, p0, p1)
                pose_o.append(P_tok)
            else:
                pose_o.append(None)

        def apply_o(attn_out: torch.Tensor) -> torch.Tensor:
            out = attn_out
            for bi in range(bsz):
                sel = mask[bi].nonzero(as_tuple=False).squeeze(-1)
                if sel.numel() == 0:
                    continue
                for mats, (d0, d1) in ((token_o[bi], (t0, t1)), (pose_o[bi], (p0, p1))):
                    if mats is None:
                        continue
                    compact = out[bi : bi + 1, :, sel, :]
                    out_c = apply_tiled_4x4_band(compact, mats, d0, d1)
                    out = out.clone()
                    out[bi, :, sel, :] = out_c[0]
            return out

        return q, k, v, apply_o
