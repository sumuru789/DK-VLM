"""Inject the geometry branch into a stock Qwen2.5-VL or Qwen3-VL model.

Selected text layers run two attention branches side by side:

* the **semantic** branch, which retains the backbone's native attention computation;
* the **geometry** branch, which uses its own Q/K/V projections and applies
  :class:`~dkvlm.geo_modules.TokenSE3PoseGTA` so that attention between two visual tokens
  depends on their relative 3D pose.

Their outputs are blended only on visual tokens that carry valid geometry, with a learned
per-layer gate ``alpha = sigmoid(geo_gate[layer])``. Text tokens, and visual tokens whose
geometry is missing, pass through the semantic branch unchanged. Layers outside the target
set are not touched at all.

Attachment patches ``forward`` only on the target-layer *instances*. The other text
layers keep their native class ``forward`` (Flash Attention 2 / SDPA). The base model
still supports ``generate``, caching and LoRA.
"""

from __future__ import annotations

import weakref
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLAttention,
    apply_multimodal_rotary_pos_emb,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)

from .geo_modules import (
    GeoBranchQKV,
    TokenSE3PoseGTA,
    geo_runtime_clear,
    geo_runtime_get,
    geo_runtime_set,
)


class DKVLMGeoWrapper(nn.Module):
    """Wraps Qwen2.5-VL or Qwen3-VL and owns the geometry branch parameters.

    The only parameters this adds are, per target layer, a :class:`GeoBranchQKV` and a
    scalar gate. ``TokenSE3PoseGTA`` is parameter-free.
    """

    def __init__(
        self,
        base: nn.Module,
        target_layers: Sequence[int],
        *,
        use_geo_branch: bool = True,
        geo_gate_init: float = 0.0,
        geo_attention_backend: str = "sdpa",
        geo_attention_chunk_size: int = 512,
        geo_branch_mrope: bool = False,
        geo_token_se3_dim: int = 48,
        geo_pose_gta_dim: int = 24,
        geo_token_se3_lambda_min: float = 0.25,
        geo_token_se3_lambda_max: float = 4.0,
    ):
        super().__init__()
        self.base = base
        self._target_layers = tuple(int(x) for x in target_layers)
        self.target_layer_set: Set[int] = set(self._target_layers)
        self.use_geo_branch = bool(use_geo_branch)
        self.geo_attention_backend = (geo_attention_backend or "sdpa").lower().strip()
        self.geo_attention_chunk_size = int(geo_attention_chunk_size)
        self.geo_branch_mrope = bool(geo_branch_mrope)

        if not self._target_layers:
            raise ValueError("target_layers must not be empty")

        head_dim = int(base.model.language_model.layers[0].self_attn.head_dim)
        self.token_se3_gta = TokenSE3PoseGTA(
            head_dim=head_dim,
            token_dim=int(geo_token_se3_dim),
            pose_dim=int(geo_pose_gta_dim),
            lambda_min=float(geo_token_se3_lambda_min),
            lambda_max=float(geo_token_se3_lambda_max),
        )
        self.geo_gate = nn.ParameterDict(
            {
                str(li): nn.Parameter(torch.tensor(float(geo_gate_init), dtype=torch.float32))
                for li in self._target_layers
            }
        )
        self.geo_branch_qkv = nn.ModuleDict(
            {
                str(li): GeoBranchQKV(
                    base.model.language_model.layers[li].self_attn, use_geo_o_proj=True
                )
                for li in self._target_layers
            }
        )

    @property
    def target_layers(self) -> Tuple[int, ...]:
        return self._target_layers


    def _publish_geometry(self, geo_coords, geo_coords_mask, input_ids) -> None:
        if (
            self.use_geo_branch
            and geo_coords is not None
            and input_ids is not None
        ):
            geo_runtime_set(
                geo_coords=geo_coords,
                geo_coords_mask=geo_coords_mask,
                input_ids=input_ids,
                image_token_id=int(self.base.config.image_token_id),
                video_token_id=int(self.base.config.video_token_id),
            )
        else:
            geo_runtime_clear()

    def forward(self, *args, **kwargs):
        geo_coords = kwargs.pop("geo_coords", None)
        geo_coords_mask = kwargs.pop("geo_coords_mask", None)
        self._publish_geometry(geo_coords, geo_coords_mask, kwargs.get("input_ids"))
        try:
            return self.base(*args, **kwargs)
        finally:
            geo_runtime_clear()

    def generate(self, *args, **kwargs):
        geo_coords = kwargs.pop("geo_coords", None)
        geo_coords_mask = kwargs.pop("geo_coords_mask", None)
        self._publish_geometry(geo_coords, geo_coords_mask, kwargs.get("input_ids"))
        try:
            return self.base.generate(*args, **kwargs)
        finally:
            geo_runtime_clear()


    def load_geo_state(
        self, state_dict: Dict[str, torch.Tensor], *, strict: bool = True
    ) -> Dict[str, Any]:
        """Load the gates and projections contained in released checkpoints.

        Non-geometry entries in older checkpoints are reported and ignored.
        """
        gate_sd: Dict[str, torch.Tensor] = {}
        qkv_sd: Dict[str, torch.Tensor] = {}
        ignored: list[str] = []
        for key, value in state_dict.items():
            if key.startswith("geo_gate."):
                gate_sd[key[len("geo_gate.") :]] = value
            elif key.startswith("geo_branch_qkv."):
                qkv_sd[key[len("geo_branch_qkv.") :]] = value
            else:
                ignored.append(key)

        missing_gates = sorted(set(self.geo_gate.keys()) - set(gate_sd))
        if strict and missing_gates:
            raise KeyError(f"checkpoint is missing geo_gate entries for layers {missing_gates}")
        for k, v in gate_sd.items():
            if k in self.geo_gate:
                self.geo_gate[k].data.copy_(v)
        self.geo_branch_qkv.load_state_dict(qkv_sd, strict=strict)
        return {
            "loaded_gates": sorted(gate_sd),
            "loaded_qkv_tensors": len(qkv_sd),
            "ignored_keys": sorted(ignored),
        }

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)


def _heads_to_hidden(attn_out_bhsd: torch.Tensor, bsz: int, q_len: int) -> torch.Tensor:
    return attn_out_bhsd.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)


def _project_and_rope(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    past_key_values,
    cache_position,
):
    """Qwen3-VL's own Q/K/V projection + mRoPE, without the GQA head repeat."""
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim)
    value_states = self.v_proj(hidden_states)
    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states)
    if hasattr(self, "k_norm"):
        key_states = self.k_norm(key_states)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = _apply_text_rope(self, query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states,
            value_states,
            self.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )
    return query_states, key_states, value_states


def _apply_text_rope(self, query_states, key_states, cos, sin):
    """Qwen3 uses RoPE already packed into ``cos``/``sin``. Qwen2.5 still splits mRoPE sections here."""
    scaling = getattr(self, "rope_scaling", None)
    section = scaling.get("mrope_section") if isinstance(scaling, dict) else None
    if section:
        return apply_multimodal_rotary_pos_emb(query_states, key_states, cos, sin, section)
    return apply_rotary_pos_emb(query_states, key_states, cos, sin)


def _prepare_sdpa_mask(attention_mask, query_states, key_states):
    if attention_mask is None or not torch.is_tensor(attention_mask):
        return attention_mask
    if attention_mask.ndim == 4:
        return attention_mask
    if attention_mask.ndim == 3:
        return attention_mask[:, None, :, :]
    if attention_mask.ndim != 2:
        return attention_mask
    bsz, _, q_len, _ = query_states.shape
    key_len = key_states.shape[-2]
    padding_mask = attention_mask[:, None, None, :key_len].to(
        device=query_states.device, dtype=torch.bool
    )
    q_pos = torch.arange(q_len, device=query_states.device)[:, None]
    k_pos = torch.arange(key_len, device=query_states.device)[None, :]
    causal_mask = k_pos <= (k_pos.new_tensor(key_len - q_len) + q_pos)
    return padding_mask & causal_mask[None, None, :, :]


def _semantic_attention(
    self,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask,
    output_attentions: bool,
    **kwargs,
):
    """Run Qwen3-VL's configured attention implementation, stopping before ``o_proj``."""
    impl = getattr(self.config, "_attn_implementation", None) or "eager"
    if output_attentions or impl == "eager":
        attention_interface, impl = eager_attention_forward, "eager"
    else:
        attention_interface = ALL_ATTENTION_FUNCTIONS[impl]
    if impl == "sdpa":
        attention_mask = _prepare_sdpa_mask(attention_mask, query_states, key_states)
    kw = {k: v for k, v in kwargs.items() if k != "is_causal"}
    attn_out, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=self.scaling,
        output_attentions=output_attentions,
        is_causal=None,
        **kw,
    )
    return attn_out.transpose(1, 2).contiguous(), attn_weights


def _geo_attention(wrapper, qv, kv, vv, scaling: float) -> torch.Tensor:
    """Dense bidirectional attention among the geometry-carrying visual tokens.

    ``chunked`` splits the query axis to bound peak memory; it computes the same values as
    ``eager`` because each query row already attends over all keys.
    """
    backend = getattr(wrapper, "geo_attention_backend", "sdpa")
    if backend == "sdpa":
        return F.scaled_dot_product_attention(
            qv, kv, vv, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scaling
        )
    if backend == "eager":
        scores = torch.matmul(qv, kv.transpose(2, 3)) * scaling
        w = F.softmax(scores, dim=-1, dtype=torch.float32).to(qv.dtype)
        return torch.matmul(w, vv)
    if backend == "chunked":
        cs = max(1, int(getattr(wrapper, "geo_attention_chunk_size", 512)))
        outs = []
        for s in range(0, int(qv.shape[2]), cs):
            scores = torch.matmul(qv[:, :, s : s + cs, :], kv.transpose(2, 3)) * scaling
            w = F.softmax(scores, dim=-1, dtype=torch.float32).to(qv.dtype)
            outs.append(torch.matmul(w, vv))
        return torch.cat(outs, dim=2) if outs else torch.zeros_like(qv)
    raise ValueError(f"unknown geo_attention_backend={backend!r}")


def _geo_alpha(wrapper, layer_idx: int, device, dtype) -> torch.Tensor:
    gate = wrapper.geo_gate[str(layer_idx)]
    return torch.sigmoid(gate.to(device=device, dtype=dtype))


def _geo_branch_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask,
    position_ids,
    past_key_values,
    output_attentions: bool,
    cache_position,
    position_embeddings,
    **kwargs,
):
    """Semantic attention plus the geometry branch, blended on geometry-valid tokens."""
    bsz, q_len, _ = hidden_states.size()
    wrapper = self._geo_wrapper_ref
    layer_idx = int(self.layer_idx)
    geo_qkv = wrapper.geo_branch_qkv[str(layer_idx)]

    ctx = geo_runtime_get()
    gcoords, gmask, vis = ctx.geo_coords, ctx.geo_coords_mask, ctx.visual_token_mask

    query_states, key_states, value_states = _project_and_rope(
        self, hidden_states, position_embeddings, past_key_values, cache_position
    )
    sem_out, attn_weights = _semantic_attention(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        output_attentions=output_attentions,
        position_ids=position_ids,
        sliding_window=getattr(self, "sliding_window", None),
        **kwargs,
    )

    geo_q, geo_k, geo_v = geo_qkv.project(hidden_states)
    if wrapper.geo_branch_mrope:
        cos, sin = position_embeddings
        geo_q, geo_k = _apply_text_rope(self, geo_q, geo_k, cos, sin)
    geo_q, geo_k, geo_v, apply_o = wrapper.token_se3_gta.apply_qkvo(
        geo_q, geo_k, geo_v, gcoords, gmask, vis
    )

    geo_v_rep = repeat_kv(geo_v, self.num_key_value_groups)
    geo = torch.zeros_like(sem_out)
    apply_mask = vis.to(torch.bool) & gmask.to(torch.bool)
    branch_executed = False
    for b in range(bsz):
        m = apply_mask[b]
        if not m.any().item():
            continue
        j = m.nonzero(as_tuple=False).squeeze(-1).long()
        gout = _geo_attention(
            wrapper,
            geo_q[b : b + 1, :, j, :],
            repeat_kv(geo_k[b : b + 1, :, j, :], self.num_key_value_groups),
            geo_v_rep[b : b + 1, :, j, :],
            self.scaling,
        )
        geo[b : b + 1, :, j, :] = gout
        branch_executed = True
    if branch_executed:
        geo = apply_o(geo)

    merged = self.o_proj(_heads_to_hidden(sem_out, bsz, q_len)).clone()
    alpha = _geo_alpha(wrapper, layer_idx, query_states.device, query_states.dtype)
    for b in range(bsz):
        m = apply_mask[b]
        if not m.any().item():
            continue
        idx = m.nonzero(as_tuple=False).squeeze(-1).long()
        sem_h = geo_qkv.o_proj(_heads_to_hidden(sem_out[b : b + 1], 1, q_len))[0]
        geo_h = geo_qkv.o_proj(_heads_to_hidden(geo[b : b + 1], 1, q_len))[0]
        merged[b, idx, :] = (1.0 - alpha) * sem_h[idx, :] + alpha * geo_h[idx, :]

    return merged, (attn_weights if output_attentions else None)


def geo_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings=None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    cache_position=None,
    position_ids=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    """Replacement for ``Qwen3VLTextAttention.forward`` on geometry target layers only.

    Prefill with matching geometry runs the dual branch. Decode steps (``q_len`` no
    longer matches the prompt-length geometry tensors) fall through to the original
    class ``forward``, which stays Flash Attention / SDPA.
    """
    if position_embeddings is None:
        raise ValueError("Qwen3-VL text attention requires position_embeddings")

    wrapper = getattr(self, "_geo_wrapper_ref", None)
    ctx = geo_runtime_get()
    bsz, q_len, _ = hidden_states.shape
    geo_active = (
        wrapper is not None
        and wrapper.use_geo_branch
        and ctx.geo_coords is not None
        and int(self.layer_idx) in wrapper.target_layer_set
        and ctx.geo_coords.shape[0] == bsz
        and ctx.geo_coords.shape[1] == q_len
    )

    if not geo_active:
        kw = {
            k: v
            for k, v in kwargs.items()
            if k
            not in (
                "past_key_values",
                "position_embeddings",
                "attention_mask",
                "cache_position",
            )
        }
        if position_ids is not None:
            kw.setdefault("position_ids", position_ids)
        if output_attentions:
            kw.setdefault("output_attentions", output_attentions)
        if use_cache:
            kw.setdefault("use_cache", use_cache)
        if isinstance(self, Qwen2_5_VLAttention):
            kw.pop("position_ids", None)
            kw.pop("output_attentions", None)
            kw.pop("use_cache", None)
            return Qwen2_5_VLAttention.forward(
                self,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kw,
            )
        return Qwen3VLTextAttention.forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kw,
        )

    return _geo_branch_forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_values,
        output_attentions,
        cache_position,
        position_embeddings,
        **kwargs,
    )


def install_geo_attention(model: nn.Module, wrapper: DKVLMGeoWrapper) -> None:
    """Patch only target-layer attention instances; leave the class method native."""
    import types

    for layer in model.model.language_model.layers:
        attn = layer.self_attn
        if not isinstance(attn, (Qwen3VLTextAttention, Qwen2_5_VLAttention)):
            raise TypeError(
                f"expected Qwen3-VL or Qwen2.5-VL text attention, got {type(attn).__name__}"
            )
        if int(attn.layer_idx) not in wrapper.target_layer_set:
            continue
        object.__setattr__(attn, "_geo_wrapper_ref", weakref.proxy(wrapper))
        attn.forward = types.MethodType(geo_attention_forward, attn)


def uninstall_geo_attention(model: nn.Module) -> None:
    """Undo :func:`install_geo_attention`."""
    for layer in model.model.language_model.layers:
        attn = layer.self_attn
        if hasattr(attn, "_geo_wrapper_ref"):
            delattr(attn, "_geo_wrapper_ref")
        if "forward" in getattr(attn, "__dict__", {}):
            delattr(attn, "forward")
