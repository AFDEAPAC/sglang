"""Gluon MLA decode wrapper for low head-count MLA (e.g. Kimi K3 TP8: 12 heads/GPU).

Uses aiter ``mla_gluon`` when import succeeds and Triton Gluon exposes ``cga_layout``
(needs Triton >= 3.7). Falls back to the caller (zero-pad + ``mla_decode_fwd``) when
Gluon is unavailable or fails at runtime.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

logger = logging.getLogger(__name__)

# mla_gluon.py:887-889 hard-requires gfx950; kernel body uses gl.amd.cdna4.* (CDNA4 ISA).
_GLUON_ARCH = "gfx950"
# mla_gluon.py:911 fp8 branch expects torch.float8_e4m3fn; gfx942 stack uses e4m3fnuz.
_GLUON_FP8_DTYPE = torch.float8_e4m3fn


@functools.lru_cache(maxsize=1)
def _gluon_hw_supported() -> bool:
    try:
        from aiter.ops.triton.utils._triton import arch_info
    except ImportError as exc:
        logger.info("aiter mla gluon is disabled: arch_info unavailable (%s)", exc)
        return False
    arch = arch_info.get_arch()
    if arch != _GLUON_ARCH:
        logger.info(
            "aiter mla gluon is disabled on %s (mla_gluon requires %s / CDNA4)",
            arch,
            _GLUON_ARCH,
        )
        return False
    if fp8_dtype != _GLUON_FP8_DTYPE:
        logger.info(
            "aiter mla gluon is disabled: stack fp8_dtype %s != mla_gluon %s",
            fp8_dtype,
            _GLUON_FP8_DTYPE,
        )
        return False
    return True


@functools.lru_cache(maxsize=1)
def _gluon_fn():
    if not envs.SGLANG_AITER_MLA_GLUON.get():
        logger.info("aiter mla gluon is disabled manually.")
        return None
    if not _gluon_hw_supported():
        return None
    try:
        import triton
        import triton.experimental.gluon.language as gl
        from aiter.ops.triton.gluon.mla_gluon import mla_gluon
    except ImportError as exc:
        logger.info("aiter mla gluon import error message: %s", exc)
        return None
    if "cga_layout" not in inspect.signature(gl.PaddedSharedLayout).parameters:
        logger.info(
            "aiter mla gluon is disabled due to triton %s has no Gluon cga_layout (need >= 3.7)",
            getattr(triton, "__version__", "unknown"),
        )
        return None
    return mla_gluon


def log_mla_gluon_capability(log: logging.Logger | None = None) -> None:
    ready = _gluon_fn() is not None
    (log or logger).info(
        "aiter mla gluon is %s",
        "enabled" if ready else "disabled; falling back to zero-pad mla_decode_fwd",
    )


def prefer_mla_gluon_decode(
    *, head_pad_mode: str, num_head: int, kv_cache_dtype: torch.dtype
) -> bool:
    return (
        head_pad_mode == "zero"
        and num_head == 12
        and kv_cache_dtype == fp8_dtype
        and _gluon_fn() is not None
    )


def mla_gluon_decode(
    *,
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    layer: RadixAttention,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    sm_scale: float,
    min_kv_seq_len: int,
    kv_scale: float = 1.0,
    qlen: int = 1,
) -> Optional[torch.Tensor]:
    mla_gluon = _gluon_fn()
    if mla_gluon is None:
        return None

    num_head = layer.tp_q_head_num
    kv_lora_rank = layer.v_head_dim
    qk_rope_head_dim = layer.qk_head_dim - kv_lora_rank
    batch_size = q.shape[0] // qlen

    q_nope, q_pe = torch.split(q, [kv_lora_rank, qk_rope_head_dim], dim=-1)
    if qlen > 1:
        q_nope = q_nope.view(batch_size, qlen, num_head, kv_lora_rank)
        q_pe = q_pe.view(batch_size, qlen, num_head, qk_rope_head_dim)
        o = q.new_empty((batch_size, qlen, num_head, kv_lora_rank))
    else:
        o = q.new_empty((batch_size, num_head, kv_lora_rank))

    try:
        mla_gluon(
            q_nope,
            q_pe,
            k_buffer.view(-1, layer.qk_head_dim),
            o,
            kv_indices,
            kv_indptr,
            sm_scale,
            k_pe=None,
            kv_pe_offset=kv_lora_rank,
            use_2d_view=False,
            kv_scale=kv_scale,
            min_kv_seq_len=min_kv_seq_len,
        )
    except Exception as exc:
        logger.warning(
            "mla_gluon decode failed (num_head=%s, kv_dtype=%s, batch=%s): %s; "
            "falling back to zero-pad mla_decode_fwd",
            layer.tp_q_head_num,
            k_buffer.dtype,
            batch_size,
            exc,
        )
        return None
    return o.flatten(0, 1) if qlen > 1 else o
