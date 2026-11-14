"""K3_MLA_FUSED_PREP: one-launch MLA decode Q-prep + fp8 KV write (skip-rope MLA on HIP).

Replaces torch.cat(q_nope_out, q_pe), torch.cat(k_nope, k_pe), the Q head zero pad and the
Triton fp8 KV write with aiter's fused_qk_rope_cat_and_cache_mla under an identity RoPE.
"""
import os

import torch

ENABLED = os.environ.get("SGLANG_K3_MLA_FUSED_PREP", "0") not in ("", "0", "false", "False")
A8W8 = os.environ.get("SGLANG_K3_MLA_DECODE_A8W8", "0") not in ("", "0", "false", "False")
_SCALE_ONE = {}
_a8w8_announced = False

# Pad heads [H:H_pad) are zeroed once at allocation and never written, so they stay zero
# across CUDA-graph replays. Buffers are never freed: a captured graph keeps the address
# it recorded, so a grown buffer must not release the old one.
_Q_BUFS = {}
_IDENTITY = {}
_KEEP_ALIVE = []
_announced = False


def _key(device, *rest):
    return (str(torch.device(device)),) + rest


def q_pad_buffer(num_tokens, num_head_padded, head_dim, dtype, device):
    key = _key(device, dtype, num_head_padded, head_dim)
    buf = _Q_BUFS.get(key)
    if buf is not None and buf.shape[0] >= num_tokens:
        return buf
    if torch.cuda.is_current_stream_capturing():
        return None
    buf = torch.zeros(
        (max(int(num_tokens), 1), num_head_padded, head_dim), dtype=dtype, device=device
    )
    _Q_BUFS[key] = buf
    _KEEP_ALIVE.append(buf)
    return buf


def identity_rope(rope_dim, device):
    key = _key(device, rope_dim)
    ident = _IDENTITY.get(key)
    if ident is not None:
        return ident
    if torch.cuda.is_current_stream_capturing():
        return None
    half = rope_dim // 2
    # Stride-0 rows: any position index reads cos=1 / sin=0, so real positions can be passed.
    cos = torch.ones(1, half, dtype=torch.float32, device=device).expand(1 << 20, half)
    sin = torch.zeros(1, half, dtype=torch.float32, device=device).expand(1 << 20, half)
    ident = (cos, sin)
    _IDENTITY[key] = ident
    _KEEP_ALIVE.append(ident)
    return ident


def scale_one(device):
    key = str(torch.device(device))
    t = _SCALE_ONE.get(key)
    if t is not None:
        return t
    t = torch.ones(1, dtype=torch.float32, device=device)
    _SCALE_ONE[key] = t
    _KEEP_ALIVE.append(t)
    return t


def q_out_dtype(src_dtype):
    if not A8W8:
        return src_dtype
    from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
    return get_fp8_e4m3_dtype()


def prepare(num_tokens, num_head_padded, kv_lora_rank, rope_dim, dtype, device):
    q_pad_buffer(num_tokens, num_head_padded, kv_lora_rank + rope_dim, dtype, device)
    if A8W8:
        q_pad_buffer(num_tokens, num_head_padded, kv_lora_rank + rope_dim, q_out_dtype(dtype), device)
        scale_one(device)
    identity_rope(rope_dim, device)


def lookup(q, num_head_padded):
    """Return the [T, H_pad, D] buffer view when q is heads [0:H) of it, else None."""
    if not ENABLED or q.dim() != 3:
        return None
    buf = _Q_BUFS.get(_key(q.device, q.dtype, num_head_padded, q.shape[-1]))
    if buf is None or q.data_ptr() != buf.data_ptr():
        return None
    # A single-row reshape may report any stride(0), so only check it when T > 1.
    if q.stride(1) != buf.stride(1) or q.stride(2) != 1:
        return None
    if q.shape[0] > 1 and q.stride(0) != buf.stride(0):
        return None
    return buf[: q.shape[0]]


def fused_prep(q_nope_out, q_pe, k_nope, k_pe, positions, out_cache_loc, kv_cache):
    """Fused Q concat (into the padded buffer) + fp8 KV write. Returns q [T, H, D] or None."""
    from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
    from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

    global _announced
    T, H, d_nope = q_nope_out.shape
    d_pe = q_pe.shape[-1]
    if (
        H not in (6, 12)
        or kv_cache.dtype != get_fp8_e4m3_dtype()
        or k_nope.dim() != 3
        or out_cache_loc.numel() != T
    ):
        return None
    qd = q_out_dtype(q_nope_out.dtype)
    buf = q_pad_buffer(T, 16, d_nope + d_pe, qd, q_nope_out.device)
    ident = identity_rope(d_pe, q_nope_out.device)
    if buf is None or ident is None:
        return None
    q_out = buf[:T, :H, :]
    fused_qk_rope_cat_and_cache_mla(
        q_nope_out,
        q_pe,
        k_nope,
        k_pe,
        kv_cache,
        out_cache_loc,
        positions,
        ident[0],
        ident[1],
        None,
        False,
        apply_scale=False,
        q_out=q_out,
        q_out_dtype=qd,
    )
    if not _announced:
        _announced = True
        print(f"[k3-mla-fused-prep] active H={H} H_pad=16 D={d_nope + d_pe}", flush=True)
    global _a8w8_announced
    if A8W8 and not _a8w8_announced:
        _a8w8_announced = True
        print(f"[k3-mla-a8w8] active H={H} H_pad=16 q_dtype={qd}", flush=True)
    return q_out
