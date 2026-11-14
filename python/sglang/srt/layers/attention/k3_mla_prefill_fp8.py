"""K3_MLA_PREFILL_FP8: MLA prefill attention in fp8 (CK tile fmha fp8bf16, per-tensor scales) on gfx942."""
import os

import torch

MODE = os.environ.get("SGLANG_K3_MLA_PREFILL_FP8", "0").strip().lower()
ON = MODE not in ("", "0", "false")
FUSED_ON = MODE == "1" and os.environ.get("SGLANG_K3_MLA_PREFILL_FP8_FUSED", "0") not in ("", "0", "false")
_announced = False


def _quant(x):
    if FUSED_ON:
        from sglang.srt.layers.attention.k3_kv_fp8_kernels import quant

        return quant(x)
    # Dynamic per-tensor scale, bf16 multiply. Measured faster than aiter per_tensor_quant_hip (0.46 vs 4.1 ms
    # at 32768 x 12 x 192). The 0.99 margin keeps bf16 rounding of x * (1 / s) below the e4m3fnuz max.
    from aiter import dtypes

    fp8 = dtypes.fp8
    s = (x.abs().amax().float() / (torch.finfo(fp8).max * 0.99)).clamp_min(1e-12)
    return (x * (1.0 / s)).to(fp8).contiguous(), s.view(1)


def kv_to_fp8(kv, k_pe, num_heads, nope_dim, v_dim):
    from sglang.srt.layers.attention.k3_kv_fp8_kernels import kv_to_fp8 as _k

    return _k(kv, k_pe, num_heads, nope_dim, v_dim)


def _tp0():
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank() == 0
    except Exception:
        return True


def attn(q, k, v, cu_q, cu_k, max_q, max_k, scale, layer_id=-1, kv8=None):
    from aiter import flash_attn_varlen_fp8_pertensor_func, flash_attn_varlen_func

    global _announced
    q8, sq = _quant(q)
    if kv8 is not None:
        k8, v8, sk, sv = kv8
    else:
        k8, sk = _quant(k)
        v8, sv = _quant(v)
    o = flash_attn_varlen_fp8_pertensor_func(
        q8, k8, v8, sq, sk, sv, cu_q, cu_k, max_q, max_k, causal=True, softmax_scale=scale
    )
    if isinstance(o, tuple):
        o = o[0]
    if not _announced:
        _announced = True
        print(f"[k3-mla-prefill-fp8] active mode={MODE} fused={int(FUSED_ON)} q={tuple(q.shape)} "
              f"k={tuple(k8.shape)} v={tuple(v8.shape)}", flush=True)
    if MODE != "shadow":
        return o
    ob = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale=scale, causal=True)
    if _tp0():
        a, b = o.float().flatten(), (ob[0] if isinstance(ob, tuple) else ob).float().flatten()
        c = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        m = (a - b).abs().max().item()
        r = b.abs().max().item()
        print(
            f"[k3-mla-prefill-fp8-shadow] layer={layer_id} q={q.shape[0]} kv={k.shape[0]} max_q={max_q} "
            f"max_k={max_k} cos={c:.6f} maxabs={m:.4g} refmax={r:.4g} sq={sq.item():.4g} sk={sk.item():.4g} "
            f"sv={sv.item():.4g}",
            flush=True,
        )
    return ob
