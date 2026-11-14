"""Env-gated MLA prefill prefix paths for Kimi-K3 on gfx942.

SGLANG_K3_KV_CHUNK_TOKENS=N (seed, the MI300 fix): when the prefix-branch K length exceeds N, gather + cast +
  kv_b_proj run per N-row chunk twice (global amax, then quantize), so the bf16 kv_b_proj output never covers L.
SGLANG_K3_PREFIX_LSE=1 (D1): the new tokens attend to their own K/V (model-side bf16, causal); the cached prefix is
  processed in chunks of at most SGLANG_K3_PREFIX_LSE_CHUNK KV tokens (summed over requests), each decompressed
  once, quantized with its own per-tensor scale and attended non-causally with LSE; partial results are merged with
  sglang merge_state. Peak transient memory is bounded by the chunk size instead of L.
Decode context parallel (prefix_lse_attn(..., dcp=(group, W, rank))): each rank holds the latent of prefix positions
  pos % W == rank; every chunk is rebuilt from the W shards with one all-gather (byte-identical to the non-DCP gather).
"""
import os

import torch

from sglang.srt.layers.attention import k3_kv_fp8_kernels as K

CHUNK_TOKENS = int(os.environ.get("SGLANG_K3_KV_CHUNK_TOKENS", "0") or 0)
LSE_ON = os.environ.get("SGLANG_K3_PREFIX_LSE", "0") == "1"
LSE_CHUNK = int(os.environ.get("SGLANG_K3_PREFIX_LSE_CHUNK", "65536") or 65536)
KVB_FP8 = os.environ.get("SGLANG_K3_PREFIX_KVB_FP8", "0") == "1"
NOSYNC = os.environ.get("SGLANG_K3_PREFIX_NOSYNC", "0") == "1"
_announced = set()
_ones = {}
_layout = {}
_dcp_plans = {}


def _kvb_fp8_weight(layer):
    """Per-output-channel fp8 copy of kv_b_proj, preshuffled for gemm_a8w8_bpreshuffle, cached on the layer."""
    cached = layer.__dict__.get("_k3_kvb_fp8")
    if cached is None:
        import aiter
        from aiter.ops.shuffle import shuffle_weight

        fp8 = aiter.dtypes.fp8
        fmax = torch.finfo(fp8).max
        w = layer.kv_b_proj.weight.float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / fmax
        q = (w / scale).clamp(-fmax, fmax).to(fp8)
        cached = (shuffle_weight(q.contiguous(), (16, 16)), scale.reshape(1, -1).contiguous())
        layer.__dict__["_k3_kvb_fp8"] = cached
    return cached


def _ones_col(n, device):
    t = _ones.get(device)
    if t is None or t.shape[0] < n:
        t = torch.ones(max(n, 65536), 1, dtype=torch.float32, device=device)
        _ones[device] = t
    return t[:n]


def _announce(tag, msg):
    if tag not in _announced:
        _announced.add(tag)
        print(f"[k3-prefix] active {tag}: {msg}", flush=True)


def _gather_decompress(layer, K_Buffer, idx, R, DR, fp8_cache, dtype):
    return _decompress(layer, torch.index_select(K_Buffer, 0, idx), R, DR, fp8_cache, dtype)


def _decompress(layer, kb, R, DR, fp8_cache, dtype):
    n = kb.shape[0]
    kvc, pe = torch.split(kb, [R, DR], dim=-1)
    if KVB_FP8 and fp8_cache:
        # D3a: the cache holds fp8 latent with unit scale (the bf16 path reads it with a plain cast), so it is the
        # fp8 A operand directly; this drops the fp8->bf16 cast of the latent and runs kv_b_proj on fp8 MFMA
        import aiter

        wq, ws = _kvb_fp8_weight(layer)
        kv = aiter.gemm_a8w8_bpreshuffle(kvc.reshape(n, R).contiguous(), wq, _ones_col(n, kb.device), ws, None, dtype)
        _announce("kvb_fp8", f"n={n} w={tuple(wq.shape)}")
        return kv, pe.to(dtype).reshape(n, DR)
    if fp8_cache:
        kvc = kvc.to(dtype)
        pe = pe.to(dtype)
    kv = layer.kv_b_proj(kvc.contiguous())[0]
    return kv.reshape(n, -1), pe.reshape(n, DR)


def prefix_twopass_attn(q, layer, K_Buffer, kv_indices, qo_indptr, kv_indptr, max_q_len, max_kv_len,
                        R, DR, DN, fp8_cache):
    from sglang.srt.layers.attention import k3_mla_prefill_fp8 as P

    H, DV, L = layer.tp_k_head_num, layer.v_head_dim, kv_indices.shape[0]
    _announce("twopass", f"chunk={CHUNK_TOKENS} L={L}")

    def chunk_kv(s, e):
        return _gather_decompress(layer, K_Buffer, kv_indices[s:e], R, DR, fp8_cache, q.dtype)

    kv8 = K.kv_to_fp8_chunked(chunk_kv, L, H, DN, DV, DR, CHUNK_TOKENS, q.device)
    return P.attn(q, None, None, qo_indptr, kv_indptr, max_q_len, max_kv_len, layer.scaling, layer.layer_id, kv8=kv8)


def _attn_lse(q8, sq, k8, sk, v8, sv, cu_q, cu_k, max_q, max_k, scale, causal, dv):
    from aiter.ops.mha import _flash_attn_varlen_forward

    out, lse, _, _ = _flash_attn_varlen_forward(
        q8, k8, v8, cu_q, cu_k, None, None, max_q, max_k, 0, 0.0, scale,
        causal=causal, logits_soft_cap=0.0, window_size_left=-1, window_size_right=-1, sink_size=0,
        bias=None, alibi_slopes=None, q_descale=sq, k_descale=sk, v_descale=sv,
        return_lse=True, return_softmax=False)
    return out[..., :dv].contiguous(), lse.transpose(0, 1).contiguous().float()


def _chunk_layout(pre, ext_lens, cl, nch, T, dev):
    """Per-chunk cu_k and empty-request row masks for one forward, shared by all its layers. Built on the host and
    sent with one non-blocking copy from pinned memory: a blocking torch.tensor(list, device=...) per chunk and layer
    synchronises the stream every time."""
    key = (tuple(pre), tuple(ext_lens), cl, nch, T, dev)
    hit = _layout.get("key") == key
    if not hit:
        lens_all = [[max(0, min(cl, p - j * cl)) for p in pre] for j in range(nch)]
        cu = torch.zeros(nch, len(pre) + 1, dtype=torch.int32)
        rows = torch.zeros(nch, T, dtype=torch.bool)
        for j, lens in enumerate(lens_all):
            cu[j, 1:] = torch.tensor(lens, dtype=torch.int32).cumsum(0)
            qs = 0
            for ln, e in zip(lens, ext_lens):
                if ln == 0:
                    rows[j, qs: qs + e] = True
                qs += e
        cu_h, rows_h = cu.pin_memory(), rows.pin_memory()
        _layout.clear()
        _layout.update(key=key, lens=lens_all, host=(cu_h, rows_h),
                       cu=cu_h.to(dev, non_blocking=True), rows=rows_h.to(dev, non_blocking=True))
    return _layout["lens"], _layout["cu"], _layout["rows"]


def _dcp_plan(lens_all, starts, cl, W, r, dev):
    """Per chunk j: (gidx, valid, perm, M) for this rank, or None for an empty chunk. Lengths-only, so cached per key.

    Chunk j of request i covers prefix positions [j*cl, j*cl + lens_all[j][i]); cl % W == 0, so position j*cl + t lives
    on rank t % W. Every rank sends M_i = ceil(len / W) rows per request (positions j*cl + W*k + rank; rows past the end
    are padding), the all-gather stacks the W blocks rank-major, and row t of request i sits at
    (t % W) * M + off_i + t // W. gidx indexes kv_indices (virtual slots), padding rows read local row 0.
    """
    key = (tuple(tuple(x) for x in lens_all), tuple(starts), cl, W, r, str(dev))
    if _dcp_plans.get("key") != key:
        pin = (lambda t: t.pin_memory()) if torch.device(dev).type == "cuda" else (lambda t: t)
        plans = []
        for j, lens in enumerate(lens_all):
            ms = [(ln + W - 1) // W for ln in lens]
            M = sum(ms)
            if M == 0:
                plans.append(None)
                continue
            g, v, p, off = [], [], [], 0
            for st, ln, m in zip(starts, lens, ms):
                if ln == 0:
                    continue
                t = torch.arange(m, dtype=torch.int64) * W + r
                v.append(t < ln)
                g.append(torch.where(t < ln, st + j * cl + t, 0))
                tt = torch.arange(ln, dtype=torch.int64)
                p.append((tt % W) * M + off + tt // W)
                off += m
            host = [pin(torch.cat(x)) for x in (g, v, p)]
            plans.append((host, [h.to(dev, non_blocking=True) for h in host], M))
        _dcp_plans.clear()
        _dcp_plans.update(key=key, plans=plans)
    return [None if x is None else (*x[1], x[2]) for x in _dcp_plans["plans"]]


def _dcp_local_rows(K_Buffer, kv_indices, plan, W, translate=None):
    gidx, valid, _, _ = plan
    slots = kv_indices[gidx]
    rows = slots // W if translate is None else translate(slots)
    return torch.index_select(K_Buffer, 0, torch.where(valid, rows, 0))


def _dcp_check_owner(kv_indices, plan, W, r):
    """The plan puts position t on rank t % W; the cache keeps slot s on rank s % W. They agree only while each
    request's slots run W-aligned with its positions, so check it once (one host sync)."""
    gidx, valid, _, _ = plan
    if not bool((kv_indices[gidx][valid] % W == r).all()):
        raise RuntimeError("k3 prefix DCP: a prefix slot is not on the rank its position maps to")


def _dcp_gather(K_Buffer, kv_indices, plan, group, W, translate=None):
    """Rows of K_Buffer for this chunk in non-DCP order (= index_select(global_K, 0, chunk virtual slots))."""
    M = plan[3]
    rows = _dcp_local_rows(K_Buffer, kv_indices, plan, W, translate)
    out = torch.empty((W * M,) + tuple(K_Buffer.shape[1:]), dtype=K_Buffer.dtype, device=K_Buffer.device)
    # pynccl has no fp8 dtype; the all-gather is a byte copy
    group.all_gather_into_tensor(out.view(torch.uint8), rows.view(torch.uint8))
    return torch.index_select(out, 0, plan[2])


def prefix_lse_attn(q, k_ext, v_ext, layer, seq_lens, ext_lens, K_Buffer, kv_indices, qo_indptr, max_q_len,
                    R, DR, DN, fp8_cache, dcp=None, translate=None):
    """seq_lens / ext_lens: python lists per request (full KV length, new tokens). Returns [T, H, DV].

    dcp: None, or (group, W, rank) when K_Buffer is this rank's decode-context-parallel shard.
    translate: maps this rank's virtual slots to K_Buffer rows under DCP; slot // W when None."""
    from sglang.srt.layers.attention.merge_state import merge_state

    H, DV, scale = layer.tp_k_head_num, layer.v_head_dim, layer.scaling
    pre = [s - e for s, e in zip(seq_lens, ext_lens)]
    starts, acc = [], 0
    for s in seq_lens:
        starts.append(acc)
        acc += s
    q8, sq = K.quant(q)
    ke8, ske = K.quant(k_ext)
    ve8, sve = K.quant(v_ext)
    acc_o, acc_l = _attn_lse(q8, sq, ke8, ske, ve8, sve, qo_indptr, qo_indptr, max_q_len, max_q_len, scale,
                             True, DV)
    with_prefix = sum(1 for p in pre if p > 0)
    cl = max(LSE_CHUNK // max(with_prefix, 1), 1024)
    if dcp is not None:
        # chunk starts must stay on a multiple of W for the owner rule pos % W == rank
        cl -= cl % dcp[1]
    nch = (max(pre) + cl - 1) // cl if pre else 0
    _announce("lse", f"chunk={LSE_CHUNK} per_req={cl} chunks={nch} bs={len(pre)} maxpre={max(pre) if pre else 0}")
    dev = q.device
    if NOSYNC and nch:
        lens_all, cu_all, rows_all = _chunk_layout(pre, ext_lens, cl, nch, q.shape[0], dev)
        _announce("nosync", f"chunks={nch} layout cached per forward")
    if dcp is not None and nch:
        plans = _dcp_plan([[max(0, min(cl, p - j * cl)) for p in pre] for j in range(nch)], starts, cl, dcp[1],
                          dcp[2], dev)
        if "dcp" not in _announced:
            first = next(p for p in plans if p is not None)
            _dcp_check_owner(kv_indices, first, dcp[1], dcp[2])
        _announce("dcp", f"W={dcp[1]} rank={dcp[2]} chunks={nch}")
    for j in range(nch):
        lens = lens_all[j] if NOSYNC else [max(0, min(cl, p - j * cl)) for p in pre]
        n = sum(lens)
        if n == 0:
            continue
        if dcp is None:
            idx = torch.cat([kv_indices[st + j * cl: st + j * cl + ln] for st, ln in zip(starts, lens) if ln > 0])
            kv, pe = _gather_decompress(layer, K_Buffer, idx, R, DR, fp8_cache, q.dtype)
        else:
            kb = _dcp_gather(K_Buffer, kv_indices, plans[j], dcp[0], dcp[1], translate)
            kv, pe = _decompress(layer, kb, R, DR, fp8_cache, q.dtype)
            del kb
        k8, v8, sk, sv = K.kv_to_fp8(kv, pe, H, DN, DV)
        del kv, pe
        if NOSYNC:
            cu_k = cu_all[j]
        else:
            cu = [0]
            for ln in lens:
                cu.append(cu[-1] + ln)
            cu_k = torch.tensor(cu, device=dev, dtype=torch.int32)
        oc, lc = _attn_lse(q8, sq, k8, sk, v8, sv, qo_indptr, cu_k, max_q_len, max(lens), scale, False, DV)
        del k8, v8
        if NOSYNC and any(ln == 0 for ln in lens):
            rows = rows_all[j]
            oc = oc.masked_fill(rows[:, None, None], 0.0)
            lc = lc.masked_fill(rows[:, None], float("-inf"))
        elif any(ln == 0 for ln in lens):
            # requests with no keys in this chunk: neutralise their rows (output 0, lse -inf) before the merge
            rows = torch.zeros(q.shape[0], dtype=torch.bool, device=dev)
            qs = 0
            for ln, e in zip(lens, ext_lens):
                if ln == 0:
                    rows[qs: qs + e] = True
                qs += e
            oc = oc.masked_fill(rows[:, None, None], 0.0)
            lc = lc.masked_fill(rows[:, None], float("-inf"))
        acc_o, acc_l = merge_state(acc_o, acc_l, oc, lc)
    return acc_o
