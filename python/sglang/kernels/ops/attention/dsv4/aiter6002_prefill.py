# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Kernel bodies copied unchanged from ROCm/aiter PR6002 head
# bbac59f949fcd84fedc0862225214f74ed02aea5. Pinned gfx942 H8/16 D512 only.
# The adapter below merges SGLang's two BF16 pools/CSRs before the upstream kernel.
import torch
import triton
import triton.language as tl

@triton.jit
def _exp(x, USE_EXP2: tl.constexpr):
    if USE_EXP2:
        return tl.math.exp2(x)
    return tl.exp(x)

@triton.jit
def _sparse_attn_prefill_kernel(
    q_ptr,  # [num_queries, num_heads, head_dim]
    kv_ptr,  # [num_kv, head_dim]
    kv_indices_ptr,  # [nnz]
    kv_indptr_ptr,  # [num_queries + 1]
    attn_sink_ptr,  # [num_heads]
    out_ptr,  # [num_queries, num_heads, head_dim]
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    num_heads,
    head_dim,
    num_kv,
    scale,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_EXP2: tl.constexpr = False,
    # num_heads % BLOCK_H == 0: no head mask.
    EVEN_H: tl.constexpr = False,
    # head_dim == BLOCK_D: no dim mask; the KV gather is masked by row only.
    EVEN_D: tl.constexpr = False,
):
    # 64-bit before the multiply, same reasoning as `slot_off` below: the
    # program id fits 32 bits, but `query_idx * q_stride_t` does not once
    # num_queries passes 32K, because q_stride_t is num_heads * head_dim
    # (128 * 512 = 65536) in the V4 layout. Unlike the pool read, the wrapped
    # offset lands outside the q/out allocations, so it page-faults.
    query_idx = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim

    # Constant all-true masks fold away in the loads and the out store.
    if EVEN_H:
        head_mask = tl.full((BLOCK_H,), True, tl.int1)
    if EVEN_D:
        dim_mask = tl.full((BLOCK_D,), True, tl.int1)
    q = tl.load(
        q_ptr
        + query_idx * q_stride_t
        + head_offsets[:, None] * q_stride_h
        + dim_offsets[None, :] * q_stride_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    if USE_EXP2:
        scale = scale * 1.4426950408889634  # log2(e): softmax runs in base 2
    m_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    kv_start = tl.load(kv_indptr_ptr + query_idx)
    kv_end = tl.load(kv_indptr_ptr + query_idx + 1)
    kv_len = kv_end - kv_start

    k_offsets = tl.arange(0, BLOCK_K)
    # Prefetch first tile's slot indices so the indirect int32 load can overlap
    # the next iteration's QK MFMA latency.
    slot = tl.load(
        kv_indices_ptr + kv_start + k_offsets, mask=k_offsets < kv_len, other=-1
    )
    for k_start in tl.range(0, kv_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < kv_len
        valid = in_range & (slot >= 0) & (slot < num_kv)

        # 64-bit before the multiply, same as the decode kernel: an int32
        # slot index fits 32 bits but `slot *
        # kv_stride_n` does not once the unified V4 pool runs to ~150M rows.
        # The wrapped offset still lands inside the same allocation, so the
        # bad read is silent.
        slot_off = slot.to(tl.int64)

        kv_ptrs = (
            kv_ptr
            + slot_off[:, None] * kv_stride_n
            + dim_offsets[None, :] * kv_stride_d
        )
        # Invalid rows are never loaded: a zero softmax weight does not
        # neutralize a non-finite value in the PV dot (0 * NaN = NaN).
        if EVEN_D:
            kv = tl.load(kv_ptrs, mask=valid[:, None], other=0.0)
        else:
            kv = tl.load(kv_ptrs, mask=valid[:, None] & dim_mask[None, :], other=0.0)

        # Prefetch next tile's indices before heavy compute on current tile.
        next_k_pos = k_start + BLOCK_K + k_offsets
        slot = tl.load(
            kv_indices_ptr + kv_start + next_k_pos,
            mask=next_k_pos < kv_len,
            other=-1,
        )

        scores = tl.dot(q, tl.trans(kv)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, float("-inf"))

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.where(m_new == float("-inf"), 0.0, _exp(m_i - m_new, USE_EXP2))
        p = tl.where(
            m_new[:, None] == float("-inf"),
            0.0,
            _exp(scores - m_new[:, None], USE_EXP2),
        )
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=float("-inf")
        ).to(tl.float32)
        if USE_EXP2:
            sink = sink * 1.4426950408889634
        m_final = tl.maximum(m_i, sink)
        alpha = tl.where(m_final == float("-inf"), 0.0, _exp(m_i - m_final, USE_EXP2))
        exp_sink = tl.where(sink == float("-inf"), 0.0, _exp(sink - m_final, USE_EXP2))
        l_final = l_i * alpha + exp_sink
        denom = tl.maximum(l_final, 1.0e-30)
        out = tl.where(
            l_final[:, None] > 0.0,
            (acc * alpha[:, None]) / denom[:, None],
            0.0,
        )
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)

    tl.store(
        out_ptr
        + query_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + dim_offsets[None, :] * out_stride_d,
        out,
        mask=head_mask[:, None] & dim_mask[None, :],
    )

@triton.jit
def _merge_csr(pi, pp, ei, ep, oi, op, P_ROWS: tl.constexpr, E_ROWS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    ps = tl.load(pp + row); pn = tl.load(pp + row + 1) - ps
    es = tl.load(ep + row); en = tl.load(ep + row + 1) - es
    dst = tl.load(op + row)
    for k in tl.range(0, tl.maximum(pn, en), BLOCK):
        j = k + tl.arange(0, BLOCK)
        pv = tl.load(pi + ps + j, mask=j < pn, other=-1)
        ev = tl.load(ei + es + j, mask=j < en, other=-1)
        pv = tl.where((pv >= 0) & (pv < P_ROWS), pv, -1)
        ev = tl.where((ev >= 0) & (ev < E_ROWS), ev + P_ROWS, -1)
        tl.store(oi + dst + j, pv, mask=j < pn)
        tl.store(oi + dst + pn + j, ev, mask=j < en)


def sparse_prefill(q, kv, csr, attn_sink, softmax_scale, extend_kv=None, extend_csr=None):
    if q.dtype != torch.bfloat16 or q.ndim != 3 or q.shape[1] not in (8,16) or q.shape[2] != 512:
        raise ValueError('PR6002 adapter is limited to BF16 H8/H16 D512')
    if kv.dtype != q.dtype or kv.ndim != 2 or kv.shape[1] != 512 or kv.device != q.device:
        raise ValueError('KV shape/dtype/device mismatch')
    indices, indptr = csr
    if indices.dtype not in (torch.int32,torch.int64) or indptr.dtype not in (torch.int32,torch.int64):
        raise ValueError('CSR indices must be int32 or int64')
    if indptr.numel() != q.shape[0]+1:
        raise ValueError('CSR indptr length mismatch')
    if extend_kv is not None:
        if extend_csr is None or extend_kv.dtype != q.dtype or extend_kv.shape[1:] != kv.shape[1:] or extend_kv.device != q.device:
            raise ValueError('extend KV/CSR mismatch')
        ei,ep=extend_csr
        if ep.numel()!=indptr.numel() or ei.dtype not in (torch.int32,torch.int64):
            raise ValueError('extend CSR mismatch')
        dtype=torch.int64 if torch.int64 in (indices.dtype,ei.dtype,indptr.dtype,ep.dtype) else torch.int32
        merged_ptr=indptr.to(dtype)+ep.to(dtype)
        merged_idx=torch.empty(max(1,indices.numel()+ei.numel()),device=q.device,dtype=dtype)
        if q.shape[0]:
            _merge_csr[(q.shape[0],)](indices,indptr,ei,ep,merged_idx,merged_ptr,P_ROWS=kv.shape[0],E_ROWS=extend_kv.shape[0],BLOCK=256,num_warps=4)
        # shortcut: concatenate per call; measure end-to-end copy cost before adoption.
        kv=torch.cat((kv,extend_kv),dim=0)
        indices,indptr=merged_idx,merged_ptr
    q=q.contiguous();out=torch.empty_like(q)
    has_sink=attn_sink is not None
    sink=attn_sink.contiguous() if has_sink else torch.empty(1,device=q.device,dtype=torch.float32)
    if has_sink and (sink.numel()!=q.shape[1] or sink.device!=q.device):
        raise ValueError('attention sink mismatch')
    if q.shape[0]:
        _sparse_attn_prefill_kernel[(q.shape[0],1)](
            q,kv,indices,indptr,sink,out,*q.stride(),*kv.stride(),*out.stride(),
            q.shape[1],q.shape[2],kv.shape[0],float(softmax_scale),
            HAS_ATTN_SINK=has_sink,BLOCK_H=16,BLOCK_D=512,BLOCK_K=16,
            USE_EXP2=True,EVEN_H=q.shape[1]%16==0,EVEN_D=True,
            num_warps=1,num_stages=2,kpack=2)
    return out
