"""Fused fp8 emission for the MLA prefill attention inputs (gfx942, e4m3fnuz, per-tensor scales).

kv_to_fp8(kv, k_pe, H, DN, DV):  kv = kv_b_proj output [L, H*(DN+DV)] bf16, k_pe [L, 1, DR] bf16
    -> k8 [L, H, DN+DR] fp8 (nope | broadcast rope), v8 [L, H, DV] fp8, sk [1], sv [1] (fp32 descales).
    Replaces torch.split + torch.cat + two dynamic per-tensor quants: one amax pass, one write pass.
quant(x) -> (x8, s): dynamic per-tensor quant of any tensor, two kernels.
Scale = amax / (fp8 max * 0.99), the margin matching the torch path it replaces.
set_mla_kv_dcp(kv_buffer, loc, src, dcp_size, dcp_rank): MLA latent write under decode context parallel.
"""
import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fnuz
FP8_MAX = torch.finfo(FP8).max
MARGIN = 0.99
BLOCK_L = 32


@triton.jit
def _kv_amax(kv_ptr, pe_ptr, amax_ptr, L, s_kv_l, s_pe_l, s_pe_d,
             H: tl.constexpr, DN: tl.constexpr, DV: tl.constexpr, DR: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    rm = rows < L
    dn = tl.arange(0, DN)
    dv = tl.arange(0, DV)
    dr = tl.arange(0, DR)
    pe = tl.load(pe_ptr + rows[:, None] * s_pe_l + dr[None, :] * s_pe_d, mask=rm[:, None], other=0.0)
    ak = tl.max(tl.max(tl.abs(pe.to(tl.float32)), axis=1), axis=0)
    av = 0.0
    for h in range(H):
        base = kv_ptr + rows[:, None] * s_kv_l + h * (DN + DV)
        x = tl.load(base + dn[None, :], mask=rm[:, None], other=0.0)
        ak = tl.maximum(ak, tl.max(tl.max(tl.abs(x.to(tl.float32)), axis=1), axis=0))
        y = tl.load(base + DN + dv[None, :], mask=rm[:, None], other=0.0)
        av = tl.maximum(av, tl.max(tl.max(tl.abs(y.to(tl.float32)), axis=1), axis=0))
    # |x| >= 0, so its int32 bit pattern orders like the float
    tl.atomic_max(amax_ptr, ak.to(tl.int32, bitcast=True))
    tl.atomic_max(amax_ptr + 1, av.to(tl.int32, bitcast=True))


@triton.jit
def _kv_quant(kv_ptr, pe_ptr, amax_ptr, k8_ptr, v8_ptr, sk_ptr, sv_ptr, L, s_kv_l, s_pe_l, s_pe_d,
              DENOM: tl.constexpr, H: tl.constexpr, DN: tl.constexpr, DV: tl.constexpr, DR: tl.constexpr,
              BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    sk = tl.maximum(tl.load(amax_ptr).to(tl.float32, bitcast=True) / DENOM, 1e-12)
    sv = tl.maximum(tl.load(amax_ptr + 1).to(tl.float32, bitcast=True) / DENOM, 1e-12)
    if pid == 0:
        tl.store(sk_ptr, sk)
        tl.store(sv_ptr, sv)
    ik = 1.0 / sk
    iv = 1.0 / sv
    rows = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    rm = rows < L
    dn = tl.arange(0, DN)
    dv = tl.arange(0, DV)
    dr = tl.arange(0, DR)
    pe = tl.load(pe_ptr + rows[:, None] * s_pe_l + dr[None, :] * s_pe_d, mask=rm[:, None], other=0.0)
    pe8 = (pe.to(tl.float32) * ik).to(tl.float8e4b8)
    for h in range(H):
        base = kv_ptr + rows[:, None] * s_kv_l + h * (DN + DV)
        x = tl.load(base + dn[None, :], mask=rm[:, None], other=0.0)
        kb = k8_ptr + rows[:, None] * (H * (DN + DR)) + h * (DN + DR)
        tl.store(kb + dn[None, :], (x.to(tl.float32) * ik).to(tl.float8e4b8), mask=rm[:, None])
        tl.store(kb + DN + dr[None, :], pe8, mask=rm[:, None])
        y = tl.load(base + DN + dv[None, :], mask=rm[:, None], other=0.0)
        vb = v8_ptr + rows[:, None] * (H * DV) + h * DV
        tl.store(vb + dv[None, :], (y.to(tl.float32) * iv).to(tl.float8e4b8), mask=rm[:, None])


def kv_to_fp8(kv, k_pe, H, DN, DV):
    L = kv.shape[0]
    kv = kv.reshape(L, H * (DN + DV))
    assert kv.stride(1) == 1
    pe = k_pe.reshape(L, -1)
    DR = pe.shape[1]
    amax = torch.zeros(2, dtype=torch.int32, device=kv.device)
    k8 = torch.empty((L, H, DN + DR), dtype=FP8, device=kv.device)
    v8 = torch.empty((L, H, DV), dtype=FP8, device=kv.device)
    sk = torch.empty(1, dtype=torch.float32, device=kv.device)
    sv = torch.empty(1, dtype=torch.float32, device=kv.device)
    grid = (triton.cdiv(L, BLOCK_L),)
    _kv_amax[grid](kv, pe, amax, L, kv.stride(0), pe.stride(0), pe.stride(1), H=H, DN=DN, DV=DV, DR=DR, BLOCK=BLOCK_L)
    _kv_quant[grid](kv, pe, amax, k8, v8, sk, sv, L, kv.stride(0), pe.stride(0), pe.stride(1),
                    DENOM=FP8_MAX * MARGIN, H=H, DN=DN, DV=DV, DR=DR, BLOCK=BLOCK_L)
    return k8, v8, sk, sv


def kv_to_fp8_chunked(chunk_kv, L, H, DN, DV, DR, chunk, device):
    """kv_to_fp8 over L rows without materialising the whole bf16 kv_b_proj output.

    chunk_kv(s, e) -> (kv [e-s, H*(DN+DV)] bf16, k_pe [e-s, DR] bf16). Pass 1 accumulates the global amax over all
    chunks; pass 2 recomputes each chunk and quantizes it into the preallocated fp8 K/V with that global scale.
    """
    amax = torch.zeros(2, dtype=torch.int32, device=device)
    k8 = torch.empty((L, H, DN + DR), dtype=FP8, device=device)
    v8 = torch.empty((L, H, DV), dtype=FP8, device=device)
    sk = torch.empty(1, dtype=torch.float32, device=device)
    sv = torch.empty(1, dtype=torch.float32, device=device)
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        kv, pe = chunk_kv(s, e)
        _kv_amax[(triton.cdiv(e - s, BLOCK_L),)](
            kv, pe, amax, e - s, kv.stride(0), pe.stride(0), pe.stride(1),
            H=H, DN=DN, DV=DV, DR=DR, BLOCK=BLOCK_L)
        del kv, pe
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        kv, pe = chunk_kv(s, e)
        _kv_quant[(triton.cdiv(e - s, BLOCK_L),)](
            kv, pe, amax, k8[s:e], v8[s:e], sk, sv, e - s, kv.stride(0), pe.stride(0), pe.stride(1),
            DENOM=FP8_MAX * MARGIN, H=H, DN=DN, DV=DV, DR=DR, BLOCK=BLOCK_L)
        del kv, pe
    return k8, v8, sk, sv


@triton.jit
def _flat_amax(x_ptr, amax_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
    tl.atomic_max(amax_ptr, tl.max(tl.abs(x.to(tl.float32)), axis=0).to(tl.int32, bitcast=True))


@triton.jit
def _flat_quant(x_ptr, amax_ptr, y_ptr, s_ptr, N, DENOM: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    s = tl.maximum(tl.load(amax_ptr).to(tl.float32, bitcast=True) / DENOM, 1e-12)
    if pid == 0:
        tl.store(s_ptr, s)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(y_ptr + offs, (x.to(tl.float32) * (1.0 / s)).to(tl.float8e4b8), mask=m)


@triton.jit
def _dcp_kv_write(kv_ptr, src_ptr, loc_ptr, s_src, s_kv, D: tl.constexpr, BLOCK: tl.constexpr,
                  DCP: tl.constexpr, RANK: tl.constexpr):
    pid = tl.program_id(0)
    loc = tl.load(loc_ptr + pid).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = (offs < D) & (loc % DCP == RANK)
    x = tl.load(src_ptr + pid.to(tl.int64) * s_src + offs, mask=m, other=0.0)
    tl.store(kv_ptr + (loc // DCP) * s_kv + offs, x, mask=m)


def set_mla_kv_dcp(kv_buffer, loc, src, dcp_size, dcp_rank):
    """Row t of src goes to local row loc[t] // dcp_size, only on the rank with loc[t] % dcp_size == dcp_rank.

    kv_buffer [N, 1, D] in the pool dtype, loc [T] virtual slots, src [T, D] or [T, 1, D] with unit stride on D.
    The value is cast on store exactly as mla_buffer.set_mla_kv_buffer_fp8_quant_kernel does. Static shape and no
    host sync, so it can be captured in a CUDA graph (MLATokenToKVPool.set_kv_buffer filters with a host sync and
    then writes the undivided loc).
    """
    T = loc.numel()
    if T == 0:
        return
    src2 = src.select(1, 0) if src.dim() == 3 else src
    D = src2.shape[1]
    assert src2.stride(1) == 1 and kv_buffer.shape[-1] == D and kv_buffer.stride(-1) == 1
    _dcp_kv_write[(T,)](kv_buffer, src2, loc, src2.stride(0), kv_buffer.stride(0), D=D,
                        BLOCK=triton.next_power_of_2(D), DCP=dcp_size, RANK=dcp_rank)


def quant(x):
    x = x.contiguous()
    n = x.numel()
    amax = torch.zeros(1, dtype=torch.int32, device=x.device)
    y = torch.empty(x.shape, dtype=FP8, device=x.device)
    s = torch.empty(1, dtype=torch.float32, device=x.device)
    blk = 8192
    grid = (triton.cdiv(n, blk),)
    _flat_amax[grid](x, amax, n, BLOCK=blk)
    _flat_quant[grid](x, amax, y, s, n, DENOM=FP8_MAX * MARGIN, BLOCK=blk)
    return y, s
