"""gfx942 Triton kernels for the DeepSeek-V4.1 low-ratio indexer: the FP8 MQA prefill operands and the FP4 index-K store.

fake_fp4_query_to_fp8: the RoPE'd bf16 query -> rope_tail_fake_quant_fp4's fp4 round-trip (rounded to bf16 as that
kernel stores it) -> e4m3fnuz with a power-of-two scale per (row, head), and the head weights times that scale. A
power-of-two scale keeps every fp4-grid value exact, which the amax / 240 scale of the torch path does not.

fp4_paged_k_to_fp8: _fp4_paged_k_to_fp8's paged split-MXFP4 index K -> contiguous e4m3fnuz [length, 128] and its
power-of-two per-position scale, in one launch instead of a chain of torch ops per 16384 positions.

quantize_store_fp4_index_k_split: quantize_fp4_indexer_tensor followed by _store_fp4_index_k_split_kernel, in one launch.
"""

import logging

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.attention.dsv4.fp4_indexer_pack_compat import quantize_fp4_indexer_row
from sglang.kernels.ops.attention.dsv4.torch_quant import FP4_AMAX_FLOOR, fake_quant_fp4

logger = logging.getLogger(__name__)


@triton.jit
def _ceil_pow2(x):
    """2 ** ceil(log2(x)) for positive fp32 x, on the IEEE bits (exact at powers of two)."""
    bits = x.to(tl.int32, bitcast=True)
    expo = ((bits >> 23) & 0xFF) - 127
    expo = expo + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    return ((expo + 127) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _fake_fp4_query_to_fp8_kernel(
    q_ptr,
    w_ptr,
    q8_ptr,
    w8_ptr,
    rows_heads,
    D: tl.constexpr,
    BLK: tl.constexpr,
    BH: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
):
    rh = tl.program_id(0) * BH + tl.arange(0, BH)
    mask = rh < rows_heads
    d = tl.arange(0, D)
    offs = rh[:, None].to(tl.int64) * D + d[None, :]
    x = tl.load(q_ptr + offs, mask=mask[:, None], other=0.0).to(tl.float32)

    xb = tl.reshape(x, (BH, D // BLK, BLK))
    amax = tl.maximum(tl.max(tl.abs(xb), axis=2), AMAX_FLOOR) * (1.0 / 6.0)
    scale = _ceil_pow2(amax)
    s = xb / scale[:, :, None]
    s = tl.minimum(tl.maximum(s, -6.0), 6.0)
    mag = tl.abs(s)
    step = tl.where(mag < 2.0, 0.5, tl.where(mag < 4.0, 1.0, 2.0))
    sgn = tl.where(s > 0, 1.0, tl.where(s < 0, -1.0, 0.0))
    f = libdevice.rint(mag / step) * step * sgn * scale[:, :, None]
    f = tl.reshape(f, (BH, D)).to(tl.bfloat16).to(tl.float32)

    s8 = _ceil_pow2(tl.maximum(tl.max(tl.abs(f), axis=1) / 240.0, 1e-12))
    tl.store(q8_ptr + offs, (f / s8[:, None]).to(q8_ptr.dtype.element_ty), mask=mask[:, None])
    w = tl.load(w_ptr + rh, mask=mask, other=0.0).to(tl.float32)
    tl.store(w8_ptr + rh, w * s8, mask=mask)


def fake_fp4_query_to_fp8(
    query: torch.Tensor, weights: torch.Tensor, q8: torch.Tensor, w8: torch.Tensor
) -> None:
    """query [T, H, D] bf16 (RoPE applied, not yet fp4-rounded), weights [T, H] -> q8 [T, H, D] e4m3fnuz, w8 [T, H] fp32."""
    T, H, D = query.shape
    assert query.is_contiguous() and query.dtype == torch.bfloat16 and D % 32 == 0
    assert weights.shape == (T, H) and weights.is_contiguous()
    assert q8.shape == query.shape and q8.is_contiguous() and w8.shape == (T, H) and w8.is_contiguous()
    n = T * H
    if n == 0:
        return
    BH = 16
    _fake_fp4_query_to_fp8_kernel[(triton.cdiv(n, BH),)](
        query, weights, q8, w8, n, D=D, BLK=32, BH=BH, AMAX_FLOOR=FP4_AMAX_FLOOR, num_warps=4
    )


@triton.jit
def _fp4_paged_k_to_fp8_kernel(
    pay_ptr,
    sc_ptr,
    pt_ptr,
    kv_ptr,
    kvs_ptr,
    length,
    sp_page,
    sp_chunk,
    sp_slot,
    ss_page,
    ss_chunk,
    PAGE: tl.constexpr,
    BP: tl.constexpr,
):
    p = tl.program_id(0) * BP + tl.arange(0, BP)
    mask = p < length
    page = tl.load(pt_ptr + p // PAGE, mask=mask, other=0).to(tl.int64)
    off = (p % PAGE).to(tl.int64)
    e = tl.arange(0, 128)
    c = e // 32
    byte = (e % 32) // 2
    b = tl.load(
        pay_ptr + page[:, None] * sp_page + c[None, :] * sp_chunk + off[:, None] * sp_slot + byte[None, :],
        mask=mask[:, None],
        other=0,
    ).to(tl.int32)
    code = tl.where((e % 2)[None, :] == 0, b & 0xF, b >> 4)
    idx = (off % 16) * 4 + off // 16
    ex = tl.load(
        sc_ptr + page[:, None] * ss_page + c[None, :] * ss_chunk + idx[:, None], mask=mask[:, None], other=0
    ).to(tl.int32)
    emax = tl.max(ex, axis=1)
    shift = tl.maximum(ex - emax[:, None] + 5, -126)

    m = code & 7
    mag = tl.where(
        m < 2,
        m.to(tl.float32) * 0.5,
        (1.0 + 0.5 * (m & 1).to(tl.float32)) * ((((m >> 1) - 1 + 127) << 23).to(tl.float32, bitcast=True)),
    )
    val = tl.where((code & 8) != 0, -mag, mag) * ((shift + 127) << 23).to(tl.float32, bitcast=True)
    val = tl.where(m == 0, 0.0, val)
    tl.store(
        kv_ptr + p[:, None].to(tl.int64) * 128 + e[None, :], val.to(kv_ptr.dtype.element_ty), mask=mask[:, None]
    )
    tl.store(kvs_ptr + p, ((emax - 5) << 23).to(tl.float32, bitcast=True), mask=mask)


def fp4_paged_k_to_fp8(
    payload: torch.Tensor,
    k_scale: torch.Tensor,
    pt_row: torch.Tensor,
    length: int,
    kv: torch.Tensor,
    kv_scale: torch.Tensor,
) -> None:
    """payload uint8 [pages, 1, 4, 64, 16], k_scale (1-byte e8m0) [pages, 1, 4, 64], pt_row [>= ceil(length / 64)]
    -> kv e4m3fnuz [length, 128], kv_scale fp32 [length]."""
    assert payload.dtype == torch.uint8 and payload.dim() == 5 and payload.stride(4) == 1
    assert k_scale.element_size() == 1 and k_scale.dim() == 4 and k_scale.stride(3) == 1
    assert kv.shape[0] >= length and kv.is_contiguous() and kv_scale.is_contiguous()
    if length == 0:
        return
    page = payload.shape[3]
    BP = 16
    _fp4_paged_k_to_fp8_kernel[(triton.cdiv(length, BP),)](
        payload,
        k_scale.view(torch.uint8),
        pt_row,
        kv,
        kv_scale,
        length,
        payload.stride(0),
        payload.stride(2),
        payload.stride(3),
        k_scale.stride(0),
        k_scale.stride(2),
        PAGE=page,
        BP=BP,
        num_warps=4,
    )


@triton.jit
def _quantize_store_fp4_index_k_split_kernel(
    x_ptr,
    payload,
    scale,
    loc,
    page_size: tl.constexpr,
    RNE: tl.constexpr,
):
    token_id = tl.program_id(0)
    row = x_ptr + token_id.to(tl.int64) * 128
    values = tl.load(row + tl.arange(0, 128)).to(tl.float32)
    pair = tl.arange(0, 64)
    v0 = tl.load(row + pair * 2).to(tl.float32)
    v1 = tl.load(row + pair * 2 + 1).to(tl.float32)
    packed, packed_sf = quantize_fp4_indexer_row(values, v0, v1, BLOCK_N=128, GROUP_N=32, RNE=RNE)

    cache_loc = tl.load(loc + token_id)
    page = cache_loc // page_size
    page_offset = cache_loc - page * page_size
    chunk = pair // 16
    byte = pair - chunk * 16
    tl.store(
        payload + page * (4 * page_size * 16) + chunk * (page_size * 16) + page_offset * 16 + byte,
        packed.to(tl.uint8),
    )
    shuffled = (page_offset % 16) * 4 + page_offset // 16
    sf_offsets = tl.arange(0, 4)
    sf_bytes = ((packed_sf >> (sf_offsets * 8)) & 0xFF).to(tl.uint8)
    tl.store(scale + page * (4 * page_size) + sf_offsets * page_size + shuffled, sf_bytes)


_CHECKS = {"kconv": [0, 0], "query": [0, 0], "index_k_store": [0, 0]}
_E2M1_VALUES = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)


def _report(kind, bad, total):
    c = _CHECKS[kind]
    c[0] += 1
    c[1] += int(bad > 0)
    if bad or c[0] <= 4 or c[0] & (c[0] - 1) == 0:
        logger.warning(
            "FP8_MQA_CHECK %s call=%d mismatched=%d/%d calls_with_mismatch=%d", kind, c[0], bad, total, c[1]
        )


def check_fp4_paged_k_to_fp8(payload, k_scale, pt_row, length, kv, kv_scale) -> None:
    """Byte-compare fp4_paged_k_to_fp8's output with the torch arithmetic it replaces (SGLANG_DSV41_FP8_MQA_CHECK)."""
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=payload.device)
    p = torch.arange(length, device=payload.device)
    page = pt_row[p // 64].long()
    off = p % 64
    pay = payload[page, 0, :, off, :].reshape(length, 64)
    codes = torch.stack((pay & 0x0F, pay >> 4), dim=-1).reshape(length, 128)
    exps = k_scale[page, 0, :, (off % 16) * 4 + off // 16].to(torch.int32)
    emax = exps.amax(dim=1)
    shift = (exps - emax[:, None] + 5).repeat_interleave(32, dim=1).float()
    ref = (lut[codes.long()] * torch.exp2(shift)).to(kv.dtype)
    ref_s = torch.exp2((emax - 132).float())
    bad = (ref.view(torch.uint8) != kv[:length].view(torch.uint8)).sum().item()
    bad += (ref_s != kv_scale[:length]).sum().item()
    _report("kconv", bad, length * 129)


def check_fake_fp4_query_to_fp8(query, weights, q8, w8) -> None:
    """Byte-compare fake_fp4_query_to_fp8 with fake_quant_fp4 + a power-of-two fp8 scale in torch."""
    fq = fake_quant_fp4(query).float()
    bits = (fq.abs().amax(dim=-1) / 240.0).clamp(min=1e-12).contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    s8 = ((e + 127) << 23).view(torch.float32)
    ref = (fq / s8[..., None]).to(q8.dtype)
    bad = (ref.view(torch.uint8) != q8.view(torch.uint8)).sum().item()
    bad += (weights.float() * s8 != w8).sum().item()
    _report("query", bad, q8.numel() + w8.numel())


def check_quantize_store_fp4_index_k_split(input, payload, scale, loc, page_size, rne) -> None:
    """Compare the values quantize_store_fp4_index_k_split wrote at loc with the values the unfused path stores,
    quantize_fp4_indexer_tensor(fake_quant_fp4(input)). Rows sharing a slot are skipped (last writer is unordered).
    Skipped under graph capture (it reads results back)."""
    if torch.cuda.is_current_stream_capturing():
        return
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import quantize_fp4_indexer_tensor

    x = input.contiguous().view(-1, 128)
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=x.device)
    loc_l = loc.long()
    page, off = loc_l // page_size, loc_l % page_size
    pay = payload.view(torch.uint8)[page, 0, :, off, :].reshape(-1, 64).to(torch.int64)
    codes = torch.stack((pay & 0xF, pay >> 4), dim=-1).reshape(-1, 128)
    exps = scale.view(torch.uint8)[page, 0, :, (off % 16) * 4 + off // 16].to(torch.int64)
    vals = lut[codes] * torch.exp2((exps - 127).float()).repeat_interleave(32, dim=1)
    q, sf = quantize_fp4_indexer_tensor(fake_quant_fp4(x), rne=rne)
    rb = q.view(torch.uint8).to(torch.int64)
    rcodes = torch.stack((rb & 0xF, rb >> 4), dim=-1).reshape(-1, 128)
    rexps = torch.stack([(sf.to(torch.int64) >> (8 * c)) & 0xFF for c in range(4)], dim=-1)
    ref = lut[rcodes] * torch.exp2((rexps - 127).float()).repeat_interleave(32, dim=1)
    _, inv, counts = torch.unique(loc_l, return_inverse=True, return_counts=True)
    unique_row = counts[inv] == 1
    diff = (vals != ref) & unique_row[:, None]
    bad = diff.sum().item()
    if bad:
        blk_amax = x.float().abs().view(-1, 4, 32).amax(-1).repeat_interleave(32, dim=1)
        logger.warning(
            "FP8_MQA_CHECK index_k_store detail: dup_rows=%d bad_vals=%d max_block_amax_of_bad=%.3e",
            int((~unique_row).sum().item()), bad, blk_amax[diff].max().item(),
        )
    _report("index_k_store", bad, int(unique_row.sum().item()) * 128)


def quantize_store_fp4_index_k_split(
    input: torch.Tensor,
    payload: torch.Tensor,
    scale: torch.Tensor,
    loc: torch.Tensor,
    *,
    page_size: int,
    rne: bool,
) -> None:
    """store_fp4_index_k_cache_split in one launch: quantize input [n, 128] to fp4 (per-32 ue8m0) and scatter row i to
    slot loc[i] of the split FlyDSL K layout."""
    x = input.contiguous().view(-1, 128)
    n = x.shape[0]
    if n == 0:
        return
    _quantize_store_fp4_index_k_split_kernel[(n,)](
        x, payload.view(torch.uint8), scale, loc, page_size, RNE=rne
    )
