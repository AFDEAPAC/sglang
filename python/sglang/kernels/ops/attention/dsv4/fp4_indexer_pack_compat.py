"""Triton helpers the fused index-K store needs, lifted from the 20261030 tree.

`fp8_mqa_pack_hip.py` imports `quantize_fp4_indexer_row` from `fp4_indexer`,
but this tree's `fp4_indexer` is an older generation without it or the four
device functions it calls. They are self-contained -- only triton.language --
so they live here rather than being grafted into a module whose other contents
0.5.22 code depends on.

Copied verbatim from
sglang/kernels/ops/attention/dsv4/fp4_indexer.py of the ali-private
20261030 image.
"""

import triton
import triton.language as tl


@triton.jit
def _select_group_value(group, v0, v1, v2, v3):
    return tl.where(
        group == 0,
        v0,
        tl.where(group == 1, v1, tl.where(group == 2, v2, v3)),
    )


@triton.jit
def _ceil_ue8m0_exp(x):
    bits = x.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    exp += mantissa != 0
    return tl.minimum(tl.maximum(exp, 1), 254)


@triton.jit
def _fp4_e2m1_code(x):
    ax = tl.minimum(tl.abs(x), 6.0)
    idx = (ax > 0.25).to(tl.uint8)
    idx += (ax > 0.75).to(tl.uint8)
    idx += (ax > 1.25).to(tl.uint8)
    idx += (ax > 1.75).to(tl.uint8)
    idx += (ax > 2.5).to(tl.uint8)
    idx += (ax > 3.5).to(tl.uint8)
    idx += (ax > 5.0).to(tl.uint8)
    sign = ((x < 0) & (idx != 0)).to(tl.uint8)
    return idx | (sign << 3)


@triton.jit
def _fp4_e2m1_code_rne(x):
    """Round-to-nearest-even e2m1 code, matching the reference rounding."""
    ax = tl.minimum(tl.abs(x), 6.0)
    idx = (ax >= 0.25).to(tl.uint8)
    idx += (ax >= 0.75).to(tl.uint8)
    idx += (ax >= 1.25).to(tl.uint8)
    idx += (ax >= 1.75).to(tl.uint8)
    idx += (ax >= 2.5).to(tl.uint8)
    idx += (ax >= 3.5).to(tl.uint8)
    idx += (ax >= 5.0).to(tl.uint8)
    # Round-half-to-even: an odd index at an exact boundary drops to the even one.
    is_boundary = (
        (ax == 0.25)
        | (ax == 0.75)
        | (ax == 1.25)
        | (ax == 1.75)
        | (ax == 2.5)
        | (ax == 3.5)
        | (ax == 5.0)
    )
    idx = tl.where(is_boundary & ((idx & 1) == 1), idx - 1, idx)
    sign = ((x < 0) & (idx != 0)).to(tl.uint8)
    return idx | (sign << 3)


@triton.jit
def quantize_fp4_indexer_row(
    values,
    v0,
    v1,
    BLOCK_N: tl.constexpr,
    GROUP_N: tl.constexpr,
    RNE: tl.constexpr,
):
    """One row of quantize_fp4_indexer_tensor: values is the fp32 [BLOCK_N] row,
    v0 / v1 its even / odd elements as fp32 [BLOCK_N // 2]. Returns
    (packed codes [BLOCK_N // 2], packed ue8m0 exponents), the two stores of the
    standalone kernel below."""
    offs = tl.arange(0, BLOCK_N)
    abs_values = tl.abs(values)

    amax0 = tl.max(tl.where(offs < GROUP_N, abs_values, 0.0), axis=0)
    amax1 = tl.max(
        tl.where((GROUP_N <= offs) & (offs < 2 * GROUP_N), abs_values, 0.0),
        axis=0,
    )
    amax2 = tl.max(
        tl.where((2 * GROUP_N <= offs) & (offs < 3 * GROUP_N), abs_values, 0.0),
        axis=0,
    )
    amax3 = tl.max(tl.where(3 * GROUP_N <= offs, abs_values, 0.0), axis=0)

    sf0 = tl.maximum(amax0 / 6.0, 1.0e-4)
    sf1 = tl.maximum(amax1 / 6.0, 1.0e-4)
    sf2 = tl.maximum(amax2 / 6.0, 1.0e-4)
    sf3 = tl.maximum(amax3 / 6.0, 1.0e-4)

    exp0 = _ceil_ue8m0_exp(sf0)
    exp1 = _ceil_ue8m0_exp(sf1)
    exp2 = _ceil_ue8m0_exp(sf2)
    exp3 = _ceil_ue8m0_exp(sf3)

    packed_sf = exp0 | (exp1 << 8) | (exp2 << 16) | (exp3 << 24)

    pair_offsets = tl.arange(0, BLOCK_N // 2)
    offs0 = pair_offsets * 2
    offs1 = offs0 + 1
    group0 = offs0 // GROUP_N
    group1 = offs1 // GROUP_N
    scale_exp0 = _select_group_value(group0, exp0, exp1, exp2, exp3)
    scale_exp1 = _select_group_value(group1, exp0, exp1, exp2, exp3)
    scale0 = (scale_exp0 << 23).to(tl.float32, bitcast=True)
    scale1 = (scale_exp1 << 23).to(tl.float32, bitcast=True)

    v0 = v0 / scale0
    v1 = v1 / scale1
    if RNE:
        code0 = _fp4_e2m1_code_rne(v0)
        code1 = _fp4_e2m1_code_rne(v1)
    else:
        code0 = _fp4_e2m1_code(v0)
        code1 = _fp4_e2m1_code(v1)
    packed = (code0 & 0x0F) | ((code1 & 0x0F) << 4)
    return packed, packed_sf
