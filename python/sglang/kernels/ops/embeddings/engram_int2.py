"""DeepSeek-V4.1 engram INT2 packed table: 2-bit symmetric codes + e8m0 block
scales, dequantized to bf16 on lookup. Mirrors engram_gather fp8 arithmetic but
decodes 4 packed 2-bit codes per byte through a symmetric codebook
[-1, -r, +r, +1]. Scheme name: dsv41_engram_int2_v1.

Contract (dsv41_engram_int2_v1):
  weight : u8 [rows, dim // 4]        4 consecutive 2-bit codes LSB-first per byte
  scale  : u8 [rows, dim // group]    e8m0 exponent bytes
  dim          = 256
  group        = 32   (== FP8_BLOCK_SIZE)
  codebook     = [-1, -r, +r, +1] for codes [0, 1, 2, 3]
  dequant      = level[code] * 2 ** (exp - 127), rounded once to bf16

The quantize / dequant math here is the single source of truth shared by: the
packed Triton gather kernel, the torch CPU/GPU oracle, the on-the-fly model
simulation, and any future offline converter.
"""

import torch
import triton
import triton.language as tl

SCHEME = "dsv41_engram_int2_v1"
INT2_DIM = 256
INT2_GROUP = 32
INT2_PACK = INT2_DIM // 4
# Default codebook r, calibrated in Phase 3 on 40960 real engram rows (layers 1
# and 14): group32 r=0.30 gives NMSE 12.65% / cosine 0.936, the min-NMSE point of
# the symmetric 4-level codebook. Overridable via SGLANG_DSV41_ENGRAM_INT2_R.
DEFAULT_R = 0.30000001192092896
_E8M0_ZERO = 2.0 ** -127


def levels(r: float) -> torch.Tensor:
    return torch.tensor([-1.0, -r, r, 1.0], dtype=torch.float32)


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """codes u8 [rows, dim] in {0..3} -> packed u8 [rows, dim//4], LSB-first."""
    rows, dim = codes.shape
    assert dim % 4 == 0
    c = codes.view(rows, dim // 4, 4).to(torch.int32)
    packed = c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)
    return packed.to(torch.uint8)


def unpack_codes(packed: torch.Tensor, dim: int = INT2_DIM) -> torch.Tensor:
    """packed u8 [rows, dim//4] -> codes u8 [rows, dim] in {0..3}."""
    rows = packed.shape[0]
    p = packed.to(torch.int32)
    c0 = p & 3
    c1 = (p >> 2) & 3
    c2 = (p >> 4) & 3
    c3 = (p >> 6) & 3
    codes = torch.stack([c0, c1, c2, c3], dim=-1).view(rows, dim)
    return codes.to(torch.uint8)


def _e8m0_scale(byte_u8: torch.Tensor) -> torch.Tensor:
    """e8m0 exponent byte -> fp32 power-of-two scale 2**(byte-127).
    byte 0 -> 2**-127, matching the kernel special case exactly."""
    return torch.exp2(byte_u8.to(torch.float32) - 127.0)


# e8m0 exponent offsets searched per group relative to ceil(log2(amax)). The
# fixed outer level is +/-1, so exp=ceil keeps all data within [-1, 1] (no clip),
# while exp=ceil-1/-2 trade a little tail clipping for ~2x/4x inner resolution.
# Each group independently keeps whichever exponent minimizes its reconstruction
# MSE, which the same kernel decodes unchanged.
_EXP_SEARCH = (-2, -1, 0)


def quantize_int2(x: torch.Tensor, r: float = DEFAULT_R, group: int = INT2_GROUP):
    """x [rows, dim] -> (packed u8 [rows, dim//4], scale u8 [rows, dim//group]).

    Exact contract quantizer: per group search an e8m0 power-of-two scale and the
    nearest symmetric codebook level that minimize group MSE. Single source of
    truth shared by kernel-data prep, oracle, simulation and the future converter.
    """
    assert x.dim() == 2 and x.shape[1] % group == 0, x.shape
    rows, dim = x.shape
    ng = dim // group
    g = x.float().view(rows, ng, group)
    amax = g.abs().amax(dim=-1)
    nonzero = amax > 0
    ceil_exp = torch.ceil(torch.log2(amax.clamp_min(1e-30)))
    lv = levels(r).to(x.device)
    best_mse = torch.full((rows, ng), float("inf"), device=x.device)
    best_byte = torch.zeros((rows, ng), dtype=torch.uint8, device=x.device)
    best_code = torch.zeros((rows, ng, group), dtype=torch.uint8, device=x.device)
    for de in _EXP_SEARCH:
        byte = (ceil_exp + de + 127.0).clamp(0.0, 254.0)
        byte = torch.where(nonzero, byte, torch.zeros_like(byte))
        scale = _e8m0_scale(byte).unsqueeze(-1)
        n = g / scale
        code = (n.unsqueeze(-1) - lv).abs().argmin(dim=-1)
        recon = lv[code] * scale
        mse = (recon - g).pow(2).mean(dim=-1)
        take = mse < best_mse
        best_mse = torch.where(take, mse, best_mse)
        best_byte = torch.where(take, byte.to(torch.uint8), best_byte)
        best_code = torch.where(take.unsqueeze(-1), code.to(torch.uint8), best_code)
    packed = pack_codes(best_code.view(rows, dim))
    return packed, best_byte


def dequant_int2_torch(
    packed: torch.Tensor,
    scale_u8: torch.Tensor,
    r: float = DEFAULT_R,
    dim: int = INT2_DIM,
    group: int = INT2_GROUP,
) -> torch.Tensor:
    """CPU/GPU oracle: packed u8 [rows, dim//4], scale u8 [rows, dim//group]
    -> bf16 [rows, dim]. Exactly the kernel arithmetic."""
    rows = packed.shape[0]
    codes = unpack_codes(packed, dim).long()
    lv = levels(r).to(packed.device)
    level = lv[codes]
    scale = _e8m0_scale(scale_u8).repeat_interleave(group, dim=1)
    return (level * scale).to(torch.bfloat16)


def simulate_int2(
    emb: torch.Tensor, r: float = DEFAULT_R, group: int = INT2_GROUP
) -> torch.Tensor:
    """On-the-fly quality model: emb [..., dim] -> same shape, after INT2
    quantize + dequant. Step-for-step identical to the converter + kernel math."""
    shape = emb.shape
    dim = shape[-1]
    x = emb.reshape(-1, dim)
    if x.shape[0] == 0:
        return emb
    packed, scale_u8 = quantize_int2(x, r, group)
    deq = dequant_int2_torch(packed, scale_u8, r, dim, group)
    return deq.reshape(shape).to(emb.dtype)


@triton.jit
def _engram_int2_gather_kernel(
    w_ptr,
    s_ptr,
    ids_ptr,
    out_ptr,
    row_lo,
    row_hi,
    ids_cols,
    ids_row_stride,
    ids_col_stride,
    R,
    DIM: tl.constexpr,
    BLK: tl.constexpr,
    PACK: tl.constexpr,
    E8M0_ZERO: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    id_row = row // ids_cols
    id_col = row - id_row * ids_cols
    id_offset = id_row * ids_row_stride + id_col * ids_col_stride
    idx = tl.load(ids_ptr + id_offset).to(tl.int64)
    # Table holds rows [row_lo, row_hi); ids outside it come out as zero rows,
    # which is what the sharded all-reduce sums.
    owned = (idx >= row_lo) & (idx < row_hi)
    local = tl.where(owned, idx - row_lo, 0)
    w = w_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    s = s_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    offs = tl.arange(0, DIM)
    byte_idx = offs // 4
    shift = (offs % 4) * 2
    packed = tl.load(w + local * PACK + byte_idx, mask=owned, other=0).to(tl.int32)
    code = (packed >> shift) & 3
    # Symmetric codebook [-1, -r, +r, +1] as sign * magnitude, no lookup table.
    sign = tl.where(code < 2, -1.0, 1.0)
    mag = tl.where((code == 0) | (code == 3), 1.0, R)
    level = sign * mag
    exps = tl.load(s + local * (DIM // BLK) + offs // BLK, mask=owned, other=0).to(
        tl.int32
    )
    scale = (exps << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exps == 0, E8M0_ZERO, scale)
    out = tl.where(owned, level * scale, 0.0)
    tl.store(out_ptr + row * DIM + offs, out.to(tl.bfloat16))


def engram_int2_gather(
    weight_ptr: int,
    scale_ptr: int,
    ids: torch.Tensor,
    out: torch.Tensor,
    dim: int,
    block_size: int,
    r: float = DEFAULT_R,
    row_lo: int = 0,
    row_hi: int = 2 ** 62,
) -> torch.Tensor:
    """Gather packed INT2 rows ids ([N] or [N, C] int) into out
    ([N*C, dim] bf16, contiguous). weight_ptr addresses [rows, dim//4] u8
    packed codes and scale_ptr [rows, dim//block_size] e8m0 bytes for global
    rows [row_lo, row_hi); both may live in device or host memory. Ids outside
    the range produce zero rows. Capture-safe: raw pointers, 1 program per id."""
    assert dim > 0 and dim & (dim - 1) == 0 and block_size > 0, (dim, block_size)
    assert dim % block_size == 0 and dim % 4 == 0, (dim, block_size)
    assert ids.is_cuda and ids.dtype in (torch.int32, torch.int64)
    assert ids.dim() in (1, 2), ids.shape
    assert out.is_contiguous() and out.dtype == torch.bfloat16
    assert out.device == ids.device and out.shape == (ids.numel(), dim), out.shape
    n = ids.numel()
    if n:
        _engram_int2_gather_kernel[(n,)](
            weight_ptr,
            scale_ptr,
            ids,
            out,
            row_lo,
            row_hi,
            ids.shape[1] if ids.dim() == 2 else 1,
            ids.stride(0),
            ids.stride(1) if ids.dim() == 2 else 1,
            float(r),
            DIM=dim,
            BLK=block_size,
            PACK=dim // 4,
            E8M0_ZERO=_E8M0_ZERO,
        )
    return out
