// gfx942 decode index logits over the paged split-MXFP4 index K:
//   out[row, p] = bf16(sum_h bf16(bf16(q[row, h] . k[p])^+ * w[row, h])) for p < lengths[row], -inf for the rest of a
// scored 128-position tile. q holds FP4 values (e2m1 * 2^e per 32 dims) in bf16, so both operands go to the fp8 MFMA
// exactly: q per head scaled by a power of two (folded into the head weight), k per position (applied after the head
// sum). Each wave scores 32 positions of one row with all 32 * MB heads: lane l is position l % 32, dim half l / 32.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef float f32x16 __attribute__((ext_vector_type(16)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef unsigned short u16x2 __attribute__((ext_vector_type(2)));

struct __attribute__((packed)) Args {
    const uint16_t* q;          // bf16 [rows, 32 * MB, 128]
    const uint16_t* w;          // bf16 [rows, 32 * MB]
    const uint8_t* payload;     // [pages, 1, 4, 64, 16]: page, 32-dim chunk, slot, byte (dims 2b, 2b + 1)
    const uint8_t* scale;       // [pages, 1, 4, 64] e8m0, slot s stored at (s % 16) * 4 + s / 16
    const int32_t* page_table;  // [rows, pt_stride]
    const int32_t* lengths;     // [rows]
    float* out;                 // [rows, out_stride]
    int32_t max_seq_len, pt_stride, out_stride, q_row_stride, q_head_stride, w_row_stride, workers;
};

// fp32 -> bf16 round-to-nearest-even, kept in fp32, two at a time: Veltkamp's split at 16 bits (exhaustively equal
// to the integer rounding over every 24-bit mantissa). Contraction would fold t - x into an exact FMA.
__device__ __forceinline__ f32x2 rne_bf16x2(f32x2 x) {
#pragma clang fp contract(off)
    const f32x2 t = x * 65537.f;
    return t - (t - x);
}

__device__ __forceinline__ float rne_bf16(float x) {
    const uint32_t u = __float_as_uint(x);
    return __uint_as_float((u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u);
}

__device__ __forceinline__ float bf16_lo(uint32_t d) { return __uint_as_float(d << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t d) { return __uint_as_float(d & 0xffff0000u); }

// Step s = 2c + t of the 128-dim dot product: k slots 8h..8h+7 of the 32x32x16 MFMA are dims
// 32c + 16h + 8t + {0, 2, 4, 6, 1, 3, 5, 7}, the order a split-MXFP4 dword unpacks into (low nibbles, high nibbles).
__device__ __forceinline__ uint64_t q_operand(u32x4 d, float inv) {
    int lo = __builtin_amdgcn_cvt_pk_fp8_f32(bf16_lo(d[0]) * inv, bf16_lo(d[1]) * inv, 0, false);
    lo = __builtin_amdgcn_cvt_pk_fp8_f32(bf16_lo(d[2]) * inv, bf16_lo(d[3]) * inv, lo, true);
    int hi = __builtin_amdgcn_cvt_pk_fp8_f32(bf16_hi(d[0]) * inv, bf16_hi(d[1]) * inv, 0, false);
    hi = __builtin_amdgcn_cvt_pk_fp8_f32(bf16_hi(d[2]) * inv, bf16_hi(d[3]) * inv, hi, true);
    return (uint64_t)(uint32_t)lo | ((uint64_t)(uint32_t)hi << 32);
}

// Eight e2m1 codes -> e4m3fnuz through an 8-entry magnitude table whose exponents already carry the chunk's shift;
// the sign is set only on nonzero magnitudes (0x80 is NaN in fnuz).
__device__ __forceinline__ uint64_t k_operand(uint32_t x, uint32_t lut_lo, uint32_t lut_hi) {
    const uint32_t me = __builtin_amdgcn_perm(lut_hi, lut_lo, x & 0x07070707u);
    const uint32_t mo = __builtin_amdgcn_perm(lut_hi, lut_lo, (x >> 4) & 0x07070707u);
    const uint32_t ne = (me + 0x7f7f7f7fu) & 0x80808080u;
    const uint32_t no = (mo + 0x7f7f7f7fu) & 0x80808080u;
    const uint32_t oe = ((x << 4) & ne) | me;
    const uint32_t oo = (x & no) | mo;
    return (uint64_t)oe | ((uint64_t)oo << 32);
}

struct KTile {
    u32x2 kr[4];
    int e[4];
};

__device__ __forceinline__ KTile load_k(const uint8_t* payload, const uint8_t* scale, int page, int slot, int h) {
    KTile t;
    const uint8_t* kp = payload + (size_t)page * 4096 + slot * 16 + 8 * h;
    const uint8_t* sp = scale + (size_t)page * 256 + (((slot & 15) << 2) | (slot >> 4));
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        t.kr[c] = *(const u32x2*)(kp + c * 1024);
        t.e[c] = sp[c * 64];
    }
    return t;
}

template <int MB>
__device__ __forceinline__ void decode_logits(const Args& a) {
    __shared__ float s_w[4][32 * MB];
    const int lane = threadIdx.x & 63;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int m = lane & 31, h = lane >> 5;
    const int row = blockIdx.x / a.workers, worker = blockIdx.x - row * a.workers;
    const int visible = min(a.lengths[row], a.max_seq_len);
    const int n_tiles = (visible + 127) >> 7;
    if (worker >= n_tiles) return;

    // Query operands of heads mb * 32 + m. The largest |q| of a head, 1 or 1.5 * 2^E, lands at 2^7 or 1.5 * 2^7.
    uint64_t aq[MB][8];
#pragma unroll
    for (int mb = 0; mb < MB; ++mb) {
        const int head = mb * 32 + m;
        const uint16_t* qh = a.q + (size_t)row * a.q_row_stride + (size_t)head * a.q_head_stride + 16 * h;
        u32x4 raw[8];
#pragma unroll
        for (int s = 0; s < 8; ++s) raw[s] = *(const u32x4*)(qh + 32 * (s >> 1) + 8 * (s & 1));
        u16x2 mx = {0, 0};
#pragma unroll
        for (int s = 0; s < 8; ++s)
#pragma unroll
            for (int d = 0; d < 4; ++d)
                mx = __builtin_elementwise_max(mx, __builtin_bit_cast(u16x2, (uint32_t)(raw[s][d] & 0x7fff7fffu)));
        uint32_t amax = max((uint32_t)mx[0], (uint32_t)mx[1]);
        amax = max(amax, (uint32_t)__shfl_xor((int)amax, 32));
        const int ef = (int)(amax >> 7);  // biased exponent of the head's max |q|; 0 for an all-zero head
        const float inv = __uint_as_float((uint32_t)min(max(261 - ef, 1), 254) << 23);  // 2^(134 - ef)
#pragma unroll
        for (int s = 0; s < 8; ++s) aq[mb][s] = q_operand(raw[s], inv);
        if (h == 0) {
            const float wv = bf16_lo(a.w[(size_t)row * a.w_row_stride + head]);
            s_w[wave][head] = ef >= 8 ? wv * __uint_as_float((uint32_t)min(ef - 7, 254) << 23) : 0.f;  // 2^(ef - 134)
        }
    }
    __syncthreads();
    // Epilogue element i of block mb is head mb * 32 + (i / 4) * 8 + h * 4 + i % 4 (32x32 MFMA output layout).
    f32x2 wq[MB][8];
#pragma unroll
    for (int mb = 0; mb < MB; ++mb)
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const float4 v = *(const float4*)&s_w[wave][mb * 32 + g * 8 + h * 4];
            wq[mb][2 * g] = f32x2{v.x, v.y};
            wq[mb][2 * g + 1] = f32x2{v.z, v.w};
        }

    const int32_t* pt = a.page_table + (size_t)row * a.pt_stride;
    float* orow = a.out + (size_t)row * a.out_stride;
    int tile = worker;
    int p0 = tile * 128 + wave * 32;
    KTile cur = load_k(a.payload, a.scale, p0 < visible ? pt[p0 >> 6] : 0, (p0 & 63) + m, h);
    for (; tile < n_tiles; tile += a.workers) {
        const int np0 = p0 + a.workers * 128;
        KTile nxt = cur;
        if (np0 < visible) nxt = load_k(a.payload, a.scale, pt[np0 >> 6], (np0 & 63) + m, h);
        if (p0 >= visible) {
            // the rest of the row's last tile reads -inf, as the Triton kernel leaves its last tile
            if (h == 0 && p0 < a.max_seq_len) orow[p0 + m] = -INFINITY;
        } else {
            const int emax = max(max(cur.e[0], cur.e[1]), max(cur.e[2], cur.e[3]));
            // chunk c's codes scaled by 2^(e_c - emax + 5): exact while the chunk is within 2^11 of the position's
            // largest; a chunk below that contributes under 2^-11 of it and is dropped.
            uint64_t kb[8];
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int sh = cur.e[c] - emax + 11;
                const uint32_t s8 = ((uint32_t)max(sh, 0) << 3) & 0xffu;
                const uint32_t inc = s8 * 0x010101u | (s8 << 24);  // s8 in every byte
                const uint32_t lut_lo = sh < 0 ? 0u : 0x14100800u + (inc & 0xffffff00u);
                const uint32_t lut_hi = sh < 0 ? 0u : 0x24201c18u + inc;
                kb[2 * c] = k_operand(cur.kr[c][0], lut_lo, lut_hi);
                kb[2 * c + 1] = k_operand(cur.kr[c][1], lut_lo, lut_hi);
            }
            const float sk = __uint_as_float((uint32_t)max(emax - 5, 1) << 23);  // 2^(emax - 132)

            f32x16 acc[MB];
#pragma unroll
            for (int mb = 0; mb < MB; ++mb) acc[mb] = f32x16{};
#pragma unroll
            for (int s = 0; s < 8; ++s)
#pragma unroll
                for (int mb = 0; mb < MB; ++mb)
                    acc[mb] = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8((long)aq[mb][s], (long)kb[s], acc[mb], 0, 0, 0);
            f32x2 v[8 * MB];
#pragma unroll
            for (int mb = 0; mb < MB; ++mb)
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const f32x2 sc = rne_bf16x2(f32x2{acc[mb][2 * i], acc[mb][2 * i + 1]});
                    v[8 * mb + i] = rne_bf16x2(__builtin_elementwise_max(sc, f32x2{0.f, 0.f}) * wq[mb][i]);
                }
#pragma unroll
            for (int n = 4 * MB; n >= 1; n >>= 1)
#pragma unroll
                for (int i = 0; i < n; ++i) v[i] += v[i + n];
            const float part = v[0][0] + v[0][1];
            const float sum = part + __shfl_xor(part, 32);
            if (h == 0) {
                const int p = p0 + m;
                orow[p] = p < visible ? rne_bf16(sum) * sk : -INFINITY;
            }
        }
        cur = nxt;
        p0 = np0;
    }
}

extern "C" __global__ void __launch_bounds__(256) fp4_decode_logits_gfx942_h32(Args a) { decode_logits<1>(a); }
extern "C" __global__ void __launch_bounds__(256) fp4_decode_logits_gfx942_h64(Args a) { decode_logits<2>(a); }
