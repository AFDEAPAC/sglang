// gfx942 sparse MLA decode attention, split-K partials in the layout of the Triton dual-scope split-K kernel
// (partial_output [split, token, head, 512] fp32 already divided by l, partial_lse [split, token, head] natural log,
// +inf for a split without valid keys); the Triton par-combine kernel merges them (sink, inverse RoPE, bf16 store).
//
// Keys: per token 448 e4m3fnuz nope dims in 7 chunks of 64 with an e8m0 scale each, then 64 bf16 rope dims; a cache
// block is block_size x 576 data bytes followed by block_size x 8 scale bytes. Two scopes (main top-k, extra SWA)
// address their own caches; a row's keys are positions [0, topk_main + topk_extra), both counts and per_split
// multiples of 16 so a 16-key tile lies in one scope.
//
// QK^T on the fp8 MFMA without dequantizing K: q (bf16) = hi + lo, two e4m3fnuz values after a per-head power-of-two
// scale (exact for bf16 inputs within the fp8 range), and each chunk's partial sum is multiplied by the key's chunk
// scale. The rope part runs on the bf16 MFMA. PV runs on the bf16 MFMA with p rounded to bf16 as in the Triton kernel
// and V = fp8 * 2^e, exact in bf16. Lane l: MFMA column l % 16 (head for S, dim slot for O), group g = l / 16.
// A CTA is 4 independent waves: two token ranges (splits) x two halves of the 512 output dims; both halves of a range
// compute the same QK^T and softmax state. The query operands are built once per CTA into LDS; each wave loads the
// next tile's indices and QK^T operands while it finishes the current tile.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef short i16x4 __attribute__((ext_vector_type(4)));
typedef unsigned short u16x2 __attribute__((ext_vector_type(2)));

struct __attribute__((packed)) Args {
    const uint16_t* q;          // bf16 [T, h_q, 512]
    const uint8_t* kv_main;     // [blocks, block_main * 584]
    const int32_t* idx_main;    // [T, topk_main] contiguous rows, -1 = no key
    const int32_t* len_main;    // [T / s_q] valid keys per request
    const uint8_t* kv_extra;
    const int32_t* idx_extra;
    const int32_t* len_extra;
    float* po;                  // [splits, T, h_q, 512]
    float* plse;                // [splits, T, h_q]
    const uint8_t* zero;        // 584 zero bytes: the key, V row and scales of an invalid key
    float sm_scale;
    int32_t q_t, q_h;
    int32_t kv_main_block, main_shift, idx_main_t;     // block bytes, log2(block size), row stride
    int32_t kv_extra_block, extra_shift, idx_extra_t;
    int32_t po_s, po_t, po_h, plse_s, plse_t, plse_h;
    int32_t T, h_q, s_q, topk_main, topk_extra, per_split, cta_splits;
};

__device__ __forceinline__ float bf16_lo(uint32_t d) { return __uint_as_float(d << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t d) { return __uint_as_float(d & 0xffff0000u); }
__device__ __forceinline__ float pow2_e8m0(uint32_t e) { return __uint_as_float(e << 23); }
__device__ __forceinline__ uint64_t u64(uint32_t a, uint32_t b) { return (uint64_t)a | ((uint64_t)b << 32); }
__device__ __forceinline__ i16x4 bf16x4(uint32_t a, uint32_t b) { return __builtin_bit_cast(i16x4, u64(a, b)); }

__device__ __forceinline__ uint32_t rne_bf16_bits(float x) {
    const uint32_t u = __float_as_uint(x);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

// Eight bf16 q values (d holds dims 2w, 2w + 1 per dword) scaled by s -> hi and lo e4m3fnuz bytes, hi + lo = q * s.
__device__ __forceinline__ void q_split(u32x4 d, float s, uint64_t& hi, uint64_t& lo) {
    float x[8];
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        x[2 * w] = bf16_lo(d[w]) * s;
        x[2 * w + 1] = bf16_hi(d[w]) * s;
    }
    uint32_t h[2], l[2];
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        int hv = __builtin_amdgcn_cvt_pk_fp8_f32(x[4 * w], x[4 * w + 1], 0, false);
        hv = __builtin_amdgcn_cvt_pk_fp8_f32(x[4 * w + 2], x[4 * w + 3], hv, true);
        const f32x2 h01 = __builtin_amdgcn_cvt_pk_f32_fp8(hv, false);
        const f32x2 h23 = __builtin_amdgcn_cvt_pk_f32_fp8(hv, true);
        int lv = __builtin_amdgcn_cvt_pk_fp8_f32(x[4 * w] - h01[0], x[4 * w + 1] - h01[1], 0, false);
        lv = __builtin_amdgcn_cvt_pk_fp8_f32(x[4 * w + 2] - h23[0], x[4 * w + 3] - h23[1], lv, true);
        h[w] = (uint32_t)hv;
        l[w] = (uint32_t)lv;
    }
    hi = u64(h[0], h[1]);
    lo = u64(l[0], l[1]);
}

// LDS query operands, read as 16-byte pairs: pair p of lane l at v[p][l]. Pairs 0-13: chunk c = p / 2,
// (hi j=0, hi j=1) for even p, (lo j=0, lo j=1) for odd p; pairs 14-15: rope steps (0, 1), (2, 3).
struct QOps {
    u32x4 v[16][64];
};

struct Tile {
    const int32_t* idx;   // index row of the tile's scope at the tile's first position
    const uint8_t* kv;
    size_t blk_bytes;
    int len;              // valid positions from the tile's first one (<= 0: none)
    int shift;
};

// The A-operand key n and the C-row keys 4g .. 4g + 3 of a tile.
struct Keys {
    const uint8_t* adata;
    const uint8_t* cdata[4];
    const uint8_t* cscale[4];
    bool cvalid[4];
};

__device__ __forceinline__ Tile tile_at(const Args& a, const int32_t* idx_main_row, const int32_t* idx_extra_row,
                                        int len_main, int len_extra, int tile, int k1) {
    const bool main = tile < a.topk_main;
    const int pos0 = main ? tile : tile - a.topk_main;
    Tile tl;
    tl.idx = (main ? idx_main_row : idx_extra_row) + pos0;
    tl.kv = main ? a.kv_main : a.kv_extra;
    tl.blk_bytes = (size_t)(main ? a.kv_main_block : a.kv_extra_block);
    tl.len = min(main ? len_main : len_extra, pos0 + (k1 - tile)) - pos0;  // keys past k1 are not this range's
    tl.shift = main ? a.main_shift : a.extra_shift;
    return tl;
}

// An invalid key reads the zero bytes (its key, V row and scales), so p = 0 meets finite values only.
__device__ __forceinline__ Keys keys_of(const Tile& tl, const uint8_t* zero, int ia, u32x4 kci, int n, int g) {
    const uint32_t mask = (1u << tl.shift) - 1u;
    Keys k;
    const bool aok = n < tl.len && ia >= 0;
    k.adata = aok ? tl.kv + (size_t)(ia >> tl.shift) * tl.blk_bytes + (ia & mask) * 576u : zero;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int ki = (int)kci[i];
        k.cvalid[i] = 4 * g + i < tl.len && ki >= 0;
        const uint8_t* blk = tl.kv + (size_t)(ki >> tl.shift) * tl.blk_bytes;
        k.cdata[i] = k.cvalid[i] ? blk + (ki & mask) * 576u : zero;
        k.cscale[i] = k.cvalid[i] ? blk + (576u << tl.shift) + (ki & mask) * 8u : zero + 576;
    }
    return k;
}

__device__ __forceinline__ void load_qk(const Keys& k, int g, u32x4 (&kr)[9], u32x2 (&sc)[4]) {
#pragma unroll
    for (int c = 0; c < 7; ++c) kr[c] = *(const u32x4*)(k.adata + 64 * c + 16 * g);
    kr[7] = *(const u32x4*)(k.adata + 448 + 32 * g);
    kr[8] = *(const u32x4*)(k.adata + 448 + 32 * g + 16);
#pragma unroll
    for (int i = 0; i < 4; ++i) sc[i] = *(const u32x2*)k.cscale[i];
}

// PV of chunks C0 .. C0 + NC - 1 into blocks 4cc + k: B row k = key 4g + k, column n = dim 64c + 4n + k.
template <int C0, int NC>
__device__ __forceinline__ void pv_chunks(i16x4 pa4, const uint32_t (&vx)[4][4], const u32x2 (&sc)[4], f32x4 (&o)[16]) {
#pragma unroll
    for (int cc = 0; cc < NC; ++cc) {
        const int c = C0 + cc;
        float v[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t ew = c < 4 ? sc[i][0] >> (8 * c) : sc[i][1] >> (8 * (c - 4));
            const float e = pow2_e8m0(ew & 0xffu);
            const f32x2 lo = __builtin_amdgcn_cvt_pk_f32_fp8((int)vx[i][cc], false) * e;
            const f32x2 hi = __builtin_amdgcn_cvt_pk_f32_fp8((int)vx[i][cc], true) * e;
            v[i][0] = lo[0];
            v[i][1] = lo[1];
            v[i][2] = hi[0];
            v[i][3] = hi[1];
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const i16x4 vb = bf16x4(__builtin_amdgcn_perm(__float_as_uint(v[1][k]), __float_as_uint(v[0][k]), 0x07060302u),
                                    __builtin_amdgcn_perm(__float_as_uint(v[3][k]), __float_as_uint(v[2][k]), 0x07060302u));
            o[4 * cc + k] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(pa4, vb, o[4 * cc + k], 0, 0, 0);
        }
    }
}

template <int HALF>
__device__ __forceinline__ void run(const Args& a, const QOps& s_q, int lane, int n, int g, int t, int split, int b,
                                    int k0, int k1, float c_nope, float c_rope);

// CTAs are (token, 16-head group, CTA split) in that order. The Args of group hg see heads 16hg .. 16hg + 15 as
// heads 0 .. 15, so one launch covers all h_q heads.
__device__ __forceinline__ Args head_group(const Args& a, int th, int& t) {
    const int groups = (a.h_q + 15) >> 4;
    t = th / groups;
    const int h0 = 16 * (th - t * groups);
    Args r = a;
    r.q += (size_t)h0 * a.q_h;
    r.po += (size_t)h0 * a.po_h;
    r.plse += (size_t)h0 * a.plse_h;
    r.h_q = min(16, a.h_q - h0);
    return r;
}

extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(2)))
sparse_mla_decode_gfx942(Args args) {
    __shared__ QOps s_q;
    __shared__ float s_c[2][16];  // per head: nope and rope multipliers of the raw MFMA sums (log2 units)
    const int lane = threadIdx.x & 63;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int half = wave & 1;
    const int n = lane & 15, g = lane >> 4;
    const int th = blockIdx.x / args.cta_splits;
    int t;
    const Args a = head_group(args, th, t);
    const int split = (blockIdx.x - th * a.cta_splits) * 2 + (wave >> 1);
    const int b = t / a.s_q;
    const int total = a.topk_main + a.topk_extra;
    const int k0 = split * a.per_split;
    const int k1 = min(k0 + a.per_split, total);
    constexpr float LOG2E = 1.4426950408889634f;

    if (wave == 0) {
        // column n is head n; lane group g holds dims 64c + 16g .. +16 of each chunk (fp8 step j: + 8j .. +8) and
        // rope dims 16g .. +16 (bf16 step s: 16g + 4s .. +4)
        const bool head_ok = n < a.h_q;
        const uint16_t* qh = a.q + (size_t)t * a.q_t + (size_t)min(n, a.h_q - 1) * a.q_h;
        u32x4 qraw[16];
#pragma unroll
        for (int c = 0; c < 7; ++c) {
            qraw[2 * c] = head_ok ? *(const u32x4*)(qh + 64 * c + 16 * g) : u32x4{0, 0, 0, 0};
            qraw[2 * c + 1] = head_ok ? *(const u32x4*)(qh + 64 * c + 16 * g + 8) : u32x4{0, 0, 0, 0};
        }
        qraw[14] = head_ok ? *(const u32x4*)(qh + 448 + 16 * g) : u32x4{0, 0, 0, 0};
        qraw[15] = head_ok ? *(const u32x4*)(qh + 448 + 16 * g + 8) : u32x4{0, 0, 0, 0};
        u16x2 mx = {0, 0};
#pragma unroll
        for (int i = 0; i < 14; ++i)
#pragma unroll
            for (int w = 0; w < 4; ++w)
                mx = __builtin_elementwise_max(mx, __builtin_bit_cast(u16x2, (uint32_t)(qraw[i][w] & 0x7fff7fffu)));
        uint32_t amax = max((uint32_t)mx[0], (uint32_t)mx[1]);
        amax = max(amax, (uint32_t)__shfl_xor((int)amax, 16));
        amax = max(amax, (uint32_t)__shfl_xor((int)amax, 32));
        // |q| * 2^(6 - E) < 128 for the head's largest |q| = m * 2^E: below the e4m3fnuz maximum 240
        const int ef = (int)(amax >> 7);
        const float qs = __uint_as_float((uint32_t)min(max(260 - ef, 1), 254) << 23);   // 2^(133 - ef)
        const float qinv = __uint_as_float((uint32_t)min(max(ef - 6, 1), 254) << 23);   // 2^(ef - 133)
#pragma unroll
        for (int c = 0; c < 7; ++c) {
            uint64_t hi0, lo0, hi1, lo1;
            q_split(qraw[2 * c], qs, hi0, lo0);
            q_split(qraw[2 * c + 1], qs, hi1, lo1);
            s_q.v[2 * c][lane] = u32x4{(uint32_t)hi0, (uint32_t)(hi0 >> 32), (uint32_t)hi1, (uint32_t)(hi1 >> 32)};
            s_q.v[2 * c + 1][lane] = u32x4{(uint32_t)lo0, (uint32_t)(lo0 >> 32), (uint32_t)lo1, (uint32_t)(lo1 >> 32)};
        }
        s_q.v[14][lane] = qraw[14];
        s_q.v[15][lane] = qraw[15];
        if (g == 0) {
            s_c[0][n] = a.sm_scale * LOG2E * (ef > 0 ? qinv : 0.f);
            s_c[1][n] = a.sm_scale * LOG2E;
        }
    }
    __syncthreads();
    const float c_nope = s_c[0][n], c_rope = s_c[1][n];

    if (half == 0)
        run<0>(a, s_q, lane, n, g, t, split, b, k0, k1, c_nope, c_rope);
    else
        run<1>(a, s_q, lane, n, g, t, split, b, k0, k1, c_nope, c_rope);
}

// One wave's token range and half of the output dims (HALF 0: chunks 0-3; HALF 1: chunks 4-6 and the rope dims).
template <int HALF>
__device__ __forceinline__ void run(const Args& a, const QOps& s_q, int lane, int n, int g, int t, int split, int b,
                                    int k0, int k1, float c_nope, float c_rope) {
    const int32_t* idx_main_row = a.idx_main + (size_t)t * a.idx_main_t;
    const int32_t* idx_extra_row = a.idx_extra + (size_t)t * a.idx_extra_t;
    const int len_main = a.len_main[b], len_extra = a.len_extra[b];

    f32x4 o[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i] = f32x4{0.f, 0.f, 0.f, 0.f};
    float m_run = -INFINITY, l_run = 0.f;

    // prologue: the first tile's indices, addresses and QK^T operands
    Tile cur_t = tile_at(a, idx_main_row, idx_extra_row, len_main, len_extra, k0, k1);
    Keys cur;
    u32x4 kr[9];
    u32x2 sc[4];
    {
        const bool live = k0 < k1 && cur_t.len > 0;
        const int ia = live && n < cur_t.len ? cur_t.idx[n] : -1;
        const u32x4 kci = live ? *(const u32x4*)(cur_t.idx + 4 * g) : u32x4{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
        cur = keys_of(cur_t, a.zero, ia, kci, n, g);
        load_qk(cur, g, kr, sc);
    }

    for (int tile = k0; tile < k1; tile += 16) {
        const Tile nxt_t = tile_at(a, idx_main_row, idx_extra_row, len_main, len_extra, tile + 16, k1);
        const bool next_live = tile + 16 < k1 && nxt_t.len > 0;
        int ia_n = -1;
        u32x4 kci_n = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
        if (next_live) {
            ia_n = n < nxt_t.len ? nxt_t.idx[n] : -1;
            kci_n = *(const u32x4*)(nxt_t.idx + 4 * g);
        }
        // PV operands of this half for the current tile: half 0 chunks 0-3, half 1 chunks 4-6 and rope
        uint32_t vx[4][4];
        u32x2 r[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if constexpr (HALF == 0) {
#pragma unroll
                for (int cc = 0; cc < 4; ++cc) vx[i][cc] = *(const uint32_t*)(cur.cdata[i] + 64 * cc + 4 * n);
            } else {
#pragma unroll
                for (int cc = 0; cc < 3; ++cc) vx[i][cc] = *(const uint32_t*)(cur.cdata[i] + 256 + 64 * cc + 4 * n);
                r[i] = *(const u32x2*)(cur.cdata[i] + 448 + 8 * n);
            }
        }

        // ---- QK^T of the current tile; chunk sums alternate between two accumulators and are folded into the
        // scaled nope sum as soon as their four MFMAs are issued
        float nope[4] = {0.f, 0.f, 0.f, 0.f};
        f32x4 acc[2];
#pragma unroll
        for (int c = 0; c < 7; ++c) {
            const u32x4 qh = s_q.v[2 * c][lane], ql = s_q.v[2 * c + 1][lane];
            f32x4& ac = acc[c & 1];
            ac = f32x4{0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const long ka8 = (long)u64(kr[c][2 * j], kr[c][2 * j + 1]);
                ac = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(ka8, (long)u64(qh[2 * j], qh[2 * j + 1]), ac, 0, 0, 0);
                ac = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(ka8, (long)u64(ql[2 * j], ql[2 * j + 1]), ac, 0, 0, 0);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t e = c < 4 ? (sc[i][0] >> (8 * c)) & 0xffu : (sc[i][1] >> (8 * (c - 4))) & 0xffu;
                nope[i] += pow2_e8m0(e) * ac[i];
            }
        }
        f32x4 ar = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const u32x4 qr = s_q.v[14 + p][lane];
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2)
                ar = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(bf16x4(kr[7 + p][2 * h2], kr[7 + p][2 * h2 + 1]),
                                                               bf16x4(qr[2 * h2], qr[2 * h2 + 1]), ar, 0, 0, 0);
        }
        float s[4];
        u32x2 sc_cur[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sc_cur[i] = sc[i];
            s[i] = cur.cvalid[i] ? c_nope * nope[i] + c_rope * ar[i] : -INFINITY;
        }
        // keep the next tile's loads below this point: the current QK^T operands die here
        __builtin_amdgcn_sched_barrier(0);

        // ---- next tile's addresses and QK^T operands, in flight during this tile's softmax and PV
        if (next_live) {
            cur = keys_of(nxt_t, a.zero, ia_n, kci_n, n, g);
            load_qk(cur, g, kr, sc);
        } else {
            // a tile without valid keys: every key reads the zero bytes (the stale score operands are masked)
            cur = keys_of(nxt_t, a.zero, -1, kci_n, n, g);
#pragma unroll
            for (int i = 0; i < 4; ++i) sc[i] = u32x2{0u, 0u};
        }
        cur_t = nxt_t;

        // ---- online softmax over the tile's 16 keys, per head (column n)
        float tmax = fmaxf(fmaxf(s[0], s[1]), fmaxf(s[2], s[3]));
        tmax = fmaxf(tmax, __shfl_xor(tmax, 16));
        tmax = fmaxf(tmax, __shfl_xor(tmax, 32));
        const float m_new = fmaxf(m_run, tmax);
        const bool none = m_new == -INFINITY;
        const float alpha = none ? 1.f : exp2f(m_run - m_new);
        float p[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) p[i] = none ? 0.f : exp2f(s[i] - m_new);
        l_run = l_run * alpha + ((p[0] + p[1]) + (p[2] + p[3]));
        m_run = m_new;
        const i16x4 pa4 = bf16x4(rne_bf16_bits(p[0]) | (rne_bf16_bits(p[1]) << 16),
                                 rne_bf16_bits(p[2]) | (rne_bf16_bits(p[3]) << 16));
        // O rows are heads 4g + i, their alpha sits in lane 4g + i; scaling by 1 changes nothing, so skip it
        if (__any(alpha != 1.f)) {
            float al[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) al[i] = __shfl(alpha, 4 * g + i);
#pragma unroll
            for (int blk = 0; blk < 16; ++blk)
#pragma unroll
                for (int i = 0; i < 4; ++i) o[blk][i] *= al[i];
        }

        // ---- PV for this half: half 0 chunks 0-3 as blocks 4c + k (dim 64c + 4n + k); half 1 chunks 4-6 as blocks
        // 4(c - 4) + k and rope dim 448 + 4n + k as block 12 + k
        if constexpr (HALF == 0) {
            pv_chunks<0, 4>(pa4, vx, sc_cur, o);
        } else {
            pv_chunks<4, 3>(pa4, vx, sc_cur, o);
            // rope dims 448 + 4n + k: dword k / 2 of each key, low half for even k
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t sel = (k & 1) ? 0x07060302u : 0x05040100u;
                const i16x4 vb = bf16x4(__builtin_amdgcn_perm(r[1][k / 2], r[0][k / 2], sel),
                                        __builtin_amdgcn_perm(r[3][k / 2], r[2][k / 2], sel));
                o[12 + k] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(pa4, vb, o[12 + k], 0, 0, 0);
            }
        }
    }

    // ---- partial output: O / l per head (rows 4g + i), lse in natural log, +inf without valid keys
    float l_tot = l_run + __shfl_xor(l_run, 16);
    l_tot += __shfl_xor(l_tot, 32);
    float* po = a.po + (size_t)split * a.po_s + (size_t)t * a.po_t;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int h = 4 * g + i;
        const float lh = __shfl(l_tot, h);
        const float inv = lh > 0.f ? 1.f / lh : 0.f;
        if (h < a.h_q) {
            float* row = po + (size_t)h * a.po_h;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int blk = 4 * c;
                const int dim = HALF == 0 ? 64 * c + 4 * n : (c < 3 ? 64 * (c + 4) + 4 * n : 448 + 4 * n);
                *(f32x4*)(row + dim) = f32x4{o[blk][i] * inv, o[blk + 1][i] * inv, o[blk + 2][i] * inv, o[blk + 3][i] * inv};
            }
        }
    }
    if (HALF == 0 && g == 0 && n < a.h_q) {
        const float lse = l_tot > 0.f ? (m_run + log2f(l_tot)) * 0.6931471805599453f : INFINITY;
        a.plse[(size_t)split * a.plse_s + (size_t)t * a.plse_t + (size_t)n * a.plse_h] = lse;
    }
}

// ---- prefill variant ---------------------------------------------------------------------------------------------
// The two waves of a token range split the QK^T work instead of both computing all of it: half 0 takes the fp8 chunks
// 0-3, half 1 chunks 4-6 and the rope dims (16 MFMAs each, and each wave loads only those key bytes for QK^T). The
// pair exchanges its partial scores through LDS and both form s = partial0 + partial1 in that order, so the halves
// keep identical softmax states. Both ranges of a CTA run the same number of tiles (a shorter range's extra tiles are
// empty) because the exchange needs a CTA barrier per tile.

template <int HALF>
__device__ __forceinline__ void load_qk_half(const Keys& k, int g, u32x4 (&kr)[5], u32x2 (&sc)[4]) {
    if constexpr (HALF == 0) {
#pragma unroll
        for (int c = 0; c < 4; ++c) kr[c] = *(const u32x4*)(k.adata + 64 * c + 16 * g);
    } else {
#pragma unroll
        for (int c = 0; c < 3; ++c) kr[c] = *(const u32x4*)(k.adata + 64 * (c + 4) + 16 * g);
        kr[3] = *(const u32x4*)(k.adata + 448 + 32 * g);
        kr[4] = *(const u32x4*)(k.adata + 448 + 32 * g + 16);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) sc[i] = *(const u32x2*)k.cscale[i];
}

// This half's share of the scaled score of keys 4g + i for head n.
template <int HALF>
__device__ __forceinline__ void qk_half(const QOps& s_q, int lane, const u32x4 (&kr)[5], const u32x2 (&sc)[4],
                                        float c_nope, float c_rope, float (&part)[4]) {
    constexpr int C0 = HALF == 0 ? 0 : 4, NC = HALF == 0 ? 4 : 3;
    float nope[4] = {0.f, 0.f, 0.f, 0.f};
    f32x4 acc[2];
#pragma unroll
    for (int cc = 0; cc < NC; ++cc) {
        const int c = C0 + cc;
        const u32x4 qh = s_q.v[2 * c][lane], ql = s_q.v[2 * c + 1][lane];
        f32x4& ac = acc[cc & 1];
        ac = f32x4{0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const long ka8 = (long)u64(kr[cc][2 * j], kr[cc][2 * j + 1]);
            ac = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(ka8, (long)u64(qh[2 * j], qh[2 * j + 1]), ac, 0, 0, 0);
            ac = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(ka8, (long)u64(ql[2 * j], ql[2 * j + 1]), ac, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t e = c < 4 ? (sc[i][0] >> (8 * c)) & 0xffu : (sc[i][1] >> (8 * (c - 4))) & 0xffu;
            nope[i] += pow2_e8m0(e) * ac[i];
        }
    }
    if constexpr (HALF == 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) part[i] = c_nope * nope[i];
    } else {
        f32x4 ar = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const u32x4 qr = s_q.v[14 + p][lane];
#pragma unroll
            for (int h2 = 0; h2 < 2; ++h2)
                ar = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(bf16x4(kr[3 + p][2 * h2], kr[3 + p][2 * h2 + 1]),
                                                               bf16x4(qr[2 * h2], qr[2 * h2 + 1]), ar, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) part[i] = c_nope * nope[i] + c_rope * ar[i];
    }
}

template <int HALF>
__device__ __forceinline__ void run_split(const Args& a, const QOps& s_q, f32x4 (*s_x)[4][64], int lane, int wave,
                                          int n, int g, int t, int split, int b, int k0, int k1, int n_iter,
                                          float c_nope, float c_rope) {
    const int32_t* idx_main_row = a.idx_main + (size_t)t * a.idx_main_t;
    const int32_t* idx_extra_row = a.idx_extra + (size_t)t * a.idx_extra_t;
    const int len_main = a.len_main[b], len_extra = a.len_extra[b];

    f32x4 o[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i] = f32x4{0.f, 0.f, 0.f, 0.f};
    float m_run = -INFINITY, l_run = 0.f;

    Tile cur_t = tile_at(a, idx_main_row, idx_extra_row, len_main, len_extra, k0, k1);
    Keys cur;
    u32x4 kr[5];
    u32x2 sc[4];
    {
        const bool live = k0 < k1 && cur_t.len > 0;
        const int ia = live && n < cur_t.len ? cur_t.idx[n] : -1;
        const u32x4 kci = live ? *(const u32x4*)(cur_t.idx + 4 * g) : u32x4{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
        cur = keys_of(cur_t, a.zero, ia, kci, n, g);
        load_qk_half<HALF>(cur, g, kr, sc);
    }

    for (int it = 0; it < n_iter; ++it) {
        const int tile = k0 + 16 * it;
        const Tile nxt_t = tile_at(a, idx_main_row, idx_extra_row, len_main, len_extra, tile + 16, k1);
        const bool next_live = tile + 16 < k1 && nxt_t.len > 0;
        int ia_n = -1;
        u32x4 kci_n = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
        if (next_live) {
            ia_n = n < nxt_t.len ? nxt_t.idx[n] : -1;
            kci_n = *(const u32x4*)(nxt_t.idx + 4 * g);
        }
        uint32_t vx[4][4];
        u32x2 r[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if constexpr (HALF == 0) {
#pragma unroll
                for (int cc = 0; cc < 4; ++cc) vx[i][cc] = *(const uint32_t*)(cur.cdata[i] + 64 * cc + 4 * n);
            } else {
#pragma unroll
                for (int cc = 0; cc < 3; ++cc) vx[i][cc] = *(const uint32_t*)(cur.cdata[i] + 256 + 64 * cc + 4 * n);
                r[i] = *(const u32x2*)(cur.cdata[i] + 448 + 8 * n);
            }
        }

        float part[4];
        qk_half<HALF>(s_q, lane, kr, sc, c_nope, c_rope, part);
        u32x2 sc_cur[4];
        bool cv[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sc_cur[i] = sc[i];
            cv[i] = cur.cvalid[i];
        }
        s_x[it & 1][wave][lane] = f32x4{part[0], part[1], part[2], part[3]};
        __builtin_amdgcn_sched_barrier(0);

        if (next_live) {
            cur = keys_of(nxt_t, a.zero, ia_n, kci_n, n, g);
            load_qk_half<HALF>(cur, g, kr, sc);
        } else {
            cur = keys_of(nxt_t, a.zero, -1, kci_n, n, g);
#pragma unroll
            for (int i = 0; i < 4; ++i) sc[i] = u32x2{0u, 0u};
        }
        cur_t = nxt_t;

        __syncthreads();
        const f32x4 other = s_x[it & 1][wave ^ 1][lane];
        float s[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i] = cv[i] ? (HALF == 0 ? part[i] + other[i] : other[i] + part[i]) : -INFINITY;

        float tmax = fmaxf(fmaxf(s[0], s[1]), fmaxf(s[2], s[3]));
        tmax = fmaxf(tmax, __shfl_xor(tmax, 16));
        tmax = fmaxf(tmax, __shfl_xor(tmax, 32));
        const float m_new = fmaxf(m_run, tmax);
        const bool none = m_new == -INFINITY;
        const float alpha = none ? 1.f : exp2f(m_run - m_new);
        float p[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) p[i] = none ? 0.f : exp2f(s[i] - m_new);
        l_run = l_run * alpha + ((p[0] + p[1]) + (p[2] + p[3]));
        m_run = m_new;
        const i16x4 pa4 = bf16x4(rne_bf16_bits(p[0]) | (rne_bf16_bits(p[1]) << 16),
                                 rne_bf16_bits(p[2]) | (rne_bf16_bits(p[3]) << 16));
        if (__any(alpha != 1.f)) {
            float al[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) al[i] = __shfl(alpha, 4 * g + i);
#pragma unroll
            for (int blk = 0; blk < 16; ++blk)
#pragma unroll
                for (int i = 0; i < 4; ++i) o[blk][i] *= al[i];
        }

        if constexpr (HALF == 0) {
            pv_chunks<0, 4>(pa4, vx, sc_cur, o);
        } else {
            pv_chunks<4, 3>(pa4, vx, sc_cur, o);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t sel = (k & 1) ? 0x07060302u : 0x05040100u;
                const i16x4 vb = bf16x4(__builtin_amdgcn_perm(r[1][k / 2], r[0][k / 2], sel),
                                        __builtin_amdgcn_perm(r[3][k / 2], r[2][k / 2], sel));
                o[12 + k] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(pa4, vb, o[12 + k], 0, 0, 0);
            }
        }
    }

    float l_tot = l_run + __shfl_xor(l_run, 16);
    l_tot += __shfl_xor(l_tot, 32);
    float* po = a.po + (size_t)split * a.po_s + (size_t)t * a.po_t;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int h = 4 * g + i;
        const float lh = __shfl(l_tot, h);
        const float inv = lh > 0.f ? 1.f / lh : 0.f;
        if (h < a.h_q) {
            float* row = po + (size_t)h * a.po_h;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int blk = 4 * c;
                const int dim = HALF == 0 ? 64 * c + 4 * n : (c < 3 ? 64 * (c + 4) + 4 * n : 448 + 4 * n);
                *(f32x4*)(row + dim) = f32x4{o[blk][i] * inv, o[blk + 1][i] * inv, o[blk + 2][i] * inv, o[blk + 3][i] * inv};
            }
        }
    }
    if (HALF == 0 && g == 0 && n < a.h_q) {
        const float lse = l_tot > 0.f ? (m_run + log2f(l_tot)) * 0.6931471805599453f : INFINITY;
        a.plse[(size_t)split * a.plse_s + (size_t)t * a.plse_t + (size_t)n * a.plse_h] = lse;
    }
}

extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(2)))
sparse_mla_prefill_gfx942(Args args) {
    __shared__ QOps s_q;
    __shared__ float s_c[2][16];
    __shared__ f32x4 s_x[2][4][64];  // partial scores, by tile parity, wave and lane
    const int lane = threadIdx.x & 63;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
    const int half = wave & 1;
    const int n = lane & 15, g = lane >> 4;
    const int th = blockIdx.x / args.cta_splits;
    int t;
    const Args a = head_group(args, th, t);
    const int split = (blockIdx.x - th * a.cta_splits) * 2 + (wave >> 1);
    const int b = t / a.s_q;
    const int total = a.topk_main + a.topk_extra;
    const int k0 = split * a.per_split;
    const int k1 = min(k0 + a.per_split, total);
    const int n_iter = a.per_split / 16;
    constexpr float LOG2E = 1.4426950408889634f;

    if (wave == 0) {
        const bool head_ok = n < a.h_q;
        const uint16_t* qh = a.q + (size_t)t * a.q_t + (size_t)min(n, a.h_q - 1) * a.q_h;
        u32x4 qraw[16];
#pragma unroll
        for (int c = 0; c < 7; ++c) {
            qraw[2 * c] = head_ok ? *(const u32x4*)(qh + 64 * c + 16 * g) : u32x4{0, 0, 0, 0};
            qraw[2 * c + 1] = head_ok ? *(const u32x4*)(qh + 64 * c + 16 * g + 8) : u32x4{0, 0, 0, 0};
        }
        qraw[14] = head_ok ? *(const u32x4*)(qh + 448 + 16 * g) : u32x4{0, 0, 0, 0};
        qraw[15] = head_ok ? *(const u32x4*)(qh + 448 + 16 * g + 8) : u32x4{0, 0, 0, 0};
        u16x2 mx = {0, 0};
#pragma unroll
        for (int i = 0; i < 14; ++i)
#pragma unroll
            for (int w = 0; w < 4; ++w)
                mx = __builtin_elementwise_max(mx, __builtin_bit_cast(u16x2, (uint32_t)(qraw[i][w] & 0x7fff7fffu)));
        uint32_t amax = max((uint32_t)mx[0], (uint32_t)mx[1]);
        amax = max(amax, (uint32_t)__shfl_xor((int)amax, 16));
        amax = max(amax, (uint32_t)__shfl_xor((int)amax, 32));
        const int ef = (int)(amax >> 7);
        const float qs = __uint_as_float((uint32_t)min(max(260 - ef, 1), 254) << 23);
        const float qinv = __uint_as_float((uint32_t)min(max(ef - 6, 1), 254) << 23);
#pragma unroll
        for (int c = 0; c < 7; ++c) {
            uint64_t hi0, lo0, hi1, lo1;
            q_split(qraw[2 * c], qs, hi0, lo0);
            q_split(qraw[2 * c + 1], qs, hi1, lo1);
            s_q.v[2 * c][lane] = u32x4{(uint32_t)hi0, (uint32_t)(hi0 >> 32), (uint32_t)hi1, (uint32_t)(hi1 >> 32)};
            s_q.v[2 * c + 1][lane] = u32x4{(uint32_t)lo0, (uint32_t)(lo0 >> 32), (uint32_t)lo1, (uint32_t)(lo1 >> 32)};
        }
        s_q.v[14][lane] = qraw[14];
        s_q.v[15][lane] = qraw[15];
        if (g == 0) {
            s_c[0][n] = a.sm_scale * LOG2E * (ef > 0 ? qinv : 0.f);
            s_c[1][n] = a.sm_scale * LOG2E;
        }
    }
    __syncthreads();
    const float c_nope = s_c[0][n], c_rope = s_c[1][n];

    if (half == 0)
        run_split<0>(a, s_q, s_x, lane, wave, n, g, t, split, b, k0, k1, n_iter, c_nope, c_rope);
    else
        run_split<1>(a, s_q, s_x, lane, wave, n, g, t, split, b, k0, k1, n_iter, c_nope, c_rope);
}
