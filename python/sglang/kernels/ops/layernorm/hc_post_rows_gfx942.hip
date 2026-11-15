// gfx942 mHC sublayer boundary, elementwise half, one wave per row: residual_out = hc_post(x, residual, post, comb)
// and y = sum_k pre_prev[k] * copy_k in the Triton _hc_boundary_partial_kernel operation order (separate multiplies
// and adds, round to nearest even: bit-identical), plus the row's sum of squares of the copies. A wave reads 1 KB
// contiguous per instruction; the 64-column slicing of the Triton kernel reads 64 B per row per instruction, which
// runs at a third of the bandwidth on MI308X.
#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int kHc = 4;

struct RowArgs {
  const uint16_t* x;
  const uint16_t* res;
  const float* post_in;
  const float* comb_in;
  const float* pre_prev;
  uint16_t* res_out;
  uint16_t* y;
  float* sq;
  int64_t x_stride;
  int64_t res_stride;
  int32_t m;
  int32_t h;
};

__device__ __forceinline__ float bf16_to_f32(uint32_t b) { return __uint_as_float(b << 16); }

__device__ __forceinline__ uint32_t f32_to_bf16(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (u >> 16) | 0x40u;
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

__device__ __forceinline__ uint32_t half_of(const uint4& v, int e) {
  const uint32_t w = e < 2 ? (e == 0 ? v.x : v.y) : (e == 2 ? v.z : v.w);
  return w;
}

template <bool kPost, bool kComb>
__device__ __forceinline__ void hc_rows(const RowArgs& a) {
  const int row = blockIdx.x * 4 + (threadIdx.x >> 6);
  const int lane = threadIdx.x & 63;
  if (row >= a.m) return;
  float pk[kHc], cm[kHc][kHc], pp[kHc];
  if constexpr (kPost) {
#pragma unroll
    for (int k = 0; k < kHc; ++k) pk[k] = a.post_in[row * kHc + k];
#pragma unroll
    for (int j = 0; j < kHc; ++j)
#pragma unroll
      for (int k = 0; k < kHc; ++k) cm[j][k] = a.comb_in[row * kHc * kHc + j * kHc + k];
  }
  if constexpr (kComb) {
#pragma unroll
    for (int k = 0; k < kHc; ++k) pp[k] = a.pre_prev[row * kHc + k];
  }
  const uint16_t* xrow = kPost ? a.x + static_cast<int64_t>(row) * a.x_stride : nullptr;
  const uint16_t* rrow = a.res + static_cast<int64_t>(row) * a.res_stride;
  uint16_t* orow = kPost ? a.res_out + static_cast<int64_t>(row) * (kHc * a.h) : nullptr;
  uint16_t* yrow = kComb ? a.y + static_cast<int64_t>(row) * a.h : nullptr;
  float sq = 0.f;
  for (int col = 8 * lane; col < a.h; col += 512) {
    uint4 xv, rv[kHc];
    if constexpr (kPost) xv = *reinterpret_cast<const uint4*>(xrow + col);
#pragma unroll
    for (int k = 0; k < kHc; ++k) rv[k] = *reinterpret_cast<const uint4*>(rrow + static_cast<int64_t>(k) * a.h + col);
    uint32_t bits[kHc][8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int sh = (e & 1) * 16;
      float r[kHc];
#pragma unroll
      for (int k = 0; k < kHc; ++k) r[k] = bf16_to_f32((half_of(rv[k], e >> 1) >> sh) & 0xffffu);
      if constexpr (kPost) {
        const float xa = bf16_to_f32((half_of(xv, e >> 1) >> sh) & 0xffffu);
#pragma unroll
        for (int k = 0; k < kHc; ++k) {
          float v = xa * pk[k];
          v = v + r[0] * cm[0][k];
          v = v + r[1] * cm[1][k];
          v = v + r[2] * cm[2][k];
          v = v + r[3] * cm[3][k];
          bits[k][e] = f32_to_bf16(v);
        }
      } else {
#pragma unroll
        for (int k = 0; k < kHc; ++k) bits[k][e] = __float_as_uint(r[k]) >> 16;
      }
    }
    if constexpr (kPost) {
#pragma unroll
      for (int k = 0; k < kHc; ++k)
        *reinterpret_cast<uint4*>(orow + static_cast<int64_t>(k) * a.h + col) =
            make_uint4(bits[k][0] | (bits[k][1] << 16), bits[k][2] | (bits[k][3] << 16),
                       bits[k][4] | (bits[k][5] << 16), bits[k][6] | (bits[k][7] << 16));
    }
    if constexpr (kComb) {
      uint32_t yb[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float yv = 0.f;
#pragma unroll
        for (int k = 0; k < kHc; ++k) yv = yv + pp[k] * bf16_to_f32(bits[k][e]);
        yb[e] = f32_to_bf16(yv);
      }
      *reinterpret_cast<uint4*>(yrow + col) =
          make_uint4(yb[0] | (yb[1] << 16), yb[2] | (yb[3] << 16), yb[4] | (yb[5] << 16), yb[6] | (yb[7] << 16));
    }
#pragma unroll
    for (int k = 0; k < kHc; ++k)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const float xt = bf16_to_f32(bits[k][e]);
        sq = __builtin_fmaf(xt, xt, sq);
      }
  }
#pragma unroll
  for (int off = 32; off > 0; off >>= 1) sq += __shfl_xor(sq, off);
  if (lane == 0) a.sq[row] = sq;
}

#define HC_ROWS_KERNEL(name, post, comb) \
  extern "C" __global__ void __launch_bounds__(256) name(const RowArgs a) { hc_rows<post, comb>(a); }

HC_ROWS_KERNEL(hc_rows_gfx942_post_comb, true, true)
HC_ROWS_KERNEL(hc_rows_gfx942_post, true, false)
HC_ROWS_KERNEL(hc_rows_gfx942_comb, false, true)
HC_ROWS_KERNEL(hc_rows_gfx942_stats, false, false)

// The one-slice mixes of the rows path -> pre / post / comb, one thread per row, in the operation order of the
// Triton _hc_mix_reduce_sinkhorn_row (sigmoid as 1 / (1 + exp(-x)), true divisions, pairwise sums of 4). The
// Triton kernel spends a 4-wave program on each row's 24 values.
struct SinkArgs {
  const float* mix;
  const float* sq;
  const float* scale;
  const float* base;
  float* pre;
  float* post;
  float* comb;
  int32_t m;
  int32_t iters;
  float inv_k;
  float rms_eps;
  float eps;
};

__device__ __forceinline__ float sum4(float a, float b, float c, float d) { return (a + b) + (c + d); }

extern "C" __global__ void __launch_bounds__(256) hc_sinkhorn_rows_gfx942(const SinkArgs a) {
  const int row = blockIdx.x * 256 + threadIdx.x;
  if (row >= a.m) return;
  float v[6 * kHc];
  const float4* mv = reinterpret_cast<const float4*>(a.mix + static_cast<int64_t>(row) * (6 * kHc));
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    const float4 t = mv[i];
    v[4 * i] = t.x;
    v[4 * i + 1] = t.y;
    v[4 * i + 2] = t.z;
    v[4 * i + 3] = t.w;
  }
  // multiply-adds as fma where Triton fuses them (enable_fp_fusion); this file builds with -ffp-contract=off
  const float rs = 1.f / sqrtf(__builtin_fmaf(a.sq[row], a.inv_k, a.rms_eps));
  const float s0 = a.scale[0], s1 = a.scale[1], s2 = a.scale[2];
  float pre[kHc], post[kHc], c[kHc * kHc];
#pragma unroll
  for (int j = 0; j < kHc; ++j) {
    pre[j] = 1.f / (1.f + expf(-__builtin_fmaf(v[j] * rs, s0, a.base[j]))) + a.eps;
    post[j] = 2.f * (1.f / (1.f + expf(-__builtin_fmaf(v[kHc + j] * rs, s1, a.base[kHc + j]))));
  }
#pragma unroll
  for (int i = 0; i < kHc * kHc; ++i) c[i] = __builtin_fmaf(v[2 * kHc + i] * rs, s2, a.base[2 * kHc + i]);
#pragma unroll
  for (int j = 0; j < kHc; ++j) {
    float* r = c + j * kHc;
    const float mx = fmaxf(fmaxf(r[0], r[1]), fmaxf(r[2], r[3]));
#pragma unroll
    for (int k = 0; k < kHc; ++k) r[k] = expf(r[k] - mx);
    const float s = sum4(r[0], r[1], r[2], r[3]);
#pragma unroll
    for (int k = 0; k < kHc; ++k) r[k] = r[k] / s + a.eps;
  }
  for (int it = 0; it < a.iters; ++it) {
    if (it > 0) {
#pragma unroll
      for (int j = 0; j < kHc; ++j) {
        float* r = c + j * kHc;
        const float s = sum4(r[0], r[1], r[2], r[3]) + a.eps;
#pragma unroll
        for (int k = 0; k < kHc; ++k) r[k] = r[k] / s;
      }
    }
#pragma unroll
    for (int k = 0; k < kHc; ++k) {
      const float s = sum4(c[k], c[kHc + k], c[2 * kHc + k], c[3 * kHc + k]) + a.eps;
#pragma unroll
      for (int j = 0; j < kHc; ++j) c[j * kHc + k] = c[j * kHc + k] / s;
    }
  }
  reinterpret_cast<float4*>(a.pre)[row] = make_float4(pre[0], pre[1], pre[2], pre[3]);
  reinterpret_cast<float4*>(a.post)[row] = make_float4(post[0], post[1], post[2], post[3]);
  float4* co = reinterpret_cast<float4*>(a.comb + static_cast<int64_t>(row) * (kHc * kHc));
#pragma unroll
  for (int j = 0; j < kHc; ++j) co[j] = make_float4(c[j * kHc], c[j * kHc + 1], c[j * kHc + 2], c[j * kHc + 3]);
}
