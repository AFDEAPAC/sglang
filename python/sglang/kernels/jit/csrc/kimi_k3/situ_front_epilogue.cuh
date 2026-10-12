// K3_FRONT_EPILOGUE: shared-expert SiTU + dense routed-latent copy in one launch.
// Reads the row-strided fused MoE front [N, W] = [gate | up | router | routed_latent] and
// writes shared_out [N, D] = situ(gate, up) and routed_out [N, R] = routed_latent.
// The SiTU math is kimi_k3::situ_activate from situ_and_mul.cuh, so the result is
// bit-identical to situ_and_mul_kernel built with the same flags.

#pragma once

#include "situ_and_mul.cuh"

namespace sglang {

struct SituFrontEpilogueParams {
  const void* __restrict__ input;
  void* __restrict__ shared_out;
  void* __restrict__ routed_out;
  float beta;
  float inv_beta;
  float linear_beta;
  float inv_linear_beta;
  uint32_t shared_vecs;     // D / vec
  uint32_t routed_vecs;     // R / vec
  uint32_t routed_col_vecs; // routed-latent column offset / vec
  uint32_t num_tokens;
  uint32_t stride_in_vecs;  // input row stride / vec
};

template <typename T, bool kHasLinearBeta, bool kUsePDL>
__global__ void situ_front_epilogue_kernel(const __grid_constant__ SituFrontEpilogueParams params) {
  using namespace device;
  constexpr auto kVecSize = kMaxVecBytes / sizeof(T);
  using vec_t = AlignedVector<T, kVecSize>;

  const uint32_t per_token = params.shared_vecs + params.routed_vecs;
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto token_id = tid / per_token;
  if (token_id >= params.num_tokens) return;
  const auto item = tid % per_token;
  const uint64_t row = static_cast<uint64_t>(token_id) * params.stride_in_vecs;

  PDLWaitPrimary<kUsePDL>();

  if (item < params.shared_vecs) {
    const auto gate = load_as<vec_t>(params.input, row + item);
    const auto up = load_as<vec_t>(params.input, row + item + params.shared_vecs);
    PDLTriggerSecondary<kUsePDL>();
    vec_t out;
#pragma unroll
    for (int i = 0; i < kVecSize; ++i) {
      const float g = cast<fp32_t>(gate[i]);
      const float u = cast<fp32_t>(up[i]);
      out[i] = cast<T>(kimi_k3::situ_activate<kHasLinearBeta>(
          g, u, params.beta, params.inv_beta, params.linear_beta, params.inv_linear_beta));
    }
    store_as<vec_t>(params.shared_out, out, static_cast<uint64_t>(token_id) * params.shared_vecs + item);
  } else {
    const auto r = item - params.shared_vecs;
    const auto v = load_as<vec_t>(params.input, row + params.routed_col_vecs + r);
    PDLTriggerSecondary<kUsePDL>();
    store_as<vec_t>(params.routed_out, v, static_cast<uint64_t>(token_id) * params.routed_vecs + r);
  }
}

template <typename T, bool kUsePDL>
struct SituFrontEpilogueKernel {
  static constexpr auto kVecSize = device::kMaxVecBytes / sizeof(T);
  static constexpr auto kBlockSize = 256u;

  static void
  run(const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView shared_out,
      const tvm::ffi::TensorView routed_out,
      const int64_t routed_col,
      const double beta,
      const double linear_beta,
      const bool has_linear_beta) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto W = SymbolicSize{"input_width"};
    auto D = SymbolicSize{"shared_width"};
    auto R = SymbolicSize{"routed_width"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({N, D}).with_dtype<T>().with_device(device_).verify(shared_out);
    TensorMatcher({N, R}).with_dtype<T>().with_device(device_).verify(routed_out);
    TensorMatcher({N, W}).with_dtype<T>().with_device(device_).with_strides({-1, 1}).verify(input);

    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto shared_width = static_cast<uint32_t>(D.unwrap());
    const auto routed_width = static_cast<uint32_t>(R.unwrap());
    const auto input_width = static_cast<int64_t>(W.unwrap());
    const auto device = device_.unwrap();
    if (num_tokens == 0) return;

    RuntimeCheck(shared_width % kVecSize == 0, "shared width must be divisible by vector size");
    RuntimeCheck(routed_width % kVecSize == 0, "routed width must be divisible by vector size");
    RuntimeCheck(routed_col % kVecSize == 0, "routed column must be divisible by vector size");
    RuntimeCheck(input.stride(0) % kVecSize == 0, "input row stride must be divisible by vector size");
    RuntimeCheck(2 * static_cast<int64_t>(shared_width) <= routed_col, "routed column overlaps gate_up");
    RuntimeCheck(routed_col + routed_width <= input_width, "routed slice exceeds the input row");

    const auto shared_vecs = shared_width / static_cast<uint32_t>(kVecSize);
    const auto routed_vecs = routed_width / static_cast<uint32_t>(kVecSize);
    const uint64_t total = static_cast<uint64_t>(num_tokens) * (shared_vecs + routed_vecs);
    RuntimeCheck(total <= std::numeric_limits<uint32_t>::max(), "too many items for 32-bit indexing");
    const auto num_blocks = div_ceil(static_cast<uint32_t>(total), kBlockSize);

    const float beta_f = static_cast<float>(beta);
    const float linear_beta_f = static_cast<float>(linear_beta);
    const auto params = SituFrontEpilogueParams{
        .input = input.data_ptr(),
        .shared_out = shared_out.data_ptr(),
        .routed_out = routed_out.data_ptr(),
        .beta = beta_f,
        .inv_beta = 1.0f / beta_f,
        .linear_beta = linear_beta_f,
        .inv_linear_beta = linear_beta_f != 0.0f ? 1.0f / linear_beta_f : 0.0f,
        .shared_vecs = shared_vecs,
        .routed_vecs = routed_vecs,
        .routed_col_vecs = static_cast<uint32_t>(routed_col / static_cast<int64_t>(kVecSize)),
        .num_tokens = num_tokens,
        .stride_in_vecs = static_cast<uint32_t>(input.stride(0) / static_cast<int64_t>(kVecSize)),
    };

    if (has_linear_beta) {
      LaunchKernel(num_blocks, kBlockSize, device)
          .enable_pdl(kUsePDL)(situ_front_epilogue_kernel<T, true, kUsePDL>, params);
    } else {
      LaunchKernel(num_blocks, kBlockSize, device)
          .enable_pdl(kUsePDL)(situ_front_epilogue_kernel<T, false, kUsePDL>, params);
    }
  }
};

}  // namespace sglang
