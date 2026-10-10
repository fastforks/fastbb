// fastbb.normconv: fused convolutions for RAFT-family encoders (BasicEncoder / MultiBasicEncoder / ResidualBlock with
// batch or instance norm), general (any size; CUTLASS 2.x implicit GEMM, sm_80-style mma.sync, fp16 accumulation, compiled
// once -- the problem size is a runtime argument). Structure from fastbb csrc/fused_conv.cu (exp 008).
//
//   y = epi(conv(pre(x), w))       x, res, y: NHWC fp16 (channels-last NCHW), w: KRSC fp16, C % 8 == 0, K % 8 == 0
//   pre (optional, mainloop fusion, CUTLASS example 25): relu(x * a[c] + b[c]) (fp16 a / b; padding stays 0) -- the
//        normalize + ReLU of an instance norm applied while the next conv loads its input (no separate pass)
//   epi: v = acc * scale[k] + shift[k] (fp32, optional = 1 / 0);
//        res_mode 0: v = act(v + res)     res_mode 1: v = relu(res + act(v))   (RAFT ResidualBlock tail, act = relu)
//   stats (optional): per-CTA partial sums of y and y^2 per channel (of the stored fp16 values) -> part[tile_m][2][K];
//        in_stats_final reduces them to the instance-norm scale / shift (batch 1). Deterministic, no atomics in global.
//   in_apply: o = relu(c * s + t) | relu(relu(c * s + t) + r) | relu(relu(c * s + t) + r * rs + rt) | c * s + t
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/default_conv2d_fprop_fusion.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution_fusion.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/device/implicit_gemm_convolution_fusion.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

using Hf = cutlass::half_t;
using NHWC = cutlass::layout::TensorNHWC;

struct EpiArgs {
    const float* scale;   // [K] or null
    const float* shift;   // [K] or null
    const __half* res;    // [M, K] or null
    __half* y;            // [M, ldy]
    float* part;          // [gridM, 2, K] or null (stats)
    int act, res_mode, ldy;
};

__device__ __forceinline__ float act_f(float v, int act) {
    switch (act) {
        case 1: return fmaxf(v, 0.f);
        case 2: return fminf(fmaxf(v, 0.f), 6.f);
        case 3: return tanhf(v);
        case 4: return 1.f / (1.f + __expf(-v));
        case 5: return v > 0.f ? v : 0.01f * v;   // leaky relu 0.01
    }
    return v;
}

template <class OutIt, int TN>
struct Visitor {
    static int const kIterations = OutIt::kIterations;
    static int const kElementsPerAccess = OutIt::kElementsPerAccess;
    static_assert(kElementsPerAccess == 8, "8 channels per access");
    using ElementAccumulator = Hf;
    using AccumulatorFragment = cutlass::Array<Hf, kElementsPerAccess>;
    using ThreadMap = typename OutIt::ThreadMap;
    struct SharedStorage {};
    static int const kFrags = ThreadMap::Iterations::kCount;

    EpiArgs const& a;
    OutIt it;
    int M, N;
    int n0, tile_m;
    float* red;   // shared [2 * TN]
    uint4 pr[kFrags];
    float s1[kFrags][8], s2[kFrags][8];
    int cols[kFrags];

    CUTLASS_DEVICE Visitor(EpiArgs const& args, typename OutIt::Params const& prm, cutlass::MatrixCoord extent,
                           int thread_idx, cutlass::MatrixCoord tb_offset, float* red_, int tile_m_)
        : a(args), it(prm, nullptr, extent, thread_idx, tb_offset), M(extent.row()), N(extent.column()),
          n0(tb_offset.column()), tile_m(tile_m_), red(red_) {
#pragma unroll
        for (int i = 0; i < kFrags; ++i) {
            cols[i] = -1;
#pragma unroll
            for (int k = 0; k < 8; ++k) s1[i][k] = s2[i][k] = 0.f;
        }
    }

    CUTLASS_DEVICE void begin_epilogue() {}
    CUTLASS_DEVICE void begin_step(int) {
        if (!a.res) return;
#pragma unroll
        for (int i = 0; i < kFrags; ++i) {
            cutlass::MatrixCoord co = it.thread_start() + ThreadMap::iteration_offset(i);
            long long p = co.row();
            int n = co.column();
            if (p < M && n < N) pr[i] = __ldg(reinterpret_cast<const uint4*>(a.res + p * N + n));
        }
    }
    CUTLASS_DEVICE void begin_row(int) {}
    CUTLASS_DEVICE void end_row(int) {}
    CUTLASS_DEVICE void end_step(int) { ++it; }
    CUTLASS_DEVICE void end_epilogue() {
        if (!a.part) return;
        // merge fragments that share a column (the usual thread map: fragments = rows of one 8-channel column)
        int c0 = -1;
        bool same = true;
#pragma unroll
        for (int i = 0; i < kFrags; ++i) {
            if (cols[i] < 0) continue;
            if (c0 < 0) c0 = cols[i];
            else if (cols[i] != c0) same = false;
        }
        unsigned lane = threadIdx.x % 32;
        // lanes l, l^8, l^16 (TN = 64: 8 lanes per row) or l^16 (TN = 128) hold the same column: shuffle-reduce first
        constexpr int LPR = TN / 8;   // lanes per output row
        int cc = same ? c0 : -2;
        bool shfl_ok = LPR < 32;
#pragma unroll
        for (int o = LPR; o < 32; o <<= 1) shfl_ok &= __shfl_xor_sync(0xffffffffu, cc, o) == cc;
        shfl_ok = __all_sync(0xffffffffu, shfl_ok);
        if (shfl_ok) {
            float v1[8], v2[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                v1[k] = 0.f; v2[k] = 0.f;
#pragma unroll
                for (int i = 0; i < kFrags; ++i) { v1[k] += s1[i][k]; v2[k] += s2[i][k]; }
#pragma unroll
                for (int o = LPR; o < 32; o <<= 1) {
                    v1[k] += __shfl_xor_sync(0xffffffffu, v1[k], o);
                    v2[k] += __shfl_xor_sync(0xffffffffu, v2[k], o);
                }
            }
            if (lane < LPR && c0 >= 0) {
                int c = c0 - n0;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    atomicAdd(red + c + k, v1[k]);
                    atomicAdd(red + TN + c + k, v2[k]);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < kFrags; ++i) {
                if (cols[i] < 0) continue;
                int c = cols[i] - n0;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    atomicAdd(red + c + k, s1[i][k]);
                    atomicAdd(red + TN + c + k, s2[i][k]);
                }
            }
        }
        __syncthreads();
        for (int j = threadIdx.x; j < TN; j += blockDim.x) {
            int n = n0 + j;
            if (n < N) {
                a.part[(long long)tile_m * 2 * N + n] = red[j];
                a.part[(long long)tile_m * 2 * N + N + n] = red[TN + j];
            }
        }
    }

    CUTLASS_DEVICE void visit(int, int, int, int frag_idx, AccumulatorFragment const& acc) {
        cutlass::MatrixCoord co = it.thread_start() + ThreadMap::iteration_offset(frag_idx);
        long long p = co.row();
        int n = co.column();
        if (p >= M || n >= N) return;
        const __half* av = reinterpret_cast<const __half*>(&acc);
        const __half* rv = reinterpret_cast<const __half*>(&pr[frag_idx]);
        __align__(16) __half o[8];
        __align__(16) float sc[8], sf[8];
        if (a.scale) {
            *reinterpret_cast<float4*>(sc) = __ldg(reinterpret_cast<const float4*>(a.scale + n));
            *reinterpret_cast<float4*>(sc + 4) = __ldg(reinterpret_cast<const float4*>(a.scale + n + 4));
        }
        if (a.shift) {
            *reinterpret_cast<float4*>(sf) = __ldg(reinterpret_cast<const float4*>(a.shift + n));
            *reinterpret_cast<float4*>(sf + 4) = __ldg(reinterpret_cast<const float4*>(a.shift + n + 4));
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __half2float(av[k]);
            if (a.scale) v *= sc[k];
            if (a.shift) v += sf[k];
            if (a.res_mode == 1) {
                v = act_f(v, a.act);
                if (a.res) v += __half2float(rv[k]);
                v = fmaxf(v, 0.f);
            } else {
                if (a.res) v += __half2float(rv[k]);
                v = act_f(v, a.act);
            }
            o[k] = __float2half(v);
        }
        *reinterpret_cast<uint4*>(a.y + p * a.ldy + n) = *reinterpret_cast<const uint4*>(o);
        if (a.part) {
            cols[frag_idx] = n;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                float v = __half2float(o[k]);
                s1[frag_idx][k] += v;
                s2[frag_idx][k] += v * v;
            }
        }
    }
};

template <class TB, class WS, int Stages, int Swz, bool Pre>
struct Fused;

template <class TB, class WS, int Stages, int Swz>
struct Fused<TB, WS, Stages, Swz, false> {
    using EpiOp = cutlass::epilogue::thread::LinearCombination<Hf, 8, Hf, Hf>;  // only shapes the epilogue
    using Base = typename cutlass::conv::kernel::DefaultConv2dFprop<
        Hf, NHWC, Hf, NHWC, Hf, NHWC, Hf, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS,
        cutlass::gemm::GemmShape<16, 8, 16>, EpiOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swz>, Stages,
        cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
    using Mma = typename Base::Mma;
    using OutIt = typename Base::Epilogue::OutputTileIterator;
    using V = Visitor<OutIt, TB::kN>;
    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
        V, typename Base::Epilogue>::Epilogue;
    struct SharedStorage {
        union {
            typename Mma::SharedStorage main_loop;
            typename Epilogue::SharedStorage epilogue;
        } u;
        float red[2 * TB::kN];
    };
    struct Params {
        typename Base::Params base;
        EpiArgs e;
    };
    static constexpr bool kPre = false;
};

template <class TB, class WS, int Stages, int Swz>
struct Fused<TB, WS, Stages, Swz, true> {
    using EpiOp = cutlass::epilogue::thread::LinearCombination<Hf, 8, Hf, Hf>;
    using Base = typename cutlass::conv::kernel::DefaultConv2dFpropFusion<
        Hf, NHWC, Hf, NHWC, Hf, cutlass::layout::RowMajor, Hf, NHWC, Hf, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, TB, WS, cutlass::gemm::GemmShape<16, 8, 16>, EpiOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swz>, Stages, cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
    using Mma = typename Base::Mma;
    using OutIt = typename Base::Epilogue::OutputTileIterator;
    using V = Visitor<OutIt, TB::kN>;
    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
        V, typename Base::Epilogue>::Epilogue;
    struct SharedStorage {
        union {
            typename Mma::SharedStorage main_loop;
            typename Epilogue::SharedStorage epilogue;
        } u;
        float red[2 * TB::kN];
    };
    struct Params {
        typename Base::Params base;
        EpiArgs e;
    };
    static constexpr bool kPre = true;
};

template <class F>
__global__ void __launch_bounds__(F::Base::kThreadCount) nc_conv_kernel(typename F::Params const P) {
    using Mma = typename F::Mma;
    extern __shared__ int smem_[];
    auto& ss = *reinterpret_cast<typename F::SharedStorage*>(smem_);
    auto const& prm = P.base;
    typename F::Base::ThreadblockSwizzle sw;
    cutlass::gemm::GemmCoord tile = sw.get_tile_offset(prm.swizzle_log_tile);
    if (prm.grid_tiled_shape.m() <= tile.m() || prm.grid_tiled_shape.n() <= tile.n()) return;
    int thread_idx = threadIdx.x;
    if (P.e.part)
        for (int j = thread_idx; j < 2 * Mma::Shape::kN; j += blockDim.x) ss.red[j] = 0.f;
    typename Mma::IteratorA iA(prm.iterator_A, prm.problem_size, prm.ptr_A, thread_idx,
                               cutlass::MatrixCoord(tile.m() * Mma::Shape::kM, tile.k() * Mma::Shape::kK));
    typename Mma::IteratorB iB(prm.iterator_B, prm.problem_size, prm.ptr_B, thread_idx,
                               cutlass::MatrixCoord(tile.k() * Mma::Shape::kK, tile.n() * Mma::Shape::kN));
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;
    Mma mma(ss.u.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC acc;
    acc.clear();
    if constexpr (F::kPre) {
        typename Mma::IteratorScaleBias iS(prm.iterator_scale_bias, prm.problem_size, prm.ptr_scale, prm.ptr_bias,
                                           thread_idx, cutlass::MatrixCoord(0, tile.k() * Mma::Shape::kK));
        mma(prm.gemm_k_iterations, acc, iA, iB, iS, acc);
    } else {
        mma(prm.gemm_k_iterations, acc, iA, iB, acc, prm.gemm_k_iterations_per_channel);
    }
    __syncthreads();   // red[] zeroing visible (the mainloop also syncs; cheap)
    cutlass::MatrixCoord tb_off(tile.m() * Mma::Shape::kM, tile.n() * Mma::Shape::kN);
    cutlass::MatrixCoord extent(prm.problem_size.N * prm.problem_size.P * prm.problem_size.Q, prm.problem_size.K);
    typename F::V v(P.e, prm.iterator_D, extent, thread_idx, tb_off, ss.red, tile.m());
    typename F::Epilogue epi(ss.u.epilogue, thread_idx, warp_idx, lane_idx);
    epi(v, acc);
}

template <class F>
static int run(cutlass::conv::Conv2dProblemSize const& p, const void* x, const void* w, const void* ps, const void* pb,
               EpiArgs const& e, int* gridm, cudaStream_t s, bool launch) {
    using Base = typename F::Base;
    typename F::Params P;
    if constexpr (F::kPre) {
        using RM = cutlass::layout::RowMajor;
        typename Base::Arguments args(p, {(Hf*)x, NHWC::packed({p.N, p.H, p.W, p.C})}, {(Hf*)w, NHWC::packed({p.K, p.R, p.S, p.C})},
                                      {(Hf*)ps, RM(p.C)}, {(Hf*)pb, RM(p.C)},
                                      {nullptr, NHWC::packed({p.N, p.P, p.Q, p.K})}, {nullptr, NHWC::packed({p.N, p.P, p.Q, p.K})},
                                      {Hf(1), Hf(0)});
        if (cutlass::conv::device::ImplicitGemmConvolutionFusion<Base>::can_implement(args) != cutlass::Status::kSuccess) return 1;
        P = typename F::Params{typename Base::Params(args, nullptr), e};
    } else {
        typename Base::Arguments args(p, {(Hf*)x, NHWC::packed({p.N, p.H, p.W, p.C})}, {(Hf*)w, NHWC::packed({p.K, p.R, p.S, p.C})},
                                      {nullptr, NHWC::packed({p.N, p.P, p.Q, p.K})}, {nullptr, NHWC::packed({p.N, p.P, p.Q, p.K})},
                                      {Hf(1), Hf(0)});
        if (cutlass::conv::device::ImplicitGemmConvolution<Base>::can_implement(args) != cutlass::Status::kSuccess) return 1;
        P = typename F::Params{typename Base::Params(args, nullptr), e};
    }
    *gridm = P.base.grid_tiled_shape.m();
    if (!launch) return 0;   // query only
    typename Base::ThreadblockSwizzle sw;
    dim3 grid = sw.get_grid_shape(P.base.grid_tiled_shape);
    int smem = int(sizeof(typename F::SharedStorage));
    static bool attr_set = false;
    if (!attr_set) {
        if (cudaFuncSetAttribute(nc_conv_kernel<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
            return 2;
        attr_set = true;
    }
    nc_conv_kernel<F><<<grid, Base::kThreadCount, smem, s>>>(P);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

template <int M, int N, int K>
using GS = cutlass::gemm::GemmShape<M, N, K>;

template <bool Pre>
static int dispatch(int cfg, cutlass::conv::Conv2dProblemSize const& p, const void* x, const void* w, const void* ps,
                    const void* pb, EpiArgs const& e, int* gridm, cudaStream_t s, bool launch) {
    switch (cfg) {
        case 0: return run<Fused<GS<128, 64, 32>, GS<32, 32, 32>, 4, 4, Pre>>(p, x, w, ps, pb, e, gridm, s, launch);
        case 1: return run<Fused<GS<128, 128, 32>, GS<32, 64, 32>, 3, 4, Pre>>(p, x, w, ps, pb, e, gridm, s, launch);
        case 2: return run<Fused<GS<64, 128, 32>, GS<32, 32, 32>, 4, 4, Pre>>(p, x, w, ps, pb, e, gridm, s, launch);
        case 3: return run<Fused<GS<64, 64, 32>, GS<32, 32, 32>, 5, 1, Pre>>(p, x, w, ps, pb, e, gridm, s, launch);
    }
    return -1;
}

static const int SMS = [] { int d = 0, n = 36; cudaGetDevice(&d); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); return n; }();

// fixed rule (no timing; RAFT encoder shapes at Midd-H, bench fastbb_enc_conv_mb.py): K <= 64 -> 128x64 (8 warps);
// K = 96 / 128 -> 128x128 (8 warps); 64x128 / 64x64 when there are too few CTAs
static int pick(long long M, int K) {
    auto tiles = [&](int tm, int tn) { return ((M + tm - 1) / tm) * (long long)((K + tn - 1) / tn); };
    if (K <= 64) return tiles(128, 64) >= 2 * SMS ? 0 : 3;
    if (tiles(128, 128) >= 2 * SMS) return 1;
    if (tiles(64, 128) >= 2 * SMS) return 2;
    return 3;
}

// -> y [N, K, P, Q] channels-last; part (stats): [gridM, 2, K] fp32 (returned via the out list)
std::vector<at::Tensor> nc_conv(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> scale, c10::optional<at::Tensor> shift,
                                c10::optional<at::Tensor> res, c10::optional<at::Tensor> pre_scale,
                                c10::optional<at::Tensor> pre_shift, std::vector<int64_t> padding, std::vector<int64_t> stride,
                                int64_t act, int64_t res_mode, bool stats, int64_t cfg) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf, "fp16 CUDA tensors");
    x = x.contiguous(at::MemoryFormat::ChannelsLast);
    TORCH_CHECK(w.is_contiguous(at::MemoryFormat::ChannelsLast), "weight channels-last");
    if (res) res = res->contiguous(at::MemoryFormat::ChannelsLast);
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int K = w.size(0), R = w.size(2), S = w.size(3);
    TORCH_CHECK(C % 8 == 0 && K % 8 == 0 && w.size(1) == C, "C % 8 == 0, K % 8 == 0");
    bool pre = pre_scale.has_value();
    TORCH_CHECK(!pre || (pre_shift && pre_scale->scalar_type() == at::kHalf && pre_scale->numel() == C && N == 1),
                "pre: fp16 [C] scale / shift, batch 1");
    TORCH_CHECK(!stats || N == 1, "stats: batch 1");
    if (scale) TORCH_CHECK(scale->scalar_type() == at::kFloat && scale->numel() == K && scale->is_contiguous(), "scale fp32 [K]");
    if (shift) TORCH_CHECK(shift->scalar_type() == at::kFloat && shift->numel() == K && shift->is_contiguous(), "shift fp32 [K]");
    int ph = padding[0], pw = padding[1], sh = stride[0], sw = stride[1];
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    auto y = at::empty({N, K, P, Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    if (res) TORCH_CHECK(res->sizes() == y.sizes() && res->scalar_type() == at::kHalf, "res shape");
    cutlass::conv::Conv2dProblemSize p({N, H, W, C}, {K, R, S, C}, {ph, ph, pw, pw}, {sh, sw}, {1, 1}, {N, P, Q, K},
                                       cutlass::conv::Mode::kCrossCorrelation, 1);
    long long M = (long long)N * P * Q;
    if (cfg < 0) cfg = pick(M, K);
    EpiArgs e{scale ? scale->data_ptr<float>() : nullptr, shift ? shift->data_ptr<float>() : nullptr,
              res ? (const __half*)res->data_ptr() : nullptr, (__half*)y.data_ptr(), nullptr, (int)act, (int)res_mode, K};
    const void* ps = pre ? pre_scale->data_ptr() : nullptr;
    const void* pb = pre ? pre_shift->data_ptr() : nullptr;
    at::Tensor part;
    int gridm = 0;
    if (stats) {
        int rc = pre ? dispatch<true>(cfg, p, x.data_ptr(), w.data_ptr(), ps, pb, e, &gridm, nullptr, false)
                     : dispatch<false>(cfg, p, x.data_ptr(), w.data_ptr(), ps, pb, e, &gridm, nullptr, false);
        TORCH_CHECK(rc == 0, "nc_conv query failed rc=", rc);
        part = at::empty({gridm, 2, K}, x.options().dtype(at::kFloat));
        e.part = part.data_ptr<float>();
    }
    auto st = at::cuda::getCurrentCUDAStream();
    int rc = pre ? dispatch<true>(cfg, p, x.data_ptr(), w.data_ptr(), ps, pb, e, &gridm, st, true)
                 : dispatch<false>(cfg, p, x.data_ptr(), w.data_ptr(), ps, pb, e, &gridm, st, true);
    TORCH_CHECK(rc == 0, "nc_conv failed rc=", rc, " cfg=", cfg, " x=", x.sizes(), " w=", w.sizes());
    if (stats) return {y, part};
    return {y};
}

// part [G, 2, K] -> scale = rstd, shift = -mean * rstd (fp32 [2, K]: row 0 rstd, row 1 mean) and fp16 copies [2, K]
__global__ void in_stats_final_k(const float* __restrict__ part, int G, int K, double inv_count, float eps,
                                 float* __restrict__ st32, __half* __restrict__ st16) {
    int k = blockIdx.x;
    double a = 0, b = 0;
    for (int g = threadIdx.x; g < G; g += blockDim.x) {
        a += part[(long long)g * 2 * K + k];
        b += part[(long long)g * 2 * K + K + k];
    }
    __shared__ double sa[256], sb[256];
    sa[threadIdx.x] = a;
    sb[threadIdx.x] = b;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) { sa[threadIdx.x] += sa[threadIdx.x + o]; sb[threadIdx.x] += sb[threadIdx.x + o]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double m = sa[0] * inv_count, var = sb[0] * inv_count - m * m;
        float rstd = (float)(1.0 / sqrt((var > 0 ? var : 0) + eps));
        float sh = (float)(-m) * rstd;
        st32[k] = rstd;
        st32[K + k] = (float)m;
        st16[k] = __float2half(rstd);
        st16[K + k] = __float2half(sh);
    }
}

std::vector<at::Tensor> in_stats_final(at::Tensor part, int64_t count, double eps) {
    TORCH_CHECK(part.dim() == 3 && part.size(1) == 2 && part.scalar_type() == at::kFloat, "part [G, 2, K] fp32");
    int G = part.size(0), K = part.size(2);
    auto st32 = at::empty({2, K}, part.options());
    auto st16 = at::empty({2, K}, part.options().dtype(at::kHalf));
    in_stats_final_k<<<K, 256, 0, at::cuda::getCurrentCUDAStream()>>>(part.data_ptr<float>(), G, K, 1.0 / (double)count,
                                                                    (float)eps, st32.data_ptr<float>(), (__half*)st16.data_ptr());
    return {st32, st16};
}

// mode 0: relu(c*s+t)  1: relu(relu(c*s+t) + r)  2: relu(relu(c*s+t) + r*rs+rt)  3: c*s+t ; st: fp32 [2, C] (rstd; mean)
__global__ void in_apply_k(const __half* __restrict__ c, const float* __restrict__ st, const __half* __restrict__ r,
                           const float* __restrict__ rst, __half* __restrict__ y, long long n8, int C, int mode) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n8) return;
    int ch = (int)((i * 8) % C);
    uint4 v = __ldg(reinterpret_cast<const uint4*>(c) + i);
    uint4 rv;
    if (mode == 1 || mode == 2) rv = __ldg(reinterpret_cast<const uint4*>(r) + i);
    const __half* h = reinterpret_cast<const __half*>(&v);
    const __half* hr = reinterpret_cast<const __half*>(&rv);
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        float u = (__half2float(h[k]) - st[C + ch + k]) * st[ch + k];
        if (mode != 3) u = fmaxf(u, 0.f);
        if (mode == 1) u = fmaxf(u + __half2float(hr[k]), 0.f);
        else if (mode == 2) u = fmaxf(u + (__half2float(hr[k]) - rst[C + ch + k]) * rst[ch + k], 0.f);
        o[k] = __float2half(u);
    }
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

static at::Tensor in_apply_impl(at::Tensor c, at::Tensor st, c10::optional<at::Tensor> r, c10::optional<at::Tensor> rst,
                                int64_t mode, bool inplace) {
    TORCH_CHECK(c.scalar_type() == at::kHalf && c.is_contiguous(at::MemoryFormat::ChannelsLast) && c.size(1) % 8 == 0, "c");
    TORCH_CHECK(st.scalar_type() == at::kFloat && st.numel() == 2 * c.size(1), "st fp32 [2, C]");
    if (mode == 1 || mode == 2) {
        TORCH_CHECK(r && r->sizes() == c.sizes() && r->is_contiguous(at::MemoryFormat::ChannelsLast) &&
                    r->scalar_type() == at::kHalf, "r");
    }
    if (mode == 2) TORCH_CHECK(rst && rst->scalar_type() == at::kFloat && rst->numel() == 2 * c.size(1), "rst");
    auto y = inplace ? c : at::empty_like(c);
    long long n8 = c.numel() / 8;
    in_apply_k<<<(n8 + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)c.data_ptr(), st.data_ptr<float>(), r ? (const __half*)r->data_ptr() : nullptr,
        rst ? rst->data_ptr<float>() : nullptr, (__half*)y.data_ptr(), n8, c.size(1), (int)mode);
    return y;
}
at::Tensor in_apply(at::Tensor c, at::Tensor st, c10::optional<at::Tensor> r, c10::optional<at::Tensor> rst, int64_t mode) {
    return in_apply_impl(c, st, r, rst, mode, false);
}
at::Tensor in_apply_(at::Tensor c, at::Tensor st, c10::optional<at::Tensor> r, c10::optional<at::Tensor> rst, int64_t mode) {
    return in_apply_impl(c, st, r, rst, mode, true);
}

// Restormer GDFN core: depthwise 3x3 (stride 1, zero pad 1) over 2h channels + GELU(erf) gate
//   y[n, p, c] = gelu(dw(x)[c]) * dw(x)[c + h],  x [N, H, W, 2h] NHWC fp16, w fp32 [9][2h], bias fp32 [2h] or null
// one thread = V channels (V = 2 if h is even, else 1) x TW consecutive output pixels of a row (input columns and weights
// loaded once per thread); consecutive threads = consecutive channel groups (coalesced); fp32 math
template <int V, int TW>
__global__ void dwgate_k(const __half* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                         __half* __restrict__ y, int N, int H, int W, int h) {
    int hg = h / V, Wt = (W + TW - 1) / TW;
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * H * Wt * hg;
    if (i >= total) return;
    int c = V * (int)(i % hg);
    long long t = i / hg;
    int x0 = (int)(t % Wt) * TW;
    t /= Wt;
    int yh = (int)(t % H);
    long long n = t / H;
    int C2 = 2 * h;
    float wa[9][V], wb[9][V];
#pragma unroll
    for (int k = 0; k < 9; ++k)
#pragma unroll
        for (int v = 0; v < V; ++v) { wa[k][v] = __ldg(w + k * C2 + c + v); wb[k][v] = __ldg(w + k * C2 + c + h + v); }
    float acc_a[TW][V], acc_b[TW][V];
#pragma unroll
    for (int j = 0; j < TW; ++j)
#pragma unroll
        for (int v = 0; v < V; ++v) {
            acc_a[j][v] = b ? __ldg(b + c + v) : 0.f;
            acc_b[j][v] = b ? __ldg(b + c + h + v) : 0.f;
        }
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        int yy = yh + r - 1;
        if (yy < 0 || yy >= H) continue;
        const __half* row = x + ((n * H + yy) * W) * (long long)C2;
#pragma unroll
        for (int q = 0; q < TW + 2; ++q) {
            int xx = x0 + q - 1;
            if (xx < 0 || xx >= W) continue;
            float va[V], vb[V];
            if (V == 4) {
                uint2 ra = *reinterpret_cast<const uint2*>(row + (long long)xx * C2 + c);
                uint2 rb = *reinterpret_cast<const uint2*>(row + (long long)xx * C2 + c + h);
                const __half* ha = reinterpret_cast<const __half*>(&ra);
                const __half* hb = reinterpret_cast<const __half*>(&rb);
#pragma unroll
                for (int v = 0; v < V; ++v) { va[v] = __half2float(ha[v]); vb[v] = __half2float(hb[v]); }
            } else if (V == 2) {
                float2 fa = __half22float2(*reinterpret_cast<const __half2*>(row + (long long)xx * C2 + c));
                float2 fb = __half22float2(*reinterpret_cast<const __half2*>(row + (long long)xx * C2 + c + h));
                va[0] = fa.x; va[V - 1] = fa.y; vb[0] = fb.x; vb[V - 1] = fb.y;
            } else {
                va[0] = __half2float(row[(long long)xx * C2 + c]);
                vb[0] = __half2float(row[(long long)xx * C2 + c + h]);
            }
#pragma unroll
            for (int j = 0; j < TW; ++j) {
                int s = q - j;   // tap column of output x0 + j
                if (s < 0 || s > 2) continue;
#pragma unroll
                for (int v = 0; v < V; ++v) {
                    acc_a[j][v] += va[v] * wa[r * 3 + s][v];
                    acc_b[j][v] += vb[v] * wb[r * 3 + s][v];
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TW; ++j) {
        if (x0 + j >= W) break;
        long long po = ((n * H + yh) * W + x0 + j) * (long long)h + c;
        float o[V];
#pragma unroll
        for (int v = 0; v < V; ++v) {
            float a = acc_a[j][v];
            o[v] = 0.5f * a * (1.f + erff(a * 0.70710678118654752f)) * acc_b[j][v];
        }
        if (V == 4) {
            __align__(8) __half o4[4];
#pragma unroll
            for (int v = 0; v < V; ++v) o4[v] = __float2half(o[v]);
            *reinterpret_cast<uint2*>(y + po) = *reinterpret_cast<const uint2*>(o4);
        } else if (V == 2) *reinterpret_cast<__half2*>(y + po) = __floats2half2_rn(o[0], o[V - 1]);
        else y[po] = __float2half(o[0]);
    }
}

at::Tensor dwgate(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> b) {
    // x [N, H, W, 2h] contiguous fp16; w fp32 [9, 2h] contiguous; -> y [N, H, W, h]
    TORCH_CHECK(x.dim() == 4 && x.is_contiguous() && x.scalar_type() == at::kHalf && x.size(3) % 2 == 0, "dwgate x");
    int N = x.size(0), H = x.size(1), W = x.size(2), h = x.size(3) / 2;
    TORCH_CHECK(w.scalar_type() == at::kFloat && w.is_contiguous() && w.size(0) == 9 && w.size(1) == 2 * h, "dwgate w");
    if (b) TORCH_CHECK(b->scalar_type() == at::kFloat && b->is_contiguous() && b->numel() == 2 * h, "dwgate b");
    auto y = at::empty({N, H, W, h}, x.options());
    constexpr int TW = 4;
    auto st = at::cuda::getCurrentCUDAStream();
    long long rows = (long long)N * H * ((W + TW - 1) / TW);
    const float* bp = b ? b->data_ptr<float>() : nullptr;
    if (h % 4 == 0 && w.data_ptr<float>() != nullptr) {
        long long total = rows * (h / 4);
        dwgate_k<4, TW><<<(total + 127) / 128, 128, 0, st>>>((const __half*)x.data_ptr(), w.data_ptr<float>(), bp,
                                                            (__half*)y.data_ptr(), N, H, W, h);
    } else if (h % 2 == 0) {
        long long total = rows * (h / 2);
        dwgate_k<2, TW><<<(total + 127) / 128, 128, 0, st>>>((const __half*)x.data_ptr(), w.data_ptr<float>(), bp,
                                                            (__half*)y.data_ptr(), N, H, W, h);
    } else {
        long long total = rows * h;
        dwgate_k<1, TW><<<(total + 127) / 128, 128, 0, st>>>((const __half*)x.data_ptr(), w.data_ptr<float>(), bp,
                                                            (__half*)y.data_ptr(), N, H, W, h);
    }
    return y;
}

// LayerNorm over the last (channel) dim of a contiguous [M, C] fp16 tensor, C % 8 == 0, C <= 2048; fp32 two-pass in registers.
// mode 0: (x - mean) * rstd * w + b (b optional)   mode 1 (Restormer BiasFree): x * rstd * w   (rstd from the variance about the mean)
// G = C / 8 threads per row (one 16-byte vector each, up to 8 per thread for C > 256)
template <int VPT>
__global__ void ln_k(const __half* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b,
                     __half* __restrict__ y, long long M, int C, float eps, int mode, int G) {
    // threads of one row: G (power of two <= 32); rows per block: blockDim.x / G
    int lane_in_row = threadIdx.x % G;
    long long row = (blockIdx.x * (long long)blockDim.x + threadIdx.x) / G;
    bool valid = row < M;
    float v[VPT][8];
    float s = 0.f;
#pragma unroll
    for (int u = 0; u < VPT; ++u) {
        int c = (lane_in_row + u * G) * 8;
        if (valid && c < C) {
            uint4 q = __ldg(reinterpret_cast<const uint4*>(x + row * C + c));
            const __half* hq = reinterpret_cast<const __half*>(&q);
#pragma unroll
            for (int k = 0; k < 8; ++k) { v[u][k] = __half2float(hq[k]); s += v[u][k]; }
        } else {
#pragma unroll
            for (int k = 0; k < 8; ++k) v[u][k] = 0.f;
        }
    }
    for (int o = G / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o, G);
    float mean = s / C, ss = 0.f;
#pragma unroll
    for (int u = 0; u < VPT; ++u) {
        int c = (lane_in_row + u * G) * 8;
        if (c < C)
#pragma unroll
            for (int k = 0; k < 8; ++k) { float d = v[u][k] - mean; ss += d * d; }
    }
    for (int o = G / 2; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o, G);
    float rstd = rsqrtf(ss / C + eps);
    if (!valid) return;
#pragma unroll
    for (int u = 0; u < VPT; ++u) {
        int c = (lane_in_row + u * G) * 8;
        if (c >= C) continue;
        __align__(16) __half o8[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float r = mode == 0 ? (v[u][k] - mean) * rstd * w[c + k] + (b ? b[c + k] : 0.f) : v[u][k] * rstd * w[c + k];
            o8[k] = __float2half(r);
        }
        *reinterpret_cast<uint4*>(y + row * C + c) = *reinterpret_cast<const uint4*>(o8);
    }
}

at::Tensor ln_lastdim(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> b, double eps, int64_t mode) {
    TORCH_CHECK(x.is_contiguous() && x.scalar_type() == at::kHalf, "ln: contiguous fp16");
    int C = x.size(-1);
    TORCH_CHECK(C % 8 == 0 && C <= 2048 && w.scalar_type() == at::kFloat && w.numel() == C, "ln: C % 8, fp32 w");
    long long M = x.numel() / C;
    auto y = at::empty_like(x);
    int G = 1;
    while (G < 32 && G * 8 < C) G <<= 1;          // threads per row
    int vpt = (C + G * 8 - 1) / (G * 8);           // vectors per thread
    int threads = 256;
    long long blocks = (M * G + threads - 1) / threads;
    auto st = at::cuda::getCurrentCUDAStream();
    const float* bp = b ? b->data_ptr<float>() : nullptr;
#define LN_CASE(V) ln_k<V><<<blocks, threads, 0, st>>>((const __half*)x.data_ptr(), w.data_ptr<float>(), bp, (__half*)y.data_ptr(), M, C, (float)eps, (int)mode, G)
    if (vpt == 1) LN_CASE(1); else if (vpt == 2) LN_CASE(2); else if (vpt <= 4) LN_CASE(4); else LN_CASE(8);
#undef LN_CASE
    return y;
}

TORCH_LIBRARY(fastbb_nc, m) {
    m.def("conv(Tensor x, Tensor w, Tensor? scale, Tensor? shift, Tensor? res, Tensor? pre_scale, Tensor? pre_shift, "
          "int[] padding, int[] stride, int act, int res_mode, bool stats, int cfg=-1) -> Tensor[]");
    m.def("in_stats_final(Tensor part, int count, float eps) -> Tensor[]");
    m.def("in_apply(Tensor c, Tensor st, Tensor? r, Tensor? rst, int mode) -> Tensor");
    m.def("in_apply_(Tensor(a!) c, Tensor st, Tensor? r, Tensor? rst, int mode) -> Tensor(a!)");
    m.def("dwgate(Tensor x, Tensor w, Tensor? b) -> Tensor");
    m.def("ln_lastdim(Tensor x, Tensor w, Tensor? b, float eps, int mode) -> Tensor");
}
TORCH_LIBRARY_IMPL(fastbb_nc, CUDA, m) {
    m.impl("conv", nc_conv);
    m.impl("in_stats_final", in_stats_final);
    m.impl("in_apply", in_apply);
    m.impl("in_apply_", in_apply_);
    m.impl("dwgate", dwgate);
    m.impl("ln_lastdim", ln_lastdim);
}
