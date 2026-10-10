// Vendored from exp 008 clf_fast/csrc/fused_conv.cu @ 555897a (op namespace clf -> fastbb). Canonical home is now fastbb.
// exp 008: size-independent fused convolutions for classification CNNs (CUTLASS 2.x implicit GEMM, sm_80-style mma.sync,
// fp16 accumulation), compiled ONCE into a torch extension -- the problem size is a runtime argument, nothing is compiled or
// tuned per input size. Pattern from exp 005 RAFT-Stereo-fast core/cutlass_conv.cu (EpilogueWithVisitor).
//
//   y = act(conv(x, w) + bias[k] (+ res[n, p, q, k]))        x, res, y: NHWC (= channels-last NCHW) fp16, w: KRSC fp16
//   act: 0 none, 1 relu, 2 relu6, 3 silu, 4 hardswish, 5 gelu(erf), 6 gelu(tanh)
// Requires C % 8 == 0 and K % 8 == 0 (16-byte vectors); dilation 1, groups 1. The caller falls back to torch otherwise.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"

using Hf = cutlass::half_t;
using NHWC = cutlass::layout::TensorNHWC;

struct EpiArgs {
    const __half* bias;   // [K] or null
    const __half* res;    // [M, K] or null
    __half* y;            // [M, ldy] (ldy >= K: output into a channel slice of a wider buffer)
    int act;
    int ldy;
};

__device__ __forceinline__ float act_f(float v, int act) {
    switch (act) {
        case 1: return fmaxf(v, 0.f);
        case 2: return fminf(fmaxf(v, 0.f), 6.f);
        case 3: return v / (1.f + __expf(-v));
        case 4: return v * fminf(fmaxf(v + 3.f, 0.f), 6.f) * (1.f / 6.f);
        case 5: return 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
        case 6: return 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
    return v;
}

template <class OutIt>
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
    uint4 pr[kFrags];

    CUTLASS_DEVICE Visitor(EpiArgs const& args, typename OutIt::Params const& prm, cutlass::MatrixCoord extent,
                           int thread_idx, cutlass::MatrixCoord tb_offset)
        : a(args), it(prm, nullptr, extent, thread_idx, tb_offset), M(extent.row()), N(extent.column()) {}

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
    CUTLASS_DEVICE void end_epilogue() {}

    CUTLASS_DEVICE void visit(int, int, int, int frag_idx, AccumulatorFragment const& acc) {
        cutlass::MatrixCoord co = it.thread_start() + ThreadMap::iteration_offset(frag_idx);
        long long p = co.row();
        int n = co.column();
        if (p >= M || n >= N) return;
        const __half* av = reinterpret_cast<const __half*>(&acc);
        __align__(16) __half bv[8];
        if (a.bias) *reinterpret_cast<uint4*>(bv) = __ldg(reinterpret_cast<const uint4*>(a.bias + n));
        const __half* rv = reinterpret_cast<const __half*>(&pr[frag_idx]);
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __half2float(av[k]);
            if (a.bias) v += __half2float(bv[k]);
            if (a.res) v += __half2float(rv[k]);
            o[k] = __float2half(act_f(v, a.act));
        }
        *reinterpret_cast<uint4*>(a.y + p * a.ldy + n) = *reinterpret_cast<const uint4*>(o);
    }
};

template <class TB, class WS, int Stages, int Swz,
          cutlass::conv::IteratorAlgorithm Alg = cutlass::conv::IteratorAlgorithm::kOptimized>
struct Fused {
    using EpiOp = cutlass::epilogue::thread::LinearCombination<Hf, 8, Hf, Hf>;  // only shapes the epilogue
    using Base = typename cutlass::conv::kernel::DefaultConv2dFprop<
        Hf, NHWC, Hf, NHWC, Hf, NHWC, Hf, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB, WS,
        cutlass::gemm::GemmShape<16, 8, 16>, EpiOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swz>, Stages,
        cutlass::arch::OpMultiplyAdd, Alg>::Kernel;
    using Mma = typename Base::Mma;
    using OutIt = typename Base::Epilogue::OutputTileIterator;
    using V = Visitor<OutIt>;
    using Epilogue = typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<
        V, typename Base::Epilogue>::Epilogue;
    union SharedStorage {
        typename Mma::SharedStorage main_loop;
        typename Epilogue::SharedStorage epilogue;
    };
    struct Params {
        typename Base::Params base;
        EpiArgs e;
    };
};

template <class F>
__global__ void __launch_bounds__(F::Base::kThreadCount) fused_conv_kernel(typename F::Params const P) {
    using Mma = typename F::Mma;
    extern __shared__ int smem_[];
    auto& ss = *reinterpret_cast<typename F::SharedStorage*>(smem_);
    auto const& prm = P.base;
    typename F::Base::ThreadblockSwizzle sw;
    cutlass::gemm::GemmCoord tile = sw.get_tile_offset(prm.swizzle_log_tile);
    if (prm.grid_tiled_shape.m() <= tile.m() || prm.grid_tiled_shape.n() <= tile.n()) return;
    int thread_idx = threadIdx.x;
    typename Mma::IteratorA iA(prm.iterator_A, prm.problem_size, prm.ptr_A, thread_idx,
                               cutlass::MatrixCoord(tile.m() * Mma::Shape::kM, tile.k() * Mma::Shape::kK));
    typename Mma::IteratorB iB(prm.iterator_B, prm.problem_size, prm.ptr_B, thread_idx,
                               cutlass::MatrixCoord(tile.k() * Mma::Shape::kK, tile.n() * Mma::Shape::kN));
    int warp_idx = cutlass::canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;
    Mma mma(ss.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC acc;
    acc.clear();
    mma(prm.gemm_k_iterations, acc, iA, iB, acc, prm.gemm_k_iterations_per_channel);
    cutlass::MatrixCoord tb_off(tile.m() * Mma::Shape::kM, tile.n() * Mma::Shape::kN);
    cutlass::MatrixCoord extent(prm.problem_size.N * prm.problem_size.P * prm.problem_size.Q, prm.problem_size.K);
    typename F::V v(P.e, prm.iterator_D, extent, thread_idx, tb_off);
    typename F::Epilogue epi(ss.epilogue, thread_idx, warp_idx, lane_idx);
    epi(v, acc);
}

template <class F>
static int run(cutlass::conv::Conv2dProblemSize const& p, const void* x, const void* w, EpiArgs const& e, cudaStream_t s) {
    using Base = typename F::Base;
    typename Base::Arguments args(p, {(Hf*)x, NHWC::packed({p.N, p.H, p.W, p.C})}, {(Hf*)w, NHWC::packed({p.K, p.R, p.S, p.C})},
                                  {nullptr, NHWC::packed({p.N, p.P, p.Q, p.K})}, {nullptr, NHWC::packed({p.N, p.P, p.Q, p.K})},
                                  {Hf(1), Hf(0)});
    if (cutlass::conv::device::ImplicitGemmConvolution<Base>::can_implement(args) != cutlass::Status::kSuccess) return 1;
    typename F::Params P{typename Base::Params(args, nullptr), e};
    typename Base::ThreadblockSwizzle sw;
    dim3 grid = sw.get_grid_shape(P.base.grid_tiled_shape);
    int smem = int(sizeof(typename F::SharedStorage));
    static bool attr_set = false;
    if (!attr_set) {
        if (cudaFuncSetAttribute(fused_conv_kernel<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
            return 2;
        attr_set = true;
    }
    fused_conv_kernel<F><<<grid, Base::kThreadCount, smem, s>>>(P);
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}

template <int M, int N, int K>
using GS = cutlass::gemm::GemmShape<M, N, K>;

static int dispatch(int cfg, cutlass::conv::Conv2dProblemSize const& p, const void* x, const void* w, EpiArgs const& e,
                    cudaStream_t s) {
    switch (cfg) {
        case 0: return run<Fused<GS<128, 128, 32>, GS<64, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 1: return run<Fused<GS<128, 64, 32>, GS<64, 32, 32>, 4, 4>>(p, x, w, e, s);
        case 2: return run<Fused<GS<64, 64, 32>, GS<32, 32, 32>, 5, 1>>(p, x, w, e, s);
        case 3: return run<Fused<GS<64, 128, 32>, GS<32, 64, 32>, 4, 4>>(p, x, w, e, s);
        case 4: return run<Fused<GS<128, 128, 32>, GS<32, 64, 32>, 4, 4>>(p, x, w, e, s);
        case 5: return run<Fused<GS<128, 256, 32>, GS<64, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 6: return run<Fused<GS<256, 128, 32>, GS<64, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 7: return run<Fused<GS<64, 128, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
        case 8: return run<Fused<GS<128, 128, 32>, GS<32, 64, 32>, 3, 4>>(p, x, w, e, s);
        case 9: return run<Fused<GS<128, 64, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
        // stem (C == 8, zero-padded RGB): fixed-channels iterator, no wasted K
        case 10: return run<Fused<GS<128, 64, 32>, GS<32, 32, 32>, 4, 4, cutlass::conv::IteratorAlgorithm::kFixedChannels>>(p, x, w, e, s);
        case 11: return run<Fused<GS<128, 128, 32>, GS<32, 64, 32>, 3, 4, cutlass::conv::IteratorAlgorithm::kFixedChannels>>(p, x, w, e, s);
        case 12: return run<Fused<GS<64, 64, 32>, GS<32, 32, 32>, 4, 1, cutlass::conv::IteratorAlgorithm::kFixedChannels>>(p, x, w, e, s);
        // narrow-K (output channel) tiles for mobile nets (K = 16 .. 160)
        case 13: return run<Fused<GS<128, 32, 32>, GS<32, 32, 32>, 4, 4>>(p, x, w, e, s);
        case 14: return run<Fused<GS<64, 64, 32>, GS<32, 32, 32>, 3, 4>>(p, x, w, e, s);
        case 15: return run<Fused<GS<256, 32, 32>, GS<64, 32, 32>, 3, 4>>(p, x, w, e, s);
        case 16: return run<Fused<GS<128, 32, 64>, GS<32, 32, 64>, 3, 4>>(p, x, w, e, s);
    }
    return -1;
}

static const int SMS = [] { int d = 0, n = 36; cudaGetDevice(&d); cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); return n; }();

// fixed rule (no timing; from the ResNet-50 sweep, bench/conv_cfg.py): 128x128 tiles with 8 warps when they fill the GPU,
// 128x64 / 8 warps for K = 64 (mod 128), smaller tiles when there are too few CTAs (batch 1)
// fixed rule (no timing; from sweeps of ResNet-50 / MobileNet V2 / EfficientNet-B0 convs at batch 1 / 8 / 64,
// bench/conv_cfg.py): 128x128 tiles (8 warps) when K fits 128-wide tiles and they fill the GPU, else 128x64 (8 warps),
// 64x64 when there are too few CTAs (batch 1).
// cuBLAS fallback for tiny-M 1x1 GEMMs: disabled by default (measured slower than the 64x64 CUTLASS tile at batch 1:
// the extra epilogue launches cost more than split-K saves); CLF_FALLBACK_MACS=<n> enables it above n MACs (A/B only)
static const long long FALLBACK_MACS = [] { const char* v = getenv("CLF_FALLBACK_MACS"); return v ? atoll(v) : (1ll << 62); }();

static int pick(long long M, int K, int C, int R, int S) {
    auto tiles = [&](int tm, int tn) { return ((M + tm - 1) / tm) * (long long)((K + tn - 1) / tn); };
    auto fits = [&](int tn) { return ((K + tn - 1) / tn) * tn <= 1.1 * K; };   // <= 10 % padded output channels
    if (C == 8) return tiles(128, 64) >= SMS ? 10 : 12;   // 128x128 fixed-channel tiles (cfg 11) measured slower everywhere
    if (fits(128)) {
        if (tiles(128, 128) >= SMS) return 8;
        if (tiles(64, 128) >= SMS) return 7;
        return 2;
    }
    if (tiles(128, 64) >= SMS) return 9;     // also for K not a multiple of 64 (mobile nets): ~= 64x64 3-stage, >> 5-stage
    return 2;
}

static at::Tensor stem_c8(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, int ph, int pw, int sh, int sw, int act);

static at::Tensor conv_impl(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, c10::optional<at::Tensor> res,
                            std::vector<int64_t> padding, std::vector<int64_t> stride, int64_t act, int64_t cfg,
                            c10::optional<at::Tensor> out) {
    // x [N, C, H, W] channels-last fp16, w [K, C, R, S] channels-last fp16 (= KRSC), res [N, K, P, Q] channels-last;
    // out (optional): [N, K, P, Q] channel slice of a channels-last buffer (pixel stride ldy, 16-byte aligned)
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && w.scalar_type() == at::kHalf, "fp16 CUDA tensors");
    x = x.contiguous(at::MemoryFormat::ChannelsLast);   // no-op when already channels-last (shuffles / splits may not be)
    TORCH_CHECK(w.is_contiguous(at::MemoryFormat::ChannelsLast), "weight channels-last");

    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int K = w.size(0), R = w.size(2), S = w.size(3);
    int ph = padding[0], pw = padding[1], sh = stride[0], sw = stride[1];
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    at::Tensor y;
    int ldy = K;
    if (out) {
        y = *out;
        ldy = y.stride(3);
        TORCH_CHECK(y.size(0) == N && y.size(1) == K && y.size(2) == P && y.size(3) == Q && y.stride(1) == 1 &&
                    y.stride(2) == (long long)Q * ldy && y.stride(0) == (long long)P * Q * ldy && ldy % 8 == 0 &&
                    ((uintptr_t)y.data_ptr()) % 16 == 0 && y.scalar_type() == at::kHalf, "conv out slice layout");
    } else {
        y = at::empty({N, K, P, Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    }
    if (C == 8 && K <= 16 && !res && !out && cfg < 0)   // direct kernel wins only for K <= 16 (smem bank conflicts above)
        return stem_c8(x, w, bias, ph, pw, sh, sw, (int)act);
    cutlass::conv::Conv2dProblemSize p({N, H, W, C}, {K, R, S, C}, {ph, ph, pw, pw}, {sh, sw}, {1, 1}, {N, P, Q, K},
                                       cutlass::conv::Mode::kCrossCorrelation, 1);
    if (res) {
        TORCH_CHECK(res->device() == x.device() && res->scalar_type() == at::kHalf, "res must match input device and dtype");
        res = res->expand(y.sizes()).contiguous(at::MemoryFormat::ChannelsLast);
    }
    EpiArgs e{bias ? (const __half*)bias->data_ptr() : nullptr,
              res ? (const __half*)res->data_ptr() : nullptr, (__half*)y.data_ptr(), (int)act, ldy};
    if (res) TORCH_CHECK(res->sizes() == y.sizes(), "res shape");
    if (cfg < 0) {
        long long M = (long long)N * P * Q;
        // 1x1 / stride 1 GEMMs with too few CTAs even for 64x64 tiles (batch 1, late stages): cuBLAS (split-K) + epilogue ops
        // (only when the GEMM is big enough for split-K to matter: otherwise the extra epilogue launches cost more)
        if (R == 1 && S == 1 && sh == 1 && sw == 1 && ph == 0 && pw == 0 && ((M + 63) / 64) * ((K + 63) / 64) < SMS &&
            M * (long long)K * C >= FALLBACK_MACS) {
            auto x2 = x.permute({0, 2, 3, 1}).reshape({M, C});
            auto w2 = w.view({K, C});
            auto o = bias ? at::addmm(*bias, x2, w2.t()) : at::mm(x2, w2.t());
            if (res) o.add_(res->permute({0, 2, 3, 1}).reshape({M, K}));
            switch (act) {
                case 1: o.relu_(); break;
                case 2: o.clamp_(0, 6); break;
                case 3: o = at::silu(o); break;
                case 4: o = at::hardswish(o); break;
                case 5: o = at::gelu(o); break;
                case 6: o = at::gelu(o, "tanh"); break;
            }
            auto o4 = o.view({N, P, Q, K}).permute({0, 3, 1, 2});
            if (out) { y.copy_(o4); return y; }
            return o4;
        }
        cfg = pick(M, K, C, R, S);
    }
    int rc = dispatch(cfg, p, x.data_ptr(), w.data_ptr(), e, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(rc == 0, "conv_fused failed rc=", rc, " cfg=", cfg, " x=", x.sizes(), " w=", w.sizes());
    return y;
}

at::Tensor conv_fused(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, c10::optional<at::Tensor> res,
                      std::vector<int64_t> padding, std::vector<int64_t> stride, int64_t act, int64_t cfg) {
    return conv_impl(x, w, bias, res, padding, stride, act, cfg, c10::nullopt);
}

void conv_fused_into(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, c10::optional<at::Tensor> res,
                     std::vector<int64_t> padding, std::vector<int64_t> stride, int64_t act, at::Tensor out) {
    conv_impl(x, w, bias, res, padding, stride, act, -1, out);
}

// y = act(x * scale[c] + shift[c]) for an NHWC fp16 tensor that may be a channel slice (pixel stride ldx); packed output.
// (pre-activation BatchNorm + ReLU of DenseNet / unfoldable BNs)
__global__ void affine_act_k(const __half* __restrict__ x, const float* __restrict__ sc, const float* __restrict__ sh,
                             __half* __restrict__ y, long long npix, int C8, long long ldx, int act) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= npix * C8) return;
    int c = i % C8; long long p = i / C8;
    uint4 v = __ldg(reinterpret_cast<const uint4*>(x + p * ldx) + c);
    const __half* h = reinterpret_cast<const __half*>(&v);
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2half(act_f(__half2float(h[k]) * sc[c * 8 + k] + sh[c * 8 + k], act));
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

at::Tensor affine_act(at::Tensor x, at::Tensor scale, at::Tensor shift, int64_t act) {
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    long long ldx = x.stride(3);
    bool slice_ok = x.stride(1) == 1 && x.stride(2) == W * ldx && x.stride(0) == (long long)H * W * ldx && ldx % 8 == 0 &&
                    ((uintptr_t)x.data_ptr()) % 16 == 0;
    if (!slice_ok) { x = x.contiguous(at::MemoryFormat::ChannelsLast); ldx = C; }
    TORCH_CHECK(x.scalar_type() == at::kHalf && C % 8 == 0 && scale.scalar_type() == at::kFloat, "affine_act input");
    auto y = at::empty({N, C, H, W}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    long long npix = (long long)N * H * W, total = npix * (C / 8);
    affine_act_k<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)x.data_ptr(), scale.data_ptr<float>(), shift.data_ptr<float>(), (__half*)y.data_ptr(), npix, C / 8, ldx, (int)act);
    return y;
}

// max pool, NHWC fp16, 8 channels per thread (C % 8 == 0), -inf padding (PyTorch semantics, no ceil_mode / dilation)
__global__ void maxpool_nhwc8(const __half* __restrict__ x, __half* __restrict__ y, int N, int H, int W, int C8, int P, int Q,
                              int R, int S, int sh, int sw, int ph, int pw) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * P * Q * C8;
    if (i >= total) return;
    int c = i % C8; long long t = i / C8; int q = t % Q; t /= Q; int p = t % P; int n = t / P;
    __half2 m[4];
    for (int k = 0; k < 4; ++k) m[k] = __half2(__ushort_as_half(0xfc00), __ushort_as_half(0xfc00));
    for (int r = 0; r < R; ++r) {
        int h = p * sh - ph + r; if (h < 0 || h >= H) continue;
        for (int s2 = 0; s2 < S; ++s2) {
            int w = q * sw - pw + s2; if (w < 0 || w >= W) continue;
            uint4 v = __ldg(reinterpret_cast<const uint4*>(x + (((long long)n * H + h) * W + w) * C8 * 8) + c);
            const __half2* hv = reinterpret_cast<const __half2*>(&v);
            for (int k = 0; k < 4; ++k) m[k] = __hmax2(m[k], hv[k]);
        }
    }
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(m);
}

at::Tensor maxpool(at::Tensor x, std::vector<int64_t> k, std::vector<int64_t> stride, std::vector<int64_t> pad, bool ceil) {
    if (x.scalar_type() != at::kHalf || x.size(1) % 8) return at::max_pool2d(x, k, stride, pad, {1, 1}, ceil);
    x = x.contiguous(at::MemoryFormat::ChannelsLast);
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    // PyTorch output size (ceil_mode: last window must start inside the input or left padding)
    auto osz = [&](int L, int kk, int s, int p) {
        int o = ceil ? (L + 2 * p - kk + s - 1) / s + 1 : (L + 2 * p - kk) / s + 1;
        if (ceil && (o - 1) * s >= L + p) --o;
        return o;
    };
    int P = osz(H, k[0], stride[0], pad[0]), Q = osz(W, k[1], stride[1], pad[1]);
    auto y = at::empty({N, C, P, Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    long long total = (long long)N * P * Q * (C / 8);
    maxpool_nhwc8<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)x.data_ptr(), (__half*)y.data_ptr(), N, H, W, C / 8, P, Q, k[0], k[1], stride[0], stride[1], pad[0], pad[1]);
    return y;
}

// depthwise conv (groups == C == K), NHWC fp16, weight [R, S, C] fp16, + bias + act; 8 channels of one output pixel per
// thread, fp32 accumulation (memory-bound: accumulation precision is free here)
__global__ void dwconv_nhwc8(const __half* __restrict__ x, const __half* __restrict__ w, const __half* __restrict__ b,
                             __half* __restrict__ y, int N, int H, int W, int C8, int P, int Q, int R, int S, int sh, int sw,
                             int ph, int pw, int act) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * P * Q * C8;
    if (i >= total) return;
    int c = i % C8; long long t = i / C8; int q = t % Q; t /= Q; int p = t % P; int n = t / P;
    float acc[8];
    if (b) {
        uint4 bv = __ldg(reinterpret_cast<const uint4*>(b) + c);
        const __half* hb = reinterpret_cast<const __half*>(&bv);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] = __half2float(hb[k]);
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[k] = 0.f;
    }
    for (int r = 0; r < R; ++r) {
        int h = p * sh - ph + r; if (h < 0 || h >= H) continue;
        for (int s2 = 0; s2 < S; ++s2) {
            int ww = q * sw - pw + s2; if (ww < 0 || ww >= W) continue;
            uint4 v = __ldg(reinterpret_cast<const uint4*>(x + (((long long)n * H + h) * W + ww) * C8 * 8) + c);
            uint4 g = __ldg(reinterpret_cast<const uint4*>(w + ((long long)r * S + s2) * C8 * 8) + c);
            const __half2* hv = reinterpret_cast<const __half2*>(&v);
            const __half2* hg = reinterpret_cast<const __half2*>(&g);
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 a = __half22float2(hv[k]), bb = __half22float2(hg[k]);
                acc[2 * k] += a.x * bb.x; acc[2 * k + 1] += a.y * bb.y;
            }
        }
    }
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2half(act_f(acc[k], act));
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

// register-tiled variant: one thread = 8 channels x TW consecutive output columns; weights and the input row segment are
// loaded once per thread and reused across the TW outputs (fp32 accumulation). R, S <= 7.
template <int TW>
__global__ void dwconv_tw(const __half* __restrict__ x, const __half* __restrict__ w, const __half* __restrict__ b,
                          __half* __restrict__ y, int N, int H, int W, int C8, int P, int Q, int R, int S, int sh, int sw,
                          int ph, int pw, int act) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    int QT = (Q + TW - 1) / TW;
    long long total = (long long)N * P * QT * C8;
    if (i >= total) return;
    int c = i % C8; long long t = i / C8; int qt = t % QT; t /= QT; int p = t % P; int n = t / P;
    int q0 = qt * TW;
    float acc[TW][8];
    float bb[8];
    if (b) {
        uint4 bv = __ldg(reinterpret_cast<const uint4*>(b) + c);
        const __half* hb = reinterpret_cast<const __half*>(&bv);
#pragma unroll
        for (int k = 0; k < 8; ++k) bb[k] = __half2float(hb[k]);
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) bb[k] = 0.f;
    }
#pragma unroll
    for (int j = 0; j < TW; ++j)
#pragma unroll
        for (int k = 0; k < 8; ++k) acc[j][k] = bb[k];
    const int L = (TW - 1) * sw + S;          // input columns touched by the TW outputs
    for (int r = 0; r < R; ++r) {
        int h = p * sh - ph + r; if (h < 0 || h >= H) continue;
        const __half* xrow = x + ((long long)n * H + h) * W * C8 * 8;
        for (int l = 0; l < L; ++l) {
            int ww = q0 * sw - pw + l; if (ww < 0 || ww >= W) continue;
            uint4 v = __ldg(reinterpret_cast<const uint4*>(xrow + (long long)ww * C8 * 8) + c);
            float xv[8];
            const __half2* hv = reinterpret_cast<const __half2*>(&v);
#pragma unroll
            for (int k = 0; k < 4; ++k) { float2 f = __half22float2(hv[k]); xv[2 * k] = f.x; xv[2 * k + 1] = f.y; }
#pragma unroll
            for (int j = 0; j < TW; ++j) {
                int s2 = l - j * sw;                       // tap index of input column l for output j
                if (s2 < 0 || s2 >= S) continue;
                uint4 g = __ldg(reinterpret_cast<const uint4*>(w + ((long long)r * S + s2) * C8 * 8) + c);
                const __half2* hg = reinterpret_cast<const __half2*>(&g);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float2 f = __half22float2(hg[k]);
                    acc[j][2 * k] += xv[2 * k] * f.x; acc[j][2 * k + 1] += xv[2 * k + 1] * f.y;
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < TW; ++j) {
        int q = q0 + j; if (q >= Q) break;
        __align__(16) __half o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] = __float2half(act_f(acc[j][k], act));
        reinterpret_cast<uint4*>(y + (((long long)n * P + p) * Q + q) * C8 * 8)[c] = *reinterpret_cast<uint4*>(o);
    }
}

static const int DW_TW = [] { const char* v = getenv("CLF_DW_TW"); return v ? atoi(v) : 2; }();   // TW 2: 7x7 -17 %, 3x3 ~0 (memory bound)

at::Tensor dwconv(at::Tensor x, at::Tensor w_rsc, c10::optional<at::Tensor> bias, std::vector<int64_t> stride,
                  std::vector<int64_t> padding, int64_t act) {
    x = x.contiguous(at::MemoryFormat::ChannelsLast);
    TORCH_CHECK(x.scalar_type() == at::kHalf && x.size(1) % 8 == 0 && w_rsc.is_contiguous(), "dwconv input");
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3), R = w_rsc.size(0), S = w_rsc.size(1);
    int P = (H + 2 * padding[0] - R) / stride[0] + 1, Q = (W + 2 * padding[1] - S) / stride[1] + 1;
    auto y = at::empty({N, C, P, Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    auto st = at::cuda::getCurrentCUDAStream();
    const __half* xp = (const __half*)x.data_ptr(); const __half* wp = (const __half*)w_rsc.data_ptr();
    const __half* bp = bias ? (const __half*)bias->data_ptr() : nullptr; __half* yp = (__half*)y.data_ptr();
    if (DW_TW == 2 || DW_TW == 4) {
        long long total = (long long)N * P * ((Q + DW_TW - 1) / DW_TW) * (C / 8);
        if (DW_TW == 4)
            dwconv_tw<4><<<(total + 127) / 128, 128, 0, st>>>(xp, wp, bp, yp, N, H, W, C / 8, P, Q, R, S, stride[0], stride[1], padding[0], padding[1], (int)act);
        else
            dwconv_tw<2><<<(total + 127) / 128, 128, 0, st>>>(xp, wp, bp, yp, N, H, W, C / 8, P, Q, R, S, stride[0], stride[1], padding[0], padding[1], (int)act);
        return y;
    }
    long long total = (long long)N * P * Q * (C / 8);
    dwconv_nhwc8<<<(total + 255) / 256, 256, 0, st>>>(xp, wp, bp, yp, N, H, W, C / 8, P, Q, R, S, stride[0], stride[1],
                                                       padding[0], padding[1], (int)act);
    return y;
}

// direct conv for the zero-padded RGB stem (C == 8) with few output channels (K <= 64): one thread = 8 output channels of
// one pixel; weights [K/8][R][S][8 in][8 out] in shared memory (implicit GEMM wastes most of a 64-wide tile at K = 16 / 32)
__global__ void stem_c8_k(const __half* __restrict__ x, const __half* __restrict__ wt, const __half* __restrict__ b,
                          __half* __restrict__ y, int N, int H, int W, int K8, int P, int Q, int R, int S, int sh, int sw,
                          int ph, int pw, int act) {
    extern __shared__ __half ws[];
    int nw = K8 * R * S * 64;
    for (int i = threadIdx.x; i < nw; i += blockDim.x) ws[i] = wt[i];
    __syncthreads();
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * P * Q * K8;
    if (i >= total) return;
    int kg = i % K8; long long t = i / K8; int q = t % Q; t /= Q; int p = t % P; int n = t / P;
    float acc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = b ? __half2float(b[kg * 8 + k]) : 0.f;
    for (int r = 0; r < R; ++r) {
        int h = p * sh - ph + r; if (h < 0 || h >= H) continue;
        for (int s2 = 0; s2 < S; ++s2) {
            int ww = q * sw - pw + s2; if (ww < 0 || ww >= W) continue;
            uint4 v = __ldg(reinterpret_cast<const uint4*>(x + (((long long)n * H + h) * W + ww) * 8));
            const __half* hv = reinterpret_cast<const __half*>(&v);
            const __half* wp = ws + ((kg * R + r) * S + s2) * 64;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                float xv = __half2float(hv[c]);
#pragma unroll
                for (int k = 0; k < 8; ++k) acc[k] += xv * __half2float(wp[c * 8 + k]);
            }
        }
    }
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2half(act_f(acc[k], act));
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

static at::Tensor stem_c8(at::Tensor x, at::Tensor w, c10::optional<at::Tensor> bias, int ph, int pw, int sh, int sw, int act) {
    int N = x.size(0), H = x.size(2), W = x.size(3), K = w.size(0), R = w.size(2), S = w.size(3);
    int P = (H + 2 * ph - R) / sh + 1, Q = (W + 2 * pw - S) / sw + 1;
    // [K, 8, R, S] -> [K/8][R][S][8 in][8 out]
    auto wt = w.contiguous().view({K / 8, 8, 8, R, S}).permute({0, 3, 4, 2, 1}).contiguous();
    auto y = at::empty({N, K, P, Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    long long total = (long long)N * P * Q * (K / 8);
    int smem = K * R * S * 8 * 2;
    stem_c8_k<<<(total + 127) / 128, 128, smem, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)x.data_ptr(), (const __half*)wt.data_ptr(), bias ? (const __half*)bias->data_ptr() : nullptr,
        (__half*)y.data_ptr(), N, H, W, K / 8, P, Q, R, S, sh, sw, ph, pw, act);
    return y;
}

// avg pool NHWC fp16 (count_include_pad as PyTorch), 8 channels per thread
__global__ void avgpool_nhwc8(const __half* __restrict__ x, __half* __restrict__ y, int N, int H, int W, int C8, int P, int Q,
                              int R, int S, int sh, int sw, int ph, int pw, bool cip) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)N * P * Q * C8;
    if (i >= total) return;
    int c = i % C8; long long t = i / C8; int q = t % Q; t /= Q; int p = t % P; int n = t / P;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int cnt = 0;
    for (int r = 0; r < R; ++r) {
        int h = p * sh - ph + r; if (h < 0 || h >= H) continue;
        for (int s2 = 0; s2 < S; ++s2) {
            int ww = q * sw - pw + s2; if (ww < 0 || ww >= W) continue;
            uint4 v = __ldg(reinterpret_cast<const uint4*>(x + (((long long)n * H + h) * W + ww) * C8 * 8) + c);
            const __half* hv = reinterpret_cast<const __half*>(&v);
#pragma unroll
            for (int k = 0; k < 8; ++k) acc[k] += __half2float(hv[k]);
            ++cnt;
        }
    }
    int h0 = p * sh - ph, w0 = q * sw - pw;
    int hend = min(h0 + R, H + ph), wend = min(w0 + S, W + pw);
    float div = cip ? (float)((hend - h0) * (wend - w0)) : (float)max(cnt, 1);
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2half(acc[k] / div);
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

at::Tensor avgpool(at::Tensor x, std::vector<int64_t> k, std::vector<int64_t> stride, std::vector<int64_t> pad, bool cip) {
    if (x.scalar_type() != at::kHalf || x.size(1) % 8) return at::avg_pool2d(x, k, stride, pad, false, cip);
    x = x.contiguous(at::MemoryFormat::ChannelsLast);
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int P = (H + 2 * pad[0] - k[0]) / stride[0] + 1, Q = (W + 2 * pad[1] - k[1]) / stride[1] + 1;
    auto y = at::empty({N, C, P, Q}, x.options().memory_format(at::MemoryFormat::ChannelsLast));
    long long total = (long long)N * P * Q * (C / 8);
    avgpool_nhwc8<<<(total + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const __half*)x.data_ptr(), (__half*)y.data_ptr(), N, H, W, C / 8, P, Q, k[0], k[1], stride[0], stride[1], pad[0], pad[1], cip);
    return y;
}

// network input: fp32 / fp16 NCHW (any strides via contiguous) -> fp16 NHWC with channels zero-padded to 8 (stem conv C 3 -> 8)
template <class T>
__global__ void input_nhwc8_k(const T* __restrict__ x, __half* __restrict__ y, int C, long long HW, long long total) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;   // one output pixel (8 channels) per thread
    if (i >= total) return;
    long long n = i / HW, hw = i % HW;
    __align__(16) __half o[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) o[c] = c < C ? __float2half((float)x[(n * C + c) * HW + hw]) : __float2half(0.f);
    reinterpret_cast<uint4*>(y)[i] = *reinterpret_cast<uint4*>(o);
}

at::Tensor input_nhwc8(at::Tensor x) {
    x = x.contiguous();
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    TORCH_CHECK(C <= 8, "input_nhwc8: C <= 8");
    auto y = at::empty({N, 8, H, W}, x.options().dtype(at::kHalf).memory_format(at::MemoryFormat::ChannelsLast));
    long long total = (long long)N * H * W;
    auto st = at::cuda::getCurrentCUDAStream();
    if (x.scalar_type() == at::kFloat)
        input_nhwc8_k<float><<<(total + 255) / 256, 256, 0, st>>>(x.data_ptr<float>(), (__half*)y.data_ptr(), C, (long long)H * W, total);
    else
        input_nhwc8_k<__half><<<(total + 255) / 256, 256, 0, st>>>((const __half*)x.data_ptr(), (__half*)y.data_ptr(), C, (long long)H * W, total);
    return y;
}

TORCH_LIBRARY(fastbb, m) {
    m.def("input_nhwc8(Tensor x) -> Tensor");
    m.def("conv_fused_into(Tensor x, Tensor w, Tensor? bias, Tensor? res, int[] padding, int[] stride, int act, Tensor(a!) out) -> ()");
    m.def("affine_act(Tensor x, Tensor scale, Tensor shift, int act) -> Tensor");
    m.def("avgpool(Tensor x, int[] k, int[] stride, int[] pad, bool count_include_pad=True) -> Tensor");
    m.def("dwconv(Tensor x, Tensor w_rsc, Tensor? bias, int[] stride, int[] padding, int act) -> Tensor");
    m.def("maxpool(Tensor x, int[] k, int[] stride, int[] pad, bool ceil=False) -> Tensor");
    m.def("conv_fused(Tensor x, Tensor w, Tensor? bias, Tensor? res, int[] padding, int[] stride, int act, int cfg=-1) -> Tensor");
}
TORCH_LIBRARY_IMPL(fastbb, CUDA, m) { m.impl("conv_fused", conv_fused); m.impl("maxpool", maxpool); m.impl("input_nhwc8", input_nhwc8); m.impl("dwconv", dwconv); m.impl("conv_fused_into", conv_fused_into); m.impl("affine_act", affine_act); m.impl("avgpool", avgpool); }

