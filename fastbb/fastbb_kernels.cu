// fastbb: prebuilt CUDA kernels for the shared backbones (DINOv2-style ViTs first).
// Compiled once into a fatbin (SASS sm_80/86/89/90/120 + compute_80 PTX), loaded through the driver API (kernels.py).
// Nothing here depends on the input size.
#include <cuda_fp16.h>
#include <stdint.h>

// ------------------------------------------------------------------------------------------------ helpers
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    int n = valid ? 16 : 0;  // src-size 0 -> zero fill
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// D(f16) = A * B + C(f16)   m16n8k16
__device__ __forceinline__ void mma_f16(uint32_t* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// D(f32) = A * B + C(f32)
__device__ __forceinline__ void mma_f32(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float2 unpack_h2(uint32_t u) { return __half22float2(*reinterpret_cast<__half2*>(&u)); }
__device__ __forceinline__ float fast_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// swizzled [rows][64] fp16 tile: 8 chunks of 16 B per row, chunk index XOR (row & 7) -> conflict-free ldmatrix / cp.async
__device__ __forceinline__ int swz(int row, int chunk) { return row * 64 + ((chunk ^ (row & 7)) << 3); }

// ------------------------------------------------------------------------------------------------ flash attention
// O = softmax(scale * Q K^T) V, head_dim 64, no mask, any N (tails masked), fp16 in / out.
// Q, K, V, O: element (b, n, h, d) at ptr[b * sb + n * sn + h * sh + d] (d contiguous; strides in elements, multiples of 8,
// e.g. straight views into the packed [B, N, 3, H, 64] qkv GEMM output, and O written as [B, N, H*64] for the projection).
// FlashAttention-2 schedule: CTA = 4 warps x (16 * RM) query rows; K/V tiles of 64 keys double-buffered with cp.async.
// QK16 / PV16: fp16-accumulate mma for S = Q K^T / for each 64-key block of P V (the running O, the row max and the row sums
// stay fp32; an fp16 accumulation never spans more than 64 products: head_dim for S, one key block for P V).
template <int RM, bool QK16, bool PV16>
__device__ __forceinline__ void flash_fwd_impl(const __half* __restrict__ q, const __half* __restrict__ k,
                                               const __half* __restrict__ v, __half* __restrict__ o, int N, int H,
                                               long long qsb, int qsn, int qsh, long long ksb, int ksn, int ksh,
                                               long long vsb, int vsn, int vsh, long long osb, int osn, int osh,
                                               float scale_log2) {
    constexpr int BR = 64 * RM, BC = 64;
    __shared__ __align__(128) __half sQ[BR * 64];
    __shared__ __align__(128) __half sK[2][BC * 64];
    __shared__ __align__(128) __half sV[2][BC * 64];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int bh = blockIdx.y, b = bh / H, h = bh - b * H;
    const int q0 = blockIdx.x * BR;
    const __half* qb = q + b * qsb + (long long)h * qsh;
    const __half* kb = k + b * ksb + (long long)h * ksh;
    const __half* vb = v + b * vsb + (long long)h * vsh;

    // Q tile -> smem
#pragma unroll
    for (int i = 0; i < BR * 8 / 128; ++i) {
        int c = tid + i * 128, row = c >> 3, ch = c & 7;
        int n = q0 + row;
        cp_async16(smem_u32(&sQ[swz(row, ch)]), qb + (long long)min(n, N - 1) * qsn + ch * 8, n < N);
    }
    auto load_kv = [&](int buf, int k0) {
#pragma unroll
        for (int i = 0; i < BC * 8 / 128; ++i) {
            int c = tid + i * 128, row = c >> 3, ch = c & 7;
            int n = k0 + row;
            bool ok = n < N;
            int nn = min(n, N - 1);
            cp_async16(smem_u32(&sK[buf][swz(row, ch)]), kb + (long long)nn * ksn + ch * 8, ok);
            cp_async16(smem_u32(&sV[buf][swz(row, ch)]), vb + (long long)nn * vsn + ch * 8, ok);
        }
    };
    load_kv(0, 0);
    cp_async_commit();

    // running state: rows (lane>>2) and (lane>>2)+8 of each of the RM m-tiles of this warp
    float m_r[RM][2], l_r[RM][2];
    float oacc[RM][8][4];
#pragma unroll
    for (int mi = 0; mi < RM; ++mi) {
        m_r[mi][0] = m_r[mi][1] = -INFINITY;
        l_r[mi][0] = l_r[mi][1] = 0.f;
#pragma unroll
        for (int u = 0; u < 8; ++u) oacc[mi][u][0] = oacc[mi][u][1] = oacc[mi][u][2] = oacc[mi][u][3] = 0.f;
    }
    uint32_t qf[RM][4][4];  // Q A-fragments: [m-tile][k-step of 16 dims][4]

    const int nblk = (N + BC - 1) / BC;
    for (int j = 0; j < nblk; ++j) {
        cp_async_wait<0>();
        __syncthreads();
        if (j == 0) {
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    int row = warp * 16 * RM + mi * 16 + (lane & 15);
                    int ch = kk * 2 + (lane >> 4);
                    ldsm_x4(smem_u32(&sQ[swz(row, ch)]), qf[mi][kk][0], qf[mi][kk][1], qf[mi][kk][2], qf[mi][kk][3]);
                }
        }
        if (j + 1 < nblk) load_kv((j + 1) & 1, (j + 1) * BC);
        cp_async_commit();
        const __half* K_ = sK[j & 1];
        const __half* V_ = sV[j & 1];

        // ---- S = Q K^T for 64 keys: 8 n-tiles of 8 keys
        float s[RM][8][4];
        if (QK16) {
            uint32_t sh[RM][8][2];
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int t = 0; t < 8; ++t) sh[mi][t][0] = sh[mi][t][1] = 0u;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                for (int t = 0; t < 8; t += 2) {
                    uint32_t b0, b1, b2, b3;
                    int row = (t + (lane >> 4)) * 8 + (lane & 7);
                    int ch = kk * 2 + ((lane >> 3) & 1);
                    ldsm_x4(smem_u32(&K_[swz(row, ch)]), b0, b1, b2, b3);
#pragma unroll
                    for (int mi = 0; mi < RM; ++mi) {
                        mma_f16(sh[mi][t], qf[mi][kk], b0, b1);
                        mma_f16(sh[mi][t + 1], qf[mi][kk], b2, b3);
                    }
                }
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    float2 a = unpack_h2(sh[mi][t][0]), c = unpack_h2(sh[mi][t][1]);
                    s[mi][t][0] = a.x; s[mi][t][1] = a.y; s[mi][t][2] = c.x; s[mi][t][3] = c.y;
                }
        } else {
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int t = 0; t < 8; ++t) s[mi][t][0] = s[mi][t][1] = s[mi][t][2] = s[mi][t][3] = 0.f;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                for (int t = 0; t < 8; t += 2) {
                    uint32_t b0, b1, b2, b3;
                    int row = (t + (lane >> 4)) * 8 + (lane & 7);
                    int ch = kk * 2 + ((lane >> 3) & 1);
                    ldsm_x4(smem_u32(&K_[swz(row, ch)]), b0, b1, b2, b3);
#pragma unroll
                    for (int mi = 0; mi < RM; ++mi) {
                        mma_f32(s[mi][t], qf[mi][kk], b0, b1);
                        mma_f32(s[mi][t + 1], qf[mi][kk], b2, b3);
                    }
                }
        }
        // ---- tail mask (last block only)
        if ((j + 1) * BC > N) {
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                int key = j * BC + t * 8 + (lane & 3) * 2;
#pragma unroll
                for (int mi = 0; mi < RM; ++mi) {
                    if (key >= N) s[mi][t][0] = s[mi][t][2] = -INFINITY;
                    if (key + 1 >= N) s[mi][t][1] = s[mi][t][3] = -INFINITY;
                }
            }
        }
        // ---- online softmax (base 2), P packed as fp16 A-fragments
        uint32_t pf[RM][4][4];  // [m-tile][k-chunk of 16 keys][4]
#pragma unroll
        for (int mi = 0; mi < RM; ++mi) {
            float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                mx0 = fmaxf(mx0, fmaxf(s[mi][t][0], s[mi][t][1]));
                mx1 = fmaxf(mx1, fmaxf(s[mi][t][2], s[mi][t][3]));
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
            float mn0 = fmaxf(m_r[mi][0], mx0 * scale_log2), mn1 = fmaxf(m_r[mi][1], mx1 * scale_log2);
            float c0 = fast_exp2(m_r[mi][0] - mn0), c1 = fast_exp2(m_r[mi][1] - mn1);
            m_r[mi][0] = mn0;
            m_r[mi][1] = mn1;
            float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                float p0 = fast_exp2(fmaf(s[mi][t][0], scale_log2, -mn0));
                float p1 = fast_exp2(fmaf(s[mi][t][1], scale_log2, -mn0));
                float p2 = fast_exp2(fmaf(s[mi][t][2], scale_log2, -mn1));
                float p3 = fast_exp2(fmaf(s[mi][t][3], scale_log2, -mn1));
                sum0 += p0 + p1;
                sum1 += p2 + p3;
                // n-tile t -> k-chunk t/2: even t = a0 (row) / a1 (row+8), odd t = a2 / a3
                pf[mi][t >> 1][(t & 1) * 2 + 0] = pack_h2(p0, p1);
                pf[mi][t >> 1][(t & 1) * 2 + 1] = pack_h2(p2, p3);
            }
            l_r[mi][0] = l_r[mi][0] * c0 + sum0;
            l_r[mi][1] = l_r[mi][1] * c1 + sum1;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                oacc[mi][u][0] *= c0; oacc[mi][u][1] *= c0;
                oacc[mi][u][2] *= c1; oacc[mi][u][3] *= c1;
            }
        }
        // ---- O += P V
        if (PV16) {
            uint32_t ot[RM][8][2];
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int u = 0; u < 8; ++u) ot[mi][u][0] = ot[mi][u][1] = 0u;
#pragma unroll
            for (int kc = 0; kc < 4; ++kc)
#pragma unroll
                for (int u = 0; u < 8; u += 2) {
                    uint32_t b0, b1, b2, b3;
                    int row = kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    int ch = u + (lane >> 4);
                    ldsm_x4_t(smem_u32(&V_[swz(row, ch)]), b0, b1, b2, b3);
#pragma unroll
                    for (int mi = 0; mi < RM; ++mi) {
                        mma_f16(ot[mi][u], pf[mi][kc], b0, b1);
                        mma_f16(ot[mi][u + 1], pf[mi][kc], b2, b3);
                    }
                }
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    float2 a = unpack_h2(ot[mi][u][0]), c = unpack_h2(ot[mi][u][1]);
                    oacc[mi][u][0] += a.x; oacc[mi][u][1] += a.y; oacc[mi][u][2] += c.x; oacc[mi][u][3] += c.y;
                }
        } else {
#pragma unroll
            for (int kc = 0; kc < 4; ++kc)
#pragma unroll
                for (int u = 0; u < 8; u += 2) {
                    uint32_t b0, b1, b2, b3;
                    int row = kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    int ch = u + (lane >> 4);
                    ldsm_x4_t(smem_u32(&V_[swz(row, ch)]), b0, b1, b2, b3);
#pragma unroll
                    for (int mi = 0; mi < RM; ++mi) {
                        mma_f32(oacc[mi][u], pf[mi][kc], b0, b1);
                        mma_f32(oacc[mi][u + 1], pf[mi][kc], b2, b3);
                    }
                }
        }
    }
    // ---- epilogue: O / l -> fp16 (staged through smem for 16-byte stores)
    __syncthreads();
#pragma unroll
    for (int mi = 0; mi < RM; ++mi) {
        float l0 = l_r[mi][0], l1 = l_r[mi][1];  // per-thread partial row sums -> quad total
        l0 += __shfl_xor_sync(0xffffffff, l0, 1);
        l0 += __shfl_xor_sync(0xffffffff, l0, 2);
        l1 += __shfl_xor_sync(0xffffffff, l1, 1);
        l1 += __shfl_xor_sync(0xffffffff, l1, 2);
        float i0 = 1.f / l0, i1 = 1.f / l1;
        int r0 = warp * 16 * RM + mi * 16 + (lane >> 2);
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            int col = u * 8 + (lane & 3) * 2;
            *reinterpret_cast<uint32_t*>(&sQ[swz(r0, col >> 3) + (col & 7)]) = pack_h2(oacc[mi][u][0] * i0, oacc[mi][u][1] * i0);
            *reinterpret_cast<uint32_t*>(&sQ[swz(r0 + 8, col >> 3) + (col & 7)]) = pack_h2(oacc[mi][u][2] * i1, oacc[mi][u][3] * i1);
        }
    }
    __syncthreads();
    __half* ob = o + b * osb + (long long)h * osh;
#pragma unroll
    for (int i = 0; i < BR * 8 / 128; ++i) {
        int c = tid + i * 128, row = c >> 3, ch = c & 7;
        int n = q0 + row;
        if (n < N) *reinterpret_cast<uint4*>(ob + (long long)n * osn + ch * 8) = *reinterpret_cast<const uint4*>(&sQ[swz(row, ch)]);
    }
}

#define FLASH_ARGS                                                                                                      \
    const __half *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v, __half *__restrict__ o,  \
        int N, int H, long long qsb, int qsn, int qsh, long long ksb, int ksn, int ksh, long long vsb, int vsn, int vsh, \
        long long osb, int osn, int osh, float scale_log2
#define FLASH_CALL q, k, v, o, N, H, qsb, qsn, qsh, ksb, ksn, ksh, vsb, vsn, vsh, osb, osn, osh, scale_log2

extern "C" __global__ void __launch_bounds__(128) flash64_r2_hh(FLASH_ARGS) { flash_fwd_impl<2, true, true>(FLASH_CALL); }
extern "C" __global__ void __launch_bounds__(128) flash64_r2_fh(FLASH_ARGS) { flash_fwd_impl<2, false, true>(FLASH_CALL); }
extern "C" __global__ void __launch_bounds__(128) flash64_r2_ff(FLASH_ARGS) { flash_fwd_impl<2, false, false>(FLASH_CALL); }
extern "C" __global__ void __launch_bounds__(128) flash64_r1_hh(FLASH_ARGS) { flash_fwd_impl<1, true, true>(FLASH_CALL); }
extern "C" __global__ void __launch_bounds__(128) flash64_r1_fh(FLASH_ARGS) { flash_fwd_impl<1, false, true>(FLASH_CALL); }
extern "C" __global__ void __launch_bounds__(128) flash64_r1_ff(FLASH_ARGS) { flash_fwd_impl<1, false, false>(FLASH_CALL); }

// ------------------------------------------------------------------------------------------------ GEMM (F.linear)
// C[M, N] = epilogue(A[M, K] @ B[N, K]^T): A = activations (row stride lda), B = nn.Linear weight (row stride ldb), fp16,
// K % 32 == 0, lda / ldb / ldc % 8 == 0. "Chunked fp16 accumulation": the tensor-core MMAs accumulate in fp16 (2x rate on
// GeForce) over CHUNK k-tiles of 32 only, then the partial sum is added to an fp32 accumulator -> no fp16 sum spans more than
// 32 * CHUNK products (error ~ an fp32-accumulate GEMM, speed ~ fp16 accumulate). CHUNK = 0: plain fp32-accumulate MMAs.
// Epilogue (fp32): + bias[n], act (0 none, 1 GELU erf), then out = res + gamma[n] * v (res, gamma optional; gamma null ->
// res + v), stored fp16. CTA 128 x 128, 8 warps (2 x 4, warp tile 64 x 32), 3-stage cp.async pipeline, swizzled smem.
__device__ __forceinline__ int swz32(int row, int chunk) { return row * 32 + ((chunk ^ ((row >> 1) & 3)) << 3); }

template <int CHUNK, int BN>
__device__ __forceinline__ void gemm_tn_impl(const __half* __restrict__ A, const __half* __restrict__ B, __half* __restrict__ C,
                                             const __half* __restrict__ bias, const __half* __restrict__ res,
                                             const __half* __restrict__ gamma, int M, int N, int K, int lda, int ldb, int ldc,
                                             int ldr, int act) {
    constexpr int BM = 128, BK = 32, S = 3, WN = BN / 4, NT = WN / 8;  // warp tile 64 x WN, NT n-tiles of 8
    extern __shared__ __align__(128) __half smem_dyn[];
    __half(*sA)[BM * BK] = reinterpret_cast<__half(*)[BM * BK]>(smem_dyn);
    __half(*sB)[BN * BK] = reinterpret_cast<__half(*)[BN * BK]>(smem_dyn + S * BM * BK);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2, wn = warp & 3;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int KT = K / BK;

    auto load = [&](int st, int kt) {
        const int k0 = kt * BK;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int c = tid + i * 256, row = c >> 2, ch = c & 3;
            int m = m0 + row;
            cp_async16(smem_u32(&sA[st][swz32(row, ch)]), A + (long long)min(m, M - 1) * lda + k0 + ch * 8, m < M);
        }
#pragma unroll
        for (int i = 0; i < BN / 64; ++i) {
            int c = tid + i * 256, row = c >> 2, ch = c & 3;
            int n = n0 + row;
            cp_async16(smem_u32(&sB[st][swz32(row, ch)]), B + (long long)min(n, N - 1) * ldb + k0 + ch * 8, n < N);
        }
    };
#pragma unroll
    for (int s = 0; s < S - 1; ++s) {
        if (s < KT) load(s, s);
        cp_async_commit();
    }
    float acc[4][NT][4];
    uint32_t tmp[4][NT][2];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
            tmp[i][j][0] = tmp[i][j][1] = 0u;
        }
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<S - 2>();
        __syncthreads();
        if (kt + S - 1 < KT) load((kt + S - 1) % S, kt + S - 1);
        cp_async_commit();
        const __half* a_ = sA[kt % S];
        const __half* b_ = sB[kt % S];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            uint32_t af[4][4], bf[NT][2];
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int row = wm * 64 + mi * 16 + (lane & 15);
                ldsm_x4(smem_u32(&a_[swz32(row, ks * 2 + (lane >> 4))]), af[mi][0], af[mi][1], af[mi][2], af[mi][3]);
            }
#pragma unroll
            for (int t = 0; t < NT; t += 2) {
                int row = wn * WN + (t + (lane >> 4)) * 8 + (lane & 7);
                ldsm_x4(smem_u32(&b_[swz32(row, ks * 2 + ((lane >> 3) & 1))]), bf[t][0], bf[t][1], bf[t + 1][0], bf[t + 1][1]);
            }
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    if (CHUNK == 0) mma_f32(acc[mi][t], af[mi], bf[t][0], bf[t][1]);
                    else mma_f16(tmp[mi][t], af[mi], bf[t][0], bf[t][1]);
                }
        }
        if (CHUNK > 0 && ((kt + 1) % (CHUNK > 0 ? CHUNK : 1) == 0 || kt == KT - 1)) {
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                for (int t = 0; t < NT; ++t) {
                    float2 x = unpack_h2(tmp[mi][t][0]), y = unpack_h2(tmp[mi][t][1]);
                    acc[mi][t][0] += x.x; acc[mi][t][1] += x.y; acc[mi][t][2] += y.x; acc[mi][t][3] += y.y;
                    tmp[mi][t][0] = tmp[mi][t][1] = 0u;
                }
        }
    }
    // epilogue
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        int n = n0 + wn * WN + t * 8 + (lane & 3) * 2;
        if (n >= N) continue;
        float b0 = bias ? __half2float(bias[n]) : 0.f, b1 = bias ? __half2float(bias[n + 1]) : 0.f;
        float g0 = gamma ? __half2float(gamma[n]) : 1.f, g1 = gamma ? __half2float(gamma[n + 1]) : 1.f;
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                int m = m0 + wm * 64 + mi * 16 + (lane >> 2) + hh * 8;
                if (m >= M) continue;
                float v0 = acc[mi][t][hh * 2] + b0, v1 = acc[mi][t][hh * 2 + 1] + b1;
                if (act == 1) {
                    v0 = 0.5f * v0 * (1.f + erff(v0 * 0.70710678118654752f));
                    v1 = 0.5f * v1 * (1.f + erff(v1 * 0.70710678118654752f));
                }
                if (res) {
                    float2 r = __half22float2(*reinterpret_cast<const __half2*>(res + (long long)m * ldr + n));
                    v0 = r.x + g0 * v0;
                    v1 = r.y + g1 * v1;
                }
                *reinterpret_cast<uint32_t*>(C + (long long)m * ldc + n) = pack_h2(v0, v1);
            }
    }
}

#define GEMM_ARGS                                                                                                       \
    const __half *__restrict__ A, const __half *__restrict__ B, __half *__restrict__ C, const __half *__restrict__ bias, \
        const __half *__restrict__ res, const __half *__restrict__ gamma, int M, int N, int K, int lda, int ldb, int ldc,  \
        int ldr, int act
#define GEMM_CALL A, B, C, bias, res, gamma, M, N, K, lda, ldb, ldc, ldr, act
// name: gemm_tn_<acc>_<BN>; dynamic smem = 3 * (128 + BN) * 32 * 2 bytes (24 KB / 36 KB per stage set: 48 / 72 KB)
#define GEMM_KERNELS(BN)                                                                                                   \
    extern "C" __global__ void __launch_bounds__(256) gemm_tn_f32_##BN(GEMM_ARGS) { gemm_tn_impl<0, BN>(GEMM_CALL); }  \
    extern "C" __global__ void __launch_bounds__(256) gemm_tn_c1_##BN(GEMM_ARGS) { gemm_tn_impl<1, BN>(GEMM_CALL); }   \
    extern "C" __global__ void __launch_bounds__(256) gemm_tn_c2_##BN(GEMM_ARGS) { gemm_tn_impl<2, BN>(GEMM_CALL); }   \
    extern "C" __global__ void __launch_bounds__(256) gemm_tn_c4_##BN(GEMM_ARGS) { gemm_tn_impl<4, BN>(GEMM_CALL); }   \
    extern "C" __global__ void __launch_bounds__(256) gemm_tn_h_##BN(GEMM_ARGS) { gemm_tn_impl<1 << 20, BN>(GEMM_CALL); }
GEMM_KERNELS(128)
GEMM_KERNELS(256)

// ------------------------------------------------------------------------------------------------ instance norm NHWC
// x [N, HW, C] (channels-last), C % 8 == 0. Stats in one read: per block, sums of (x - a) and (x - a)^2 around the block's
// first value a (no cancellation) -> (mean, M2); blocks merged in a fixed order with Chan's formula (in_stats_final)
// -> st[N, C, 2] = (mean, rstd). Deterministic, no atomics. Apply: y = (x - mean) * rstd (* w + b) (relu) -> same layout, in place allowed.
#include <cuda_bf16.h>
template <typename T> __device__ __forceinline__ float2 to_f2(uint32_t v);
template <> __device__ __forceinline__ float2 to_f2<__half>(uint32_t v) { return __half22float2(*reinterpret_cast<__half2*>(&v)); }
template <> __device__ __forceinline__ float2 to_f2<__nv_bfloat16>(uint32_t v) { return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&v)); }
template <typename T> __device__ __forceinline__ uint32_t from_f2(float a, float b);
template <> __device__ __forceinline__ uint32_t from_f2<__half>(float a, float b) { __half2 h = __floats2half2_rn(a, b); return *reinterpret_cast<uint32_t*>(&h); }
template <> __device__ __forceinline__ uint32_t from_f2<__nv_bfloat16>(float a, float b) { __nv_bfloat162 h = __floats2bfloat162_rn(a, b); return *reinterpret_cast<uint32_t*>(&h); }

// merge (nb, mb, qb) into (na, ma, qa): Chan et al. (count, mean, sum of squared deviations)
__device__ __forceinline__ void chan_merge(float& na, float& ma, float& qa, float nb, float mb, float qb) {
    const float n = na + nb;
    if (nb == 0.f) return;
    const float d = mb - ma, f = nb / n;
    ma += d * f;
    qa += qb + d * d * na * f;
    na = n;
}

// grid (blocks over HW, N), 256 threads; thread t: channel group g = t % G (G = C/8), row r0 = t / G, rows step 256/G.
// Sums of (x - a) and (x - a)^2 around a = the block's first row (a value of the same channel, so no cancellation), reduced
// in shared memory in a fixed order -> partial [N, gridDim.x, C, 2] = per block (mean, M2); the block's count is
// min(rpb, HW - blockIdx.x * rpb).
template <typename T>
__device__ void in_stats_impl(const T* __restrict__ x, float* __restrict__ partial, int HW, int C, int rows_per_block) {
    extern __shared__ float sh[];  // [256][16] partial sums
    const int G = C >> 3, t = threadIdx.x, n = blockIdx.y;
    const int R = 256 / G;  // rows in flight per iteration (G <= 256)
    const int g = t % G, r0 = t / G;
    const int beg = blockIdx.x * rows_per_block, end = min(HW, beg + rows_per_block);
    if (beg >= end) return;  // empty tail block (in_stats_final skips it)
    const T* xn = x + (size_t)n * HW * C + g * 8;
    float a[8], s[8], q[8];
    {
        uint4 v = *reinterpret_cast<const uint4*>(xn + (size_t)beg * C);
        uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < 4; i++) { float2 f = to_f2<T>(w[i]); a[2 * i] = f.x; a[2 * i + 1] = f.y; }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) s[i] = q[i] = 0.f;
    if (r0 < R) {
        for (int r = beg + r0; r < end; r += R) {
            uint4 v = *reinterpret_cast<const uint4*>(xn + (size_t)r * C);
            uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float2 f = to_f2<T>(w[i]);
                const float dx = f.x - a[2 * i], dy = f.y - a[2 * i + 1];
                s[2 * i] += dx; s[2 * i + 1] += dy;
                q[2 * i] = fmaf(dx, dx, q[2 * i]); q[2 * i + 1] = fmaf(dy, dy, q[2 * i + 1]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) { sh[t * 16 + i] = s[i]; sh[t * 16 + 8 + i] = q[i]; }
    __syncthreads();
    if (t < G) {  // thread g sums the R partials of its group
        float sa[16];
#pragma unroll
        for (int i = 0; i < 16; i++) sa[i] = 0.f;
        for (int r = 0; r < R; r++) {
            const float* p = sh + (r * G + t) * 16;
#pragma unroll
            for (int i = 0; i < 16; i++) sa[i] += p[i];
        }
        const float inv = 1.f / (float)(end - beg);
        float* o = partial + (((size_t)n * gridDim.x + blockIdx.x) * C + t * 8) * 2;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float m = sa[i] * inv;
            o[2 * i] = a[i] + m;
            o[2 * i + 1] = fmaxf(sa[8 + i] - sa[i] * m, 0.f);
        }
    }
}

// one warp per (n, c): lane l merges blocks l, l + 32, ... in order, then a fixed shuffle tree -> st[n, c] = (mean, rstd)
// nc32 / nc16 (optional, batch 1): the same statistics in fastbb.normconv's layout, fp32 [2, C] (rstd; mean) and fp16 [2, C]
// (rstd; -mean * rstd)
extern "C" __global__ void __launch_bounds__(256) in_stats_final(const float* __restrict__ partial, float* __restrict__ st, int N,
                                                                 int nb, int HW, int C, int rpb, float eps,
                                                                 float* __restrict__ nc32, __half* __restrict__ nc16) {
    const int i = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (i >= N * C) return;
    const int n = i / C, c = i % C;
    float na = 0.f, ma = 0.f, qa = 0.f;
    for (int b = lane; b < nb; b += 32) {
        const int cnt = min(rpb, HW - b * rpb);
        if (cnt <= 0) break;
        const float* p = partial + (((size_t)n * nb + b) * C + c) * 2;
        chan_merge(na, ma, qa, (float)cnt, p[0], p[1]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float nb2 = __shfl_down_sync(0xffffffffu, na, o), mb = __shfl_down_sync(0xffffffffu, ma, o),
                    qb = __shfl_down_sync(0xffffffffu, qa, o);
        if (lane < o) chan_merge(na, ma, qa, nb2, mb, qb);
    }
    if (lane == 0) {
        const float rstd = rsqrtf(qa / (float)HW + eps);
        st[2 * i] = ma;
        st[2 * i + 1] = rstd;
        if (nc32) {
            nc32[i] = rstd; nc32[C + i] = ma;
            nc16[i] = __float2half(rstd); nc16[C + i] = __float2half(-ma * rstd);
        }
    }
}

// grid-stride over N*HW*C/8 vectors
template <typename T>
__device__ void in_apply_impl(const T* __restrict__ x, T* __restrict__ y, const float* __restrict__ st, const float* __restrict__ w,
                              const float* __restrict__ b, const T* __restrict__ res, int N, int HW, int C, float eps, int relu) {
    const int G = C >> 3;
    const size_t total = (size_t)N * HW * G;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int g = (int)(i % G);
        const int n = (int)(i / ((size_t)HW * G));
        uint4 v = reinterpret_cast<const uint4*>(x)[i];
        uint32_t u[4] = {v.x, v.y, v.z, v.w};
        uint4 rv = make_uint4(0, 0, 0, 0);
        if (relu == 2) rv = reinterpret_cast<const uint4*>(res)[i];
        const uint32_t ru[4] = {rv.x, rv.y, rv.z, rv.w};
        const float* sp = st + ((size_t)n * C + g * 8) * 2;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            float2 f = to_f2<T>(u[k]);
            float2 fr = to_f2<T>(ru[k]);
            float r[2] = {f.x, f.y}, rr[2] = {fr.x, fr.y};
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int c = 2 * k + j;
                r[j] = (r[j] - sp[2 * c]) * sp[2 * c + 1];
                if (w) r[j] = r[j] * w[g * 8 + c] + b[g * 8 + c];
                if (relu) r[j] = fmaxf(r[j], 0.f);
                if (relu == 2) r[j] = fmaxf(r[j] + rr[j], 0.f);  // RAFT ResidualBlock tail: relu(res + relu(norm(y)))
            }
            u[k] = from_f2<T>(r[0], r[1]);
        }
        reinterpret_cast<uint4*>(y)[i] = make_uint4(u[0], u[1], u[2], u[3]);
    }
}
extern "C" __global__ void __launch_bounds__(256) in_stats_h(const __half* x, float* partial, int HW, int C, int rpb) { in_stats_impl<__half>(x, partial, HW, C, rpb); }
extern "C" __global__ void __launch_bounds__(256) in_stats_b(const __nv_bfloat16* x, float* partial, int HW, int C, int rpb) { in_stats_impl<__nv_bfloat16>(x, partial, HW, C, rpb); }
extern "C" __global__ void __launch_bounds__(256) in_apply_h(const __half* x, __half* y, const float* st, const float* w, const float* b, const __half* res, int N, int HW, int C, float eps, int relu) { in_apply_impl<__half>(x, y, st, w, b, res, N, HW, C, eps, relu); }
extern "C" __global__ void __launch_bounds__(256) in_apply_b(const __nv_bfloat16* x, __nv_bfloat16* y, const float* st, const float* w, const float* b, const __nv_bfloat16* res, int N, int HW, int C, float eps, int relu) { in_apply_impl<__nv_bfloat16>(x, y, st, w, b, res, N, HW, C, eps, relu); }
