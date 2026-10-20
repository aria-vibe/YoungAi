/* v41_attn_seg_bench_rq.cuh — V11/V12: Q 不住 shared, 每个 warp 把自己 4 个 k 步的 A 片装进寄存器整段复用(2026-10-20)。
 *
 * 为什么: 现役 seg 核(= V7)shared 41.7 KB(qs 16 + ks 16 + spart 8 + 杂 1.7)+ 95 寄存器 ⇒ 每 SM 只挂 2 个 block
 * (板子 shared/SM 100 KB, 寄存器 65536; cuobjdump 现役二进制 REG:95)。09-29 nsys 联合账: 验证批 n=4 时 seg+merge 独占 5.2 ms/步而几乎不读字节,
 * 结构性原因就是这 2 个槽位 —— 每个 block 的 gather 是随机行的全局读(带载 ~1.2 µs 一跳), 同驻 block 少就是延迟盖不住。
 * 去掉 qs 那 16 KB 后 shared 25.7 KB ⇒ 按 shared 能挂 3 个; 寄存器还得压到 ≤80 才真挂 3 个(65536 / 3 / 256 = 85, 按 8 对齐 = 80),
 * 所以 V12 用 __launch_bounds__(256, 3) 钉; V11 不钉(看编译器自然用多少)。
 *
 * ★逐位同的依据★: A 片 = 同一份 bf16 Q 的同一 16×16 子块(先借 ks 那 16 KB 转一次 bf16, 装完片再让 gather 覆盖), 只是整段放寄存器
 * 而不是每个键块从 shared 重载; k 步序、spart 固定序相加、在线 max/sum、P·V 子块序全照 V7 ⇒ mma 输入逐位同 ⇒ 输出逐位同。
 * 只被 v41_attn_seg_bench.cu include(拆出来是守 500 行); 依赖它前面 include 的 _kernels.cuh 与 .cu 里的 gather_v1 / attn_stats_par / attn_pv16。 */
#pragma once

/* 一个 16 键列块的 S 片, A 片来自寄存器(其余与 ds4_attn_mma_scores<16> 逐字: 8 warp 各 4 个 k 步, partial 按 warp 序相加) */
__device__ __forceinline__ static void attn_scores_rq(
        float *stile, float *spart,
        const nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> *qa,
        const __nv_bfloat16 *ks, const int *valid, uint32_t nt, float scale) {
    namespace wmma = nvcuda::wmma;
    const uint32_t warp = threadIdx.x >> 5;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    const uint32_t ksteps = DS4_ATTN_MMA_HD / 16u / DS4_ATTN_MMA_WARPS;
    #pragma unroll
    for (uint32_t s = 0; s < ksteps; s++) {
        wmma::load_matrix_sync(b, ks + (warp * ksteps + s) * 16u, DS4_ATTN_MMA_HD);
        wmma::mma_sync(c, qa[s], b, c);
    }
    wmma::store_matrix_sync(spart + (size_t)warp * 256u, c, 16, wmma::mem_row_major);
    __syncthreads();
    for (uint32_t e = threadIdx.x; e < 256u; e += blockDim.x) {
        float v = 0.f;
        for (uint32_t w = 0; w < DS4_ATTN_MMA_WARPS; w++) v += spart[(size_t)w * 256u + e];
        const uint32_t k = e & 15u, h = e >> 4;
        stile[h * 16u + k] = (k < nt && valid[k]) ? v * scale : -1e30f;
    }
    __syncthreads();
}

/* shared: ks 16 KB + spart 8 KB + stile 1 KB + ptile 0.5 KB + rmax/rsum 128 B + valid 64 B = 25.7 KB */
static inline size_t smem_rq_bytes(void) {
    return (size_t)DS4_ATTN_MMA_KT * DS4_ATTN_MMA_HD * sizeof(__nv_bfloat16) + (size_t)DS4_ATTN_MMA_WARPS * 256u * sizeof(float)
         + 256u * sizeof(float) + 256u * sizeof(__nv_bfloat16) + 2u * DS4_ATTN_MMA_HEADS * sizeof(float) + DS4_ATTN_MMA_KT * sizeof(int);
}

template <int MINB>
__global__ static void __launch_bounds__(256, MINB)
attn_seg_rq_kernel(float *pacc, float *pmax, float *psum, const float *q, const float *kvw, const uint8_t *kvc,
                   const int32_t *idx, uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk, uint32_t n_head,
                   float scale, uint32_t ratio, uint32_t nseg) {
    namespace wmma = nvcuda::wmma;
    extern __shared__ char ds4_attn_mma_smem[];
    __nv_bfloat16 *ks = (__nv_bfloat16 *)ds4_attn_mma_smem;                          /* [16][512]; 开头借来放 bf16 Q(正好 16 头 × 512) */
    float *spart = (float *)(ks + DS4_ATTN_MMA_KT * DS4_ATTN_MMA_HD);
    float *stile = spart + DS4_ATTN_MMA_WARPS * 256u;                                 /* [16][16] */
    __nv_bfloat16 *ptile = (__nv_bfloat16 *)(stile + 256u);                           /* [16][16] */
    float *rmax = (float *)(ptile + 256u), *rsum = rmax + DS4_ATTN_MMA_HEADS;
    int *valid = (int *)(rsum + DS4_ATTN_MMA_HEADS);
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t seg = blockIdx.x, h0 = blockIdx.y * DS4_ATTN_MMA_HEADS, i = blockIdx.z;
    const uint64_t pbase = ((uint64_t)i * nseg + seg) * n_head + h0;
    const uint32_t p = pos0 + i;
    const uint32_t lo = p + 1u > window ? p + 1u - window : 0u;
    const uint32_t nwin = p - lo + 1u, nkeys = nwin + topk;
    const uint32_t seg_keys = v41_attn_seg_keys(p, window, ratio, topk);
    const uint32_t k0 = seg * seg_keys;
    if (k0 >= nkeys) {
        for (uint32_t e = threadIdx.x; e < DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_HD; e += blockDim.x)
            pacc[(pbase + e / DS4_ATTN_MMA_HD) * DS4_ATTN_MMA_HD + e % DS4_ATTN_MMA_HD] = 0.f;
        if (threadIdx.x < DS4_ATTN_MMA_HEADS) { pmax[pbase + threadIdx.x] = -1e30f; psum[pbase + threadIdx.x] = 0.f; }
        return;
    }
    const uint32_t k1 = (k0 + seg_keys) < nkeys ? (k0 + seg_keys) : nkeys;
    for (uint32_t e = threadIdx.x; e < DS4_ATTN_MMA_HEADS * DS4_ATTN_MMA_HD; e += blockDim.x)
        ks[e] = __float2bfloat16(q[((uint64_t)i * n_head + h0 + e / DS4_ATTN_MMA_HD) * DS4_ATTN_MMA_HD + e % DS4_ATTN_MMA_HD]);
    if (threadIdx.x < DS4_ATTN_MMA_HEADS) { rmax[threadIdx.x] = -1e30f; rsum[threadIdx.x] = 0.f; }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qa[DS4_ATTN_MMA_HD / 16u / DS4_ATTN_MMA_WARPS];
    #pragma unroll
    for (uint32_t s = 0; s < DS4_ATTN_MMA_HD / 16u / DS4_ATTN_MMA_WARPS; s++)
        wmma::load_matrix_sync(qa[s], ks + (warp * (DS4_ATTN_MMA_HD / 16u / DS4_ATTN_MMA_WARPS) + s) * 16u, DS4_ATTN_MMA_HD);
    __syncthreads();   /* 片装完, ks 才能让 gather 覆盖 */
    for (uint32_t base = k0; base < k1; base += DS4_ATTN_MMA_KT) {
        const uint32_t nt = (k1 - base) < DS4_ATTN_MMA_KT ? (k1 - base) : DS4_ATTN_MMA_KT;
        __syncthreads();
        gather_v1<DS4_ATTN_MMA_KT, 1>(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk);
        __syncthreads();
        attn_scores_rq(stile, spart, qa, ks, valid, nt, scale);
        attn_stats_par<16u>(stile, rmax, rsum);
    }
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[4];
    for (int j = 0; j < 4; j++) wmma::fill_fragment(oacc[j], 0.0f);
    for (uint32_t base = k0; base < k1; base += DS4_ATTN_MMA_KT) {
        const uint32_t nt = (k1 - base) < DS4_ATTN_MMA_KT ? (k1 - base) : DS4_ATTN_MMA_KT;
        __syncthreads();
        gather_v1<DS4_ATTN_MMA_KT, 1>(ks, valid, kvw, kvc, idx, i, base, nt, nwin, lo, pos0, window, ng, topk);
        __syncthreads();
        attn_scores_rq(stile, spart, qa, ks, valid, nt, scale);
        for (uint32_t e = threadIdx.x; e < 256u; e += blockDim.x)
            ptile[e] = __float2bfloat16(expf(stile[e] - rmax[e >> 4]));
        __syncthreads();
        attn_pv16(oacc, ptile, ks);
        __syncthreads();
    }
    float *otile = (float *)ks;   /* 出口: 借 ks(16 KB = 8 warp × 2 片), 两轮存完 4 片(与 V7 同) */
    for (int r = 0; r < 2; r++) {
        __syncthreads();
        wmma::store_matrix_sync(otile + (size_t)warp * 512u, oacc[2 * r], 16, wmma::mem_row_major);
        wmma::store_matrix_sync(otile + (size_t)warp * 512u + 256u, oacc[2 * r + 1], 16, wmma::mem_row_major);
        __syncthreads();
        for (uint32_t e = threadIdx.x; e < DS4_ATTN_MMA_WARPS * 512u; e += blockDim.x) {
            const uint32_t w = e >> 9, rr = e & 511u, jj = rr >> 8, t = rr & 255u, h = t >> 4, d16 = t & 15u;
            pacc[(pbase + h) * DS4_ATTN_MMA_HD + w * 64u + (uint32_t)(2 * r) * 16u + jj * 16u + d16] = otile[e];
        }
    }
    __syncthreads();
    if (threadIdx.x < DS4_ATTN_MMA_HEADS) { pmax[pbase + threadIdx.x] = rmax[threadIdx.x]; psum[pbase + threadIdx.x] = rsum[threadIdx.x]; }
}
