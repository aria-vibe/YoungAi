/* cuda_amp_route.inc.cu — 带路由的低秩放大器(2026-10-29, 反修 B 段): y += ((x·B) ⊙ u)·A, u[t][k] = 1 + Σ_j rw[t][j]·S[sel[t][j]][k]。
 * 为什么: 段 B 原来的 y += x·B·A 只看这一位的输入, 不知道这一位走了哪几个专家; 量化误差 101% 在专家上、随路由逐位变(fable5 10-21 账本),
 *   10-29 两趟逐层靶证明这份误差里有用的部分 x·B·A 表达不出来。这里每层共用一组底(B、A), 每个专家一组 K 维系数 S[e], 按这一位的路由权重合成。
 *   S = 0 时与 ds4_gpu_v41_amp_apply_tensor 同一个函数(多乘一个 1.0f, 逐位同)。
 * 记号与 ds4_gpu_v41_amp_apply_tensor 相同(行主序 ↔ 列主序的账见那里); sel/rw 是本层 MoE 的路由(int32 / f32 [n][NU]), S 是 [NE][K] f32。 */

/* T[t][k] *= u[t][k](前向) */
__global__ static void amp_route_scale_kernel(float *T, const float *S, const int32_t *sel, const float *rw, uint32_t K, uint32_t NU) {
    const uint32_t t = blockIdx.x;
    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) {
        float u = 1.0f;
        for (uint32_t j = 0; j < NU; j++) u += rw[t * NU + j] * S[(uint64_t)sel[t * NU + j] * K + k];
        T[(uint64_t)t * K + k] *= u;
    }
}

int ds4_gpu_v41_amp_route_apply_tensor(ds4_gpu_tensor *y, const ds4_gpu_tensor *x, const ds4_gpu_tensor *A, const ds4_gpu_tensor *B,
                                       const ds4_gpu_tensor *S, const ds4_gpu_tensor *sel, const ds4_gpu_tensor *rw, ds4_gpu_tensor *T,
                                       uint32_t n_tok, uint32_t D, uint32_t K, uint32_t NU, uint32_t NE) {
    if (!y || !x || !A || !B || !S || !sel || !rw || !T || !g_cublas_ready || n_tok == 0 || K == 0 || NU == 0) return 0;
    if (T->bytes < (uint64_t)n_tok * K * 4 || S->bytes < (uint64_t)NE * K * 4 || sel->bytes < (uint64_t)n_tok * NU * 4) return 0;
    const float a1 = 1.0f, b0 = 0.0f, b1 = 1.0f;
    cudaStream_t s = v41_cublas_stream();
    (void)cublasSetStream(g_cublas, s);
    cublasStatus_t st = cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)K, (int)n_tok, (int)D, &a1,
                                    (const float *)B->ptr, (int)D, (const float *)x->ptr, (int)D, &b0, (float *)T->ptr, (int)K);
    if (!cublas_ok(st, "amp route T=xB")) return 0;
    amp_route_scale_kernel<<<n_tok, K < 256u ? K : 256u, 0, s>>>((float *)T->ptr, (const float *)S->ptr, (const int32_t *)sel->ptr, (const float *)rw->ptr, K, NU);
    if (!cuda_ok(cudaGetLastError(), "amp route scale")) return 0;
    st = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)D, (int)n_tok, (int)K, &a1,
                     (const float *)A->ptr, (int)D, (const float *)T->ptr, (int)K, &b1, (float *)y->ptr, (int)D);
    return cublas_ok(st, "amp route y+=TA");
}

/* 反向中段: 进来 T = x·B(T₀), gT = gy·Aᵀ(对 T' = T₀⊙u 的梯度); 出去 T = T'(给 gA), gT = gT'⊙u(对 T₀ 的梯度, 给 gB/gx),
 * gS[e][k] += Σ_{t,j: sel=e} rw[t][j]·gT'[t][k]·T₀[t][k]。路由权重 rw 当常数(放大器对 rw 的梯度不回传给路由, 见 core_restoreb_bwd.c)。 */
__global__ static void amp_route_bwd_kernel(float *T, float *gT, float *gS, const float *S, const int32_t *sel, const float *rw, uint32_t K, uint32_t NU) {
    const uint32_t t = blockIdx.x;
    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) {
        const uint64_t i = (uint64_t)t * K + k;
        float u = 1.0f;
        for (uint32_t j = 0; j < NU; j++) u += rw[t * NU + j] * S[(uint64_t)sel[t * NU + j] * K + k];
        const float t0 = T[i], g1 = gT[i], gc = g1 * t0;
        for (uint32_t j = 0; j < NU; j++) atomicAdd(&gS[(uint64_t)sel[t * NU + j] * K + k], rw[t * NU + j] * gc);
        T[i] = t0 * u; gT[i] = g1 * u;
    }
}

int ds4_gpu_bwd_amp_route_tensor(ds4_gpu_tensor *gA, ds4_gpu_tensor *gB, ds4_gpu_tensor *gS, ds4_gpu_tensor *gx, const ds4_gpu_tensor *gy,
                                 const ds4_gpu_tensor *x, const ds4_gpu_tensor *A, const ds4_gpu_tensor *B, const ds4_gpu_tensor *S,
                                 const ds4_gpu_tensor *sel, const ds4_gpu_tensor *rw, ds4_gpu_tensor *T, ds4_gpu_tensor *gT,
                                 uint32_t n_tok, uint32_t D, uint32_t K, uint32_t NU, uint32_t NE) {
    if (!gA || !gB || !gS || !gy || !x || !A || !B || !S || !sel || !rw || !T || !gT || !g_cublas_ready || n_tok == 0 || NU == 0) return 0;
    if (S->bytes < (uint64_t)NE * K * 4 || gS->bytes < (uint64_t)NE * K * 4) return 0;
    const float one = 1.f, zero = 0.f;
    cudaStream_t s = v41_cublas_stream();
    (void)cublasSetStream(g_cublas, s);
    cublasStatus_t st = cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)K, (int)n_tok, (int)D, &one,
                                    (const float *)B->ptr, (int)D, (const float *)x->ptr, (int)D, &zero, (float *)T->ptr, (int)K);
    if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)K, (int)n_tok, (int)D, &one,
                         (const float *)A->ptr, (int)D, (const float *)gy->ptr, (int)D, &zero, (float *)gT->ptr, (int)K);
    if (!cublas_ok(st, "bwd amp route T/gT")) return 0;
    amp_route_bwd_kernel<<<n_tok, K < 256u ? K : 256u, 0, s>>>((float *)T->ptr, (float *)gT->ptr, (float *)gS->ptr, (const float *)S->ptr,
                                                                (const int32_t *)sel->ptr, (const float *)rw->ptr, K, NU);
    if (!cuda_ok(cudaGetLastError(), "bwd amp route mid")) return 0;
    st = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T, (int)D, (int)K, (int)n_tok, &one,   /* gA += gy·T'ᵀ */
                     (const float *)gy->ptr, (int)D, (const float *)T->ptr, (int)K, &one, (float *)gA->ptr, (int)D);
    if (st == CUBLAS_STATUS_SUCCESS)   /* gB += x·gT₀ᵀ */
        st = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T, (int)D, (int)K, (int)n_tok, &one,
                         (const float *)x->ptr, (int)D, (const float *)gT->ptr, (int)K, &one, (float *)gB->ptr, (int)D);
    if (st == CUBLAS_STATUS_SUCCESS && gx)   /* gx += B·gT₀ */
        st = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, (int)D, (int)n_tok, (int)K, &one,
                         (const float *)B->ptr, (int)D, (const float *)gT->ptr, (int)K, &one, (float *)gx->ptr, (int)D);
    return cublas_ok(st, "bwd amp route gA/gB/gx");
}
