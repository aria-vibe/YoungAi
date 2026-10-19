// v41_prefill_mm.metal — V4.1 预填(n > 8)的 half 瓦片 GEMM: q4_K / bf16 / fp8blk 权重, simdgroup 矩阵与 Metal 4 TensorOps 两套(2026-10-17 M5)。
// 激活在 threadgroup 里舍成 half(llama.cpp kernel_mul_mm 同法), 只给推理路用: 训练原语留在 v41_dense.metal 的 f32 kernel_v41_wgemm(见 metal_v41_dense.m)。
// 必须排在 v41_common.metal / dense_mm_mpp.metal 之后。

// ---- q4_K 预填 GEMM(2026-10-17 M5 提速): 移植自 llama.cpp kernel_mul_mm(simdgroup half 8×8, MIT, ggml-org/llama.cpp)。
// out[m][n] (+)= Σ_k x[m][k] · W[n][k]; 输出块 64(n)×32(m), K 步 32, 128 线程。权重解成 half、激活转 half 进 threadgroup 内存, f32 累加。
// 出口一律经 threadgroup 暂存(边界 + beta 累加 + bf16 舍)。要求 wnn == 0、K % 256 == 0(主机校验)。门是 NLL。
struct v41_bq4k { half d; half dmin; uchar scales[12]; uchar qs[128]; };
inline uchar2 v41_q4k_scmin2(int j, int k, device const uchar *q) {
    return j < 4 ? uchar2{uchar(q[j + 0 + k] & 63), uchar(q[j + 4 + k] & 63)}
                 : uchar2{uchar((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xc0) >> 2)), uchar((q[j + 4 + k] >> 4) | ((q[j - 0 + k] & 0xc0) >> 2))};
}
inline void v41_deq_q4k_16(device const v41_bq4k *xb, short il, thread half4x4 &reg) {
    device const uchar *q = xb->qs;
    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = v41_q4k_scmin2(is, il / 2, xb->scales);
    const float d = il < 2 ? float(xb->d) : float(xb->d) / 16.f;
    const float mn = float(xb->dmin);
    const float dl = d * sc[0], ml = mn * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) reg[i / 4][i % 4] = half(dl * (q[i] & mask) - ml);
}
kernel void kernel_v41_wgemm_q4k_mm(constant v41_wgemm_args &a [[buffer(0)]], device const uchar *w [[buffer(1)]], device const float *xin [[buffer(2)]],
                                    device float *out [[buffer(3)]],
                                    uint3 tgpig [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]],
                                    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup char shmem[8192];
    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    constexpr int NR0 = 64, NR1 = 32, NK = 32, NL0 = NK / 16, NL1 = NK / 8, NLQ = 16;
    const uint g = tgpig.z;
    const int r0 = tgpig.y * NR0, r1 = tgpig.x * NR1;
    const short nr0 = (int(a.N) - r0 < NR0) ? short(int(a.N) - r0) : short(NR0);
    const short nr1 = (int(a.M) - r1 < NR1) ? short(int(a.M) - r1) : short(NR1);
    const short lr0 = ((short)tiitg / NL0) < nr0 ? ((short)tiitg / NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg / NL1) < nr1 ? ((short)tiitg / NL1) : nr1 - 1;
    const short il0 = (tiitg % NL0);
    short il = il0;
    const ulong rowb = (ulong)(a.K / 256u) * 144u;
    device const v41_bq4k *xq = (device const v41_bq4k *)(w + a.w_off + (ulong)g * a.w_gstride + rowb * (ulong)(r0 + lr0)) + il0 / NLQ;
    const short iy = 8 * (tiitg % NL1);
    device const float *y = xin + (ulong)g * a.x_gstride + (ulong)(r1 + lr1) * a.lda + iy;
    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (uint loop_k = 0; loop_k < a.K; loop_k += NK) {
        half4x4 temp_a;
        v41_deq_q4k_16(xq, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8, sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8, ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
        }
        {
            const short sx = (tiitg % NL1), sy = (tiitg / NL1) / 8, ly = (tiitg / NL1) % 8;
            const short ib = 4 * sx + sy;
            const float4 y0 = *((device const float4 *)y), y1 = *((device const float4 *)(y + 4));
            threadgroup half4 *dst4 = (threadgroup half4 *)(sb + 64 * ib + 8 * ly);
            dst4[0] = half4(y0); dst4[1] = half4(y1);
        }
        il = (il + 2 < NLQ) ? il + 2 : il % 2;
        xq = (il < 2) ? xq + (2 + NLQ - 1) / NLQ : xq;
        y += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half *lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half *lsmb = (sb + 2 * 64 * (sgitg / 2));
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *temp_str = ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; i++) simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *og = out + (ulong)g * a.out_gstride;
    threadgroup const float *tbase = (threadgroup const float *)shmem;
    for (int j = tiitg; j < nr1 * NR0; j += 128) {
        const int mm = j / NR0, nn = j % NR0;
        if (nn >= nr0) continue;
        device float *D = og + (ulong)(r1 + mm) * a.ldc + (r0 + nn);
        float v = tbase[mm * NR0 + nn];
        if (a.beta) v += *D;
        *D = a.round_out ? v41_bf16r(v) : v;
    }
}


#ifdef DS4_METAL_HAS_TENSOR
// q4_K 预填走 Metal 4 TensorOps(引擎现成的 kernel_mul_mm_mpp_direct_rhs 模板, 2026-10-17 M5): 权重 q4_K 解成 half 瓦片, 激活 f32 由 matmul2d 直读。
template [[host_name("kernel_v41_mm_q4K_mpp")]]      kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<32,  half, half4x4, block_q4_K, 16, dequantize_q4_K, float, float4x4, float>;
template [[host_name("kernel_v41_mm_q4K_mpp_n64")]]  kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<64,  half, half4x4, block_q4_K, 16, dequantize_q4_K, float, float4x4, float>;
template [[host_name("kernel_v41_mm_q4K_mpp_n128")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<128, half, half4x4, block_q4_K, 16, dequantize_q4_K, float, float4x4, float>;
#endif

// ---- 窄输出预填 GEMM(2026-10-17 M5): f32/bf16 权重、N 小(原瓦片核 64×64 网格只开几十个 threadgroup)。
// threadgroup = 8 个输出 × 8 个 token, 一 simdgroup 管一个输出 n, 权重行读一次给 8 个 token 用; f32 累加。门是 NLL。
kernel void kernel_v41_wgemm_skinny(constant v41_wgemm_args &a [[buffer(0)]], device const uchar *w [[buffer(1)]], device const float *x [[buffer(2)]],
                                    device float *out [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                                    uint sgitg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint n = tg.y * 8u + sgitg, m0 = tg.x * 8u;
    if (n >= a.N) return;
    float acc[8] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
    const uint mt = a.M - m0 < 8u ? a.M - m0 : 8u;
    for (uint k = lane * 4u; k < a.K; k += 128u) {
        float4 wv;
        if (a.wtype == V41_WT_F32) wv = *(device const float4 *)(w + a.w_off + ((ulong)n * a.K + k) * 4u);
        else { const ushort4 u = *(device const ushort4 *)(w + a.w_off + ((ulong)n * a.K + k) * 2u);
               wv = float4(as_type<float>((uint)u.x << 16), as_type<float>((uint)u.y << 16), as_type<float>((uint)u.z << 16), as_type<float>((uint)u.w << 16)); }
        for (uint t = 0; t < 8u; t++) if (t < mt) acc[t] += dot(wv, *(device const float4 *)(x + (ulong)(m0 + t) * a.lda + k));
    }
    for (uint t = 0; t < 8u; t++) {
        const float v = simd_sum(acc[t]);
        if (lane == 0u && t < mt) {
            device float *o = out + (ulong)(m0 + t) * a.ldc + n;
            float r = a.beta ? v + o[0] : v;
            o[0] = a.round_out ? v41_bf16r(r) : r;
        }
    }
}

#ifdef DS4_METAL_HAS_TENSOR
// bf16 权重的 TensorOps 预填(2026-10-17 M5): 16 个 bf16 当一块, 解成 half 进瓦片(router / 小投影)。门是 NLL。
struct v41_bf16x16 { ushort v[16]; };
template <typename type4x4>
void v41_deq_bf16_16(device const v41_bf16x16 *xb, short il, thread type4x4 &reg) {
    (void)il;
    for (int i = 0; i < 16; ++i) reg[i / 4][i % 4] = (half)as_type<float>((uint)xb->v[i] << 16);
}
template [[host_name("kernel_v41_mm_bf16_mpp")]]      kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<32,  half, half4x4, v41_bf16x16, 1, v41_deq_bf16_16, float, float4x4, float>;
template [[host_name("kernel_v41_mm_bf16_mpp_n64")]]  kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<64,  half, half4x4, v41_bf16x16, 1, v41_deq_bf16_16, float, float4x4, float>;
template [[host_name("kernel_v41_mm_bf16_mpp_n128")]] kernel mul_mm_mpp_direct_rhs_t kernel_mul_mm_mpp_direct_rhs<128, half, half4x4, v41_bf16x16, 1, v41_deq_bf16_16, float, float4x4, float>;
#endif

// ---- bf16 预填 GEMM(2026-10-18 M5): 同 kernel_v41_wgemm_q4k_mm 的瓦片(64 输出 × 32 token, K 步 32), 权重 bf16 → half。门是 NLL。
kernel void kernel_v41_wgemm_bf16_mm(constant v41_wgemm_args &a [[buffer(0)]], device const uchar *w [[buffer(1)]], device const float *xin [[buffer(2)]],
                                     device float *out [[buffer(3)]],
                                     uint3 tgpig [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]],
                                     ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup char shmem[8192];
    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    constexpr int NR0 = 64, NR1 = 32, NK = 32, NL0 = 2, NL1 = 4;
    const uint g = tgpig.z;
    const int r0 = tgpig.y * NR0, r1 = tgpig.x * NR1;
    const short nr0 = (int(a.N) - r0 < NR0) ? short(int(a.N) - r0) : short(NR0);
    const short nr1 = (int(a.M) - r1 < NR1) ? short(int(a.M) - r1) : short(NR1);
    const short lr0 = ((short)tiitg / NL0) < nr0 ? ((short)tiitg / NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg / NL1) < nr1 ? ((short)tiitg / NL1) : nr1 - 1;
    const short il0 = (tiitg % NL0);
    device const ushort *wrow = (device const ushort *)(w + a.w_off + (ulong)g * a.w_gstride) + (ulong)(r0 + lr0) * a.K;
    const short iy = 8 * (tiitg % NL1);
    device const float *y = xin + (ulong)g * a.x_gstride + (ulong)(r1 + lr1) * a.lda + iy;
    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (uint loop_k = 0; loop_k < a.K; loop_k += NK) {
        device const ushort *wp = wrow + loop_k + 16u * (uint)il0;
        half tv[16];
        for (short i = 0; i < 16; i++) tv[i] = (half)as_type<float>((uint)wp[i] << 16);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8, sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8, ly = i % 8;
            *(sa + 64 * (8 * sx + sy) + 8 * ly + lx) = tv[i];
        }
        {
            const short sx = (tiitg % NL1), sy = (tiitg / NL1) / 8, ly = (tiitg / NL1) % 8;
            const float4 y0 = *((device const float4 *)y), y1 = *((device const float4 *)(y + 4));
            threadgroup half4 *dst4 = (threadgroup half4 *)(sb + 64 * (4 * sx + sy) + 8 * ly);
            dst4[0] = half4(y0); dst4[1] = half4(y1);
        }
        y += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half *lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half *lsmb = (sb + 2 * 64 * (sgitg / 2));
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *temp_str = ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; i++) simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *og = out + (ulong)g * a.out_gstride;
    threadgroup const float *tbase = (threadgroup const float *)shmem;
    for (int j = tiitg; j < nr1 * NR0; j += 128) {
        const int mm = j / NR0, nn = j % NR0;
        if (nn >= nr0) continue;
        device float *D = og + (ulong)(r1 + mm) * a.ldc + (r0 + nn);
        float v = tbase[mm * NR0 + nn];
        if (a.beta) v += *D;
        *D = a.round_out ? v41_bf16r(v) : v;
    }
}

// ---- 通用(fp8blk)预填 GEMM 的 half 瓦片版(2026-10-18 M5): 同 bf16_mm 瓦片, 权重经 v41_w8 解码。只在主机确认该矩阵全部缩放后值在 half 里精确时用。
kernel void kernel_v41_wgemm_w8_mm(constant v41_wgemm_args &a [[buffer(0)]], device const uchar *w [[buffer(1)]], device const float *xin [[buffer(2)]],
                                   device float *out [[buffer(3)]],
                                   uint3 tgpig [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]],
                                   ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup char shmem[8192];
    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    constexpr int NR0 = 64, NR1 = 32, NK = 32, NL0 = 2, NL1 = 4;
    const uint g = tgpig.z;
    const int r0 = tgpig.y * NR0, r1 = tgpig.x * NR1;
    const short nr0 = (int(a.N) - r0 < NR0) ? short(int(a.N) - r0) : short(NR0);
    const short nr1 = (int(a.M) - r1 < NR1) ? short(int(a.M) - r1) : short(NR1);
    const short lr0 = ((short)tiitg / NL0) < nr0 ? ((short)tiitg / NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg / NL1) < nr1 ? ((short)tiitg / NL1) : nr1 - 1;
    const short il0 = (tiitg % NL0);
    const ulong w_off = a.w_off + (ulong)g * a.w_gstride, sc_off = a.sc_off + (ulong)g * a.sc_gstride;
    const uint row = (uint)(r0 + lr0);
    const short iy = 8 * (tiitg % NL1);
    device const float *y = xin + (ulong)g * a.x_gstride + (ulong)(r1 + lr1) * a.lda + iy;
    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (uint loop_k = 0; loop_k < a.K; loop_k += NK) {
        float o0[8], o1[8];
        const uint c0 = loop_k + 16u * (uint)il0;
        v41_w8(a.wtype, w, w_off, w, sc_off, a.sbc, row, c0, a.wcols, o0);
        v41_w8(a.wtype, w, w_off, w, sc_off, a.sbc, row, c0 + 8u, a.wcols, o1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8, sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8, ly = i % 8;
            *(sa + 64 * (8 * sx + sy) + 8 * ly + lx) = half(i < 8 ? o0[i] : o1[i - 8]);
        }
        {
            const short sx = (tiitg % NL1), sy = (tiitg / NL1) / 8, ly = (tiitg / NL1) % 8;
            const float4 y0 = *((device const float4 *)y), y1 = *((device const float4 *)(y + 4));
            threadgroup half4 *dst4 = (threadgroup half4 *)(sb + 64 * (4 * sx + sy) + 8 * ly);
            dst4[0] = half4(y0); dst4[1] = half4(y1);
        }
        y += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half *lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half *lsmb = (sb + 2 * 64 * (sgitg / 2));
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *temp_str = ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; i++) simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *og = out + (ulong)g * a.out_gstride;
    threadgroup const float *tbase = (threadgroup const float *)shmem;
    for (int j = tiitg; j < nr1 * NR0; j += 128) {
        const int mm = j / NR0, nn = j % NR0;
        if (nn >= nr0) continue;
        device float *D = og + (ulong)(r1 + mm) * a.ldc + (r0 + nn);
        float v = tbase[mm * NR0 + nn];
        if (a.beta) v += *D;
        *D = a.round_out ? v41_bf16r(v) : v;
    }
}
// bfloat 版: E4M3×2^s 与 bf16 格点激活在 bfloat 里都精确, 不受 half 范围限制
kernel void kernel_v41_wgemm_w8_mmbf(constant v41_wgemm_args &a [[buffer(0)]], device const uchar *w [[buffer(1)]], device const float *xin [[buffer(2)]],
                                   device float *out [[buffer(3)]],
                                   uint3 tgpig [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]],
                                   ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup char shmem[8192];
    threadgroup bfloat *sa = (threadgroup bfloat *)(shmem);
    threadgroup bfloat *sb = (threadgroup bfloat *)(shmem + 4096);
    constexpr int NR0 = 64, NR1 = 32, NK = 32, NL0 = 2, NL1 = 4;
    const uint g = tgpig.z;
    const int r0 = tgpig.y * NR0, r1 = tgpig.x * NR1;
    const short nr0 = (int(a.N) - r0 < NR0) ? short(int(a.N) - r0) : short(NR0);
    const short nr1 = (int(a.M) - r1 < NR1) ? short(int(a.M) - r1) : short(NR1);
    const short lr0 = ((short)tiitg / NL0) < nr0 ? ((short)tiitg / NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg / NL1) < nr1 ? ((short)tiitg / NL1) : nr1 - 1;
    const short il0 = (tiitg % NL0);
    const ulong w_off = a.w_off + (ulong)g * a.w_gstride, sc_off = a.sc_off + (ulong)g * a.sc_gstride;
    const uint row = (uint)(r0 + lr0);
    const short iy = 8 * (tiitg % NL1);
    device const float *y = xin + (ulong)g * a.x_gstride + (ulong)(r1 + lr1) * a.lda + iy;
    simdgroup_bfloat8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (uint loop_k = 0; loop_k < a.K; loop_k += NK) {
        float o0[8], o1[8];
        const uint c0 = loop_k + 16u * (uint)il0;
        v41_w8(a.wtype, w, w_off, w, sc_off, a.sbc, row, c0, a.wcols, o0);
        v41_w8(a.wtype, w, w_off, w, sc_off, a.sbc, row, c0 + 8u, a.wcols, o1);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8, sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8, ly = i % 8;
            *(sa + 64 * (8 * sx + sy) + 8 * ly + lx) = bfloat(i < 8 ? o0[i] : o1[i - 8]);
        }
        {
            const short sx = (tiitg % NL1), sy = (tiitg / NL1) / 8, ly = (tiitg / NL1) % 8;
            const float4 y0 = *((device const float4 *)y), y1 = *((device const float4 *)(y + 4));
            threadgroup bfloat4 *dst4 = (threadgroup bfloat4 *)(sb + 64 * (4 * sx + sy) + 8 * ly);
            dst4[0] = bfloat4(y0); dst4[1] = bfloat4(y1);
        }
        y += NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const bfloat *lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const bfloat *lsmb = (sb + 2 * 64 * (sgitg / 2));
        for (short ik = 0; ik < NK / 8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float *temp_str = ((threadgroup float *)shmem) + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
    for (short i = 0; i < 8; i++) simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *og = out + (ulong)g * a.out_gstride;
    threadgroup const float *tbase = (threadgroup const float *)shmem;
    for (int j = tiitg; j < nr1 * NR0; j += 128) {
        const int mm = j / NR0, nn = j % NR0;
        if (nn >= nr0) continue;
        device float *D = og + (ulong)(r1 + mm) * a.ldc + (r0 + nn);
        float v = tbase[mm * NR0 + nn];
        if (a.beta) v += *D;
        *D = a.round_out ? v41_bf16r(v) : v;
    }
}
