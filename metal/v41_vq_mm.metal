// v41_vq_mm.metal — V4.1 VQ 专家的预填(n > 8)矩阵乘: simdgroup half 版与 Metal 4 TensorOps 版(2026-10-17 M5)。只给推理路用(见 metal_v41_moe.m)。
// 必须排在 v41_vq.metal 之后(用它的 v41_vq_mat / v41_vq_open / v41_vq_gain / v41_vq_swiglu)。

// ---- 预填 VQ 专家的 simdgroup 矩阵乘(2026-10-17 M5 提速)。按专家分组: grid (ceil(maxcnt/32), ceil(ROWS/64), n_act), 128 线程。
// A = 该专家矩阵 64 行 × 32 k(12 位码号 → f16 层码本), B = 本专家的 32 个 token 行(gate/up 经 perm 取 x 行, down 取 h 的排序行)。
// mode 0: h[pos][r] = bf16(gate·g_r);  mode 1: h = swiglu(h, bf16(up·g_r), clamp);  mode 2: ys[pos][r] = bf16(down·g_r·gov)。门是 NLL。
kernel void kernel_v41_vq_mm(constant v41_vq_args &a [[buffer(0)]], device float *outp [[buffer(1)]], device const uchar *blob [[buffer(2)]],
                             device const uint *meta [[buffer(3)]], device const int *perm [[buffer(4)]], device const float *src [[buffer(5)]],
                             constant ulong &blob_off [[buffer(6)]], device const half *cbh [[buffer(7)]], device const float *gov [[buffer(8)]],
                             uint3 tgpig [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]],
                             ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup char shmem[8192];
    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + 4096);
    constexpr int NR0 = 64, NR1 = 32, NK = 32, NL0 = 2, NL1 = 4;
    const uint mode = a.pad1, k = tgpig.z;
    const uint e = meta[k], p0 = meta[a.n_act + k], cnt = meta[2u * a.n_act + k];
    const int t0 = tgpig.x * NR1;
    if ((uint)t0 >= cnt) return;
    const uint ROWS = mode == 2u ? a.OUT : a.MID, KD = mode == 2u ? a.MID : a.IN;
    const int r0 = tgpig.y * NR0;
    if ((uint)r0 >= ROWS) return;
    device const uchar *b = blob + blob_off;
    v41_vq_mat m = v41_vq_open(b, e, mode == 2u ? 2u : mode, ROWS, KD, a.v3);
    if (!m.ok) return;
    const short nr0 = (int(ROWS) - r0 < NR0) ? short(int(ROWS) - r0) : short(NR0);
    const short nr1 = (int(cnt) - t0 < NR1) ? short(int(cnt) - t0) : short(NR1);
    const short lr0 = ((short)tiitg / NL0) < nr0 ? ((short)tiitg / NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg / NL1) < nr1 ? ((short)tiitg / NL1) : nr1 - 1;
    const short il0 = tiitg % NL0;
    const uint row = (uint)(r0 + lr0);
    device const uchar *rowp = m.ix + (((ulong)row * m.nidx_row * 12u) >> 3);
    device const uchar *exr = m.ex + (ulong)row * ((m.nidx_row + 7u) >> 3);
    const uint pos = p0 + (uint)(t0 + lr1);
    device const float *yrow = mode == 2u ? src + (ulong)pos * KD : src + (ulong)(perm[pos] / (int)a.K) * KD;
    const short iy = 8 * (tiitg % NL1);
    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (uint loop_k = 0; loop_k < KD; loop_k += NK) {
        /* A: 本线程 = 第 row 行、k ∈ [loop_k + 16·il0, +16) = 两个码字(码号 j0, j0+1; j0 偶数 ⇒ 3 字节对齐到 12 位对) */
        const uint j0 = (loop_k >> 3) + 2u * (uint)il0;
        device const uchar *pp = rowp + 3u * (j0 >> 1);
        uint v0 = (uint)pp[0] | (((uint)pp[1] & 0xFu) << 8), v1 = ((uint)pp[1] >> 4) | ((uint)pp[2] << 4);
        if (m.has_ex) { const uint eb = exr[j0 >> 3]; v0 |= ((eb >> (j0 & 7u)) & 1u) << 12; v1 |= ((eb >> ((j0 + 1u) & 7u)) & 1u) << 12; }
        device const half4 *c0 = (device const half4 *)(cbh + (ulong)v0 * 8u), *c1 = (device const half4 *)(cbh + (ulong)v1 * 8u);
        half4x4 temp_a;
        temp_a[0] = c0[0]; temp_a[1] = c0[1]; temp_a[2] = c1[0]; temp_a[3] = c1[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; i++) {
            const short sx = 2 * il0 + i / 8, sy = (tiitg / NL0) / 8;
            const short lx = (tiitg / NL0) % 8, ly = i % 8;
            *(sa + 64 * (8 * sx + sy) + 8 * ly + lx) = temp_a[i / 4][i % 4];
        }
        {
            const short sx = (tiitg % NL1), sy = (tiitg / NL1) / 8, ly = (tiitg / NL1) % 8;
            device const float *yp = yrow + loop_k + iy;
            const float4 y0 = *((device const float4 *)yp), y1 = *((device const float4 *)(yp + 4));
            threadgroup half4 *dst4 = (threadgroup half4 *)(sb + 64 * (4 * sx + sy) + 8 * ly);
            dst4[0] = half4(y0); dst4[1] = half4(y1);
        }
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
    threadgroup const float *tbase = (threadgroup const float *)shmem;
    for (int j = tiitg; j < nr1 * NR0; j += 128) {
        const int mm = j / NR0, nn = j % NR0;
        if (nn >= nr0) continue;
        const uint r = (uint)(r0 + nn), pp2 = p0 + (uint)(t0 + mm);
        const float gain = v41_vq_gain(m, r, gov + (ulong)e * a.OUT, mode == 2u ? a.has_gov : 0u);
        const float val = v41_bf16r(tbase[mm * NR0 + nn] * gain);
        device float *o = outp + (ulong)pp2 * ROWS + r;
        if (mode == 0u) o[0] = val;
        else if (mode == 1u) o[0] = v41_vq_swiglu(o[0], val, a.clamp_v);
        else o[0] = val;
    }
}

#ifdef DS4_METAL_HAS_TENSOR
// ---- 预填 VQ 的 TensorOps 版(2026-10-18 M5): 乘法交给 matmul2d, 每步 64 行 × 64 token × 64 k, 暂存与输出块共用 16 KB; 收尾同 kernel_v41_vq_mm。门是 NLL。
template <int NK>
kernel void kernel_v41_vq_mpp_t(constant v41_vq_args &a [[buffer(0)]], device float *outp [[buffer(1)]], device const uchar *blob [[buffer(2)]],
                               device const uint *meta [[buffer(3)]], device const int *perm [[buffer(4)]], device const float *src [[buffer(5)]],
                               constant ulong &blob_off [[buffer(6)]], device const half *cbh [[buffer(7)]], device const float *gov [[buffer(8)]],
                               uint3 tgpig [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]]) {
    constexpr int NR0 = 64, NR1 = 64;
    threadgroup char shm[NK == 128 ? 32768 : 16384];
    threadgroup half *sa = (threadgroup half *)shm;                 /* 64 行 × 64 k */
    threadgroup half *sb = (threadgroup half *)(shm + 64 * NK * 2);        /* 64 token × 64 k */
    threadgroup float *ot = (threadgroup float *)shm;               /* 循环后复用: 64 × 64 f32 */
    const uint mode = a.pad1, k = tgpig.z;
    const uint e = meta[k], p0 = meta[a.n_act + k], cnt = meta[2u * a.n_act + k];
    const int t0 = tgpig.x * NR1;
    if ((uint)t0 >= cnt) return;
    const uint ROWS = mode == 2u ? a.OUT : a.MID, KD = mode == 2u ? a.MID : a.IN;
    const int r0 = tgpig.y * NR0;
    if ((uint)r0 >= ROWS) return;
    device const uchar *b = blob + blob_off;
    v41_vq_mat m = v41_vq_open(b, e, mode == 2u ? 2u : mode, ROWS, KD, a.v3);
    if (!m.ok) return;
    const short nr0 = (int(ROWS) - r0 < NR0) ? short(int(ROWS) - r0) : short(NR0);
    const short nr1 = (int(cnt) - t0 < NR1) ? short(int(cnt) - t0) : short(NR1);
    auto tA = tensor(sa, dextents<int32_t, 2>(NK, NR0));
    auto tB = tensor(sb, dextents<int32_t, 2>(NK, NR1));
    matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
    /* A: 线程 → 行 tiitg/2, k 段 32·(tiitg%2) = 4 个码字(两对); B: 线程 → token tiitg/2, k 段 32·(tiitg%2) */
    const uint row_l = tiitg / 2u, half_k = (tiitg % 2u) * (uint)(NK / 2);
    const uint row = (uint)r0 + (row_l < (uint)nr0 ? row_l : (uint)nr0 - 1u);
    device const uchar *rowp = m.ix + (((ulong)row * m.nidx_row * 12u) >> 3);
    device const uchar *exr = m.ex + (ulong)row * ((m.nidx_row + 7u) >> 3);
    const uint col = tiitg / 2u;
    const uint pos = p0 + (uint)t0 + (col < (uint)nr1 ? col : (uint)nr1 - 1u);
    device const float *yrow = mode == 2u ? src + (ulong)pos * KD : src + (ulong)(perm[pos] / (int)a.K) * KD;
    for (uint loop_k = 0; loop_k < KD; loop_k += NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint pr = 0; pr < (uint)(NK / 32); pr++) {
            const uint j0 = ((loop_k + half_k) >> 3) + 2u * pr;
            device const uchar *pp = rowp + 3u * (j0 >> 1);
            uint v0 = (uint)pp[0] | (((uint)pp[1] & 0xFu) << 8), v1 = ((uint)pp[1] >> 4) | ((uint)pp[2] << 4);
            if (m.has_ex) { const uint eb = exr[j0 >> 3]; v0 |= ((eb >> (j0 & 7u)) & 1u) << 12; v1 |= ((eb >> ((j0 + 1u) & 7u)) & 1u) << 12; }
            device const half4 *c0 = (device const half4 *)(cbh + (ulong)v0 * 8u), *c1 = (device const half4 *)(cbh + (ulong)v1 * 8u);
            threadgroup half4 *dst = (threadgroup half4 *)(sa + row_l * NK + half_k + 16u * pr);
            dst[0] = c0[0]; dst[1] = c0[1]; dst[2] = c1[0]; dst[3] = c1[1];
        }
        {
            device const float4 *y4 = (device const float4 *)(yrow + loop_k + half_k);
            threadgroup half4 *dst = (threadgroup half4 *)(sb + col * NK + half_k);
            for (uint q = 0; q < (uint)(NK / 8); q++) dst[q] = half4(y4[q]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA = tA.slice(0, 0); auto mB = tB.slice(0, 0);
        mm.run(mB, mA, cT);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto tO = tensor(ot, dextents<int32_t, 2>(NR0, NR1));
    cT.store(tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = tiitg; j < nr1 * NR0; j += 128) {
        const int mm2 = j / NR0, nn = j % NR0;
        if (nn >= nr0) continue;
        const uint r = (uint)(r0 + nn), pp2 = p0 + (uint)(t0 + mm2);
        const float gain = v41_vq_gain(m, r, gov + (ulong)e * a.OUT, mode == 2u ? a.has_gov : 0u);
        const float val = v41_bf16r(ot[mm2 * NR0 + nn] * gain);
        device float *o = outp + (ulong)pp2 * ROWS + r;
        if (mode == 1u) o[0] = v41_vq_swiglu(o[0], val, a.clamp_v); else o[0] = val;
    }
}

typedef decltype(kernel_v41_vq_mpp_t<64>) v41_vq_mpp_fn;
template [[host_name("kernel_v41_vq_mpp")]] kernel v41_vq_mpp_fn kernel_v41_vq_mpp_t<64>;
#endif
