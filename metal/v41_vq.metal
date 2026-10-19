// v41_vq.metal — VQ 专家的"解码即乘"(Metal, 2026-10-08), 对应 cuda_vq_row / cuda_vq_decode / cuda_vq_prefill 一族 + 草稿塔 dense MoE(cuda_v41_draft)。
// 盘上布局的唯一消费点在 v41_vq_open / v41_vq_code / v41_vq_cw 三件: DQVL v2(载荷自带 f16 码本, NBIT 位直排位流) 与
// v3(层码本 E4M3 一层一本, 12 位主流 + 13 位层的位平面), 与 vq_fmt.h / cuda_vq_row.inc.cu 逐式同。
// 值 = 码本[idx][d] × g_r[row] × 反修覆盖 gov[row]; 专家内按官方 Expert 的 bf16 边界舍: gate/up 出 bf16, h 出 bf16, down 出 bf16。
// 解码(n ≤ 8): 一 (token, 专家) 对一组 threadgroup, 一 simdgroup 一行; 预填: 按专家分组, 同一行的码字只解一次、对该专家的每个 token 各乘一遍。
// 必须排在 v41_common.metal 之后。

struct v41_vq_mat { device const uchar *cb; device const uchar *gr; device const uchar *ix; device const uchar *ex; device const half *cbh; uint nidx_row, mb, imsk, nc; bool ok, v3, has_ex, cbh_on; };
inline ulong v41_rd_u64(device const uchar *p) { return (ulong)(*(device const uint *)p) | ((ulong)(*(device const uint *)(p + 4)) << 32); }
inline v41_vq_mat v41_vq_open(device const uchar *blob, uint e, uint which, uint rows, uint cols, uint v3) {
    v41_vq_mat m; m.ok = false; m.v3 = v3 != 0u; m.has_ex = false; m.cbh_on = false; m.cbh = (device const half *)blob; m.cb = m.gr = m.ix = m.ex = blob; m.nidx_row = m.mb = m.imsk = m.nc = 0u;
    const ulong off = v41_rd_u64(blob + 16u + ((ulong)e * 3u + which) * 8u);
    if (off == 0ul) return m;
    device const uchar *pay = blob + off;
    const uint mg = *(device const uint *)pay;
    if (mg != (v3 ? 0x33565144u : 0x51565144u)) return m;   /* 'DQV3' / 'DQVQ' */
    const uint d16 = *(device const ushort *)(pay + 4), n16 = *(device const ushort *)(pay + 6);
    const uint r32 = *(device const uint *)(pay + 8), c32 = *(device const uint *)(pay + 12);
    if (r32 != rows || c32 != cols || d16 != 8u) return m;
    uint nbit = 0u; while ((1u << nbit) < n16) nbit++; if (nbit < 1u) nbit = 1u;
    m.nidx_row = cols / 8u; m.nc = n16;
    if (v3) {
        const uint flags = *(device const uint *)(pay + 16), mnb = *(device const uint *)(pay + 20);
        const ulong cb_off = v41_rd_u64(pay + 24);
        if (mnb != 12u || !(flags & 1u)) return m;
        m.mb = 12u; m.imsk = 0xFFFu;
        m.cb = blob + cb_off; m.gr = pay + 32u; m.ix = m.gr + (ulong)rows * 2u;
        const uint mrow = (m.nidx_row * 12u + 7u) / 8u;
        m.has_ex = (flags & 2u) != 0u;
        if (m.has_ex) m.ex = m.ix + (ulong)rows * mrow;
        if (m.has_ex != (nbit > 12u)) return m;
    } else {
        m.mb = nbit; m.imsk = (1u << nbit) - 1u;
        m.cb = pay + 16u; m.gr = m.cb + (ulong)n16 * 16u; m.ix = m.gr + (ulong)rows * 2u;
    }
    m.ok = true;
    return m;
}
// 第 r 行第 j 个码字号(主流 LSB 先读 + 可选第 13 位平面), 与 cuda_bwd_vq 的 vqb_code 同式
inline uint v41_vq_code(const thread v41_vq_mat &m, uint r, uint j) {
    device const uchar *rowp = m.ix + (((ulong)r * m.nidx_row * m.mb) >> 3);
    const uint bit = j * m.mb, by = bit >> 3, sh = bit & 7u;
    const uint w = (uint)rowp[by] | ((uint)rowp[by + 1u] << 8) | ((uint)rowp[by + 2u] << 16);
    uint v = (w >> sh) & m.imsk;
    if (m.v3 && m.has_ex) v |= ((uint)(m.ex[(ulong)r * ((m.nidx_row + 7u) >> 3) + (j >> 3)] >> (j & 7u)) & 1u) << 12;
    return v;
}
// 码字 v → 8 个 f32(v3: 8 个 E4M3; v2: 8 个 f16)
inline void v41_vq_cw(const thread v41_vq_mat &m, uint v, thread float *c) {
    if (m.cbh_on) {   /* 主机预解的 f16 层码本(E4M3 在 f16 里精确, 值逐位同) */
        device const half4 *p = (device const half4 *)(m.cbh + (ulong)v * 8u);
        const float4 lo = float4(p[0]), hi = float4(p[1]);
        c[0] = lo.x; c[1] = lo.y; c[2] = lo.z; c[3] = lo.w; c[4] = hi.x; c[5] = hi.y; c[6] = hi.z; c[7] = hi.w;
        return;
    }
    if (m.v3) { device const uchar *p = m.cb + (ulong)v * 8u; for (uint q = 0; q < 8u; q++) c[q] = v41_e4m3_to_f32(p[q]); }
    else { device const ushort *p = (device const ushort *)(m.cb + (ulong)v * 16u); for (uint q = 0; q < 8u; q++) c[q] = v41_f16_to_f32(p[q]); }
}
inline float v41_vq_dot8(const thread float *c, device const float *x) {
    float s = 0.0f;
    for (uint q = 0; q < 8u; q++) s += c[q] * x[q];
    return s;
}
inline float v41_vq_gain(const thread v41_vq_mat &m, uint r, device const float *gov, uint has_gov) {
    return v41_f16_to_f32(*(device const ushort *)(m.gr + (ulong)r * 2u)) * (has_gov ? gov[r] : 1.0f);
}
// 一 simdgroup 算一行与 x 的点积(已 simd 规约, 含增益)
inline float v41_vq_row_dot(const thread v41_vq_mat &m, uint r, device const float *x, device const float *gov, uint has_gov, uint lane) {
    float acc = 0.0f;
    for (uint j = lane; j < m.nidx_row; j += 32u) {
        float c[8];
        v41_vq_cw(m, v41_vq_code(m, r, j), c);
        acc += v41_vq_dot8(c, x + (ulong)j * 8u);
    }
    acc = simd_sum(acc);
    return acc * v41_vq_gain(m, r, gov, has_gov);
}
inline float v41_vq_swiglu(float gi, float ui, float clamp_v) {
    if (clamp_v > 0.0f) { if (gi > clamp_v) gi = clamp_v; if (ui > clamp_v) ui = clamp_v; if (ui < -clamp_v) ui = -clamp_v; }
    return v41_bf16r((gi / (1.0f + exp(-gi))) * ui);
}
struct v41_vq_args { uint IN, MID, OUT, K, v3, has_gov, n_tok, n_act; float clamp_v; uint pad0, pad1, pad2; };

// ---- 解码(n ≤ 8): gate/up 同核, grid (ceil(MID/8), n_tok·K), 256 线程; down grid (ceil(OUT/8), np)
kernel void kernel_v41_vq_gateup(constant v41_vq_args &a [[buffer(0)]], device float *h [[buffer(1)]], device const uchar *blob [[buffer(2)]],
                                 device const int *sel [[buffer(3)]], device const float *x [[buffer(4)]], constant ulong &blob_off [[buffer(5)]],
                                 device const half *cbh [[buffer(6)]],
                                 uint2 tg [[threadgroup_position_in_grid]], uint sgitg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint pair = tg.y, t = pair / a.K, r = tg.x * 8u + sgitg;
    const int e = sel[pair];
    if (e < 0 || r >= a.MID) return;
    device const uchar *b = blob + blob_off;
    v41_vq_mat mg = v41_vq_open(b, (uint)e, 0u, a.MID, a.IN, a.v3), mu = v41_vq_open(b, (uint)e, 1u, a.MID, a.IN, a.v3);
    if (!mg.ok || !mu.ok) return;
    if (a.pad0) { mg.cbh = cbh; mg.cbh_on = true; mu.cbh = cbh; mu.cbh_on = true; }
    device const float *xs = x + (ulong)t * a.IN;
    const float gv = v41_bf16r(v41_vq_row_dot(mg, r, xs, x, 0u, lane));
    const float ui = v41_bf16r(v41_vq_row_dot(mu, r, xs, x, 0u, lane));
    if (lane == 0u) h[(ulong)pair * a.MID + r] = v41_vq_swiglu(gv, ui, a.clamp_v);
}
kernel void kernel_v41_vq_down(constant v41_vq_args &a [[buffer(0)]], device float *partial [[buffer(1)]], device const uchar *blob [[buffer(2)]],
                               device const int *sel [[buffer(3)]], device const float *h [[buffer(4)]], constant ulong &blob_off [[buffer(5)]],
                               device const float *gov [[buffer(6)]], device const half *cbh [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint sgitg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint pair = tg.y, r = tg.x * 8u + sgitg;
    const int e = sel[pair];
    if (r >= a.OUT) return;
    if (e < 0) { if (lane == 0u) partial[(ulong)pair * a.OUT + r] = 0.0f; return; }
    v41_vq_mat md = v41_vq_open(blob + blob_off, (uint)e, 2u, a.OUT, a.MID, a.v3);
    if (!md.ok) { if (lane == 0u) partial[(ulong)pair * a.OUT + r] = 0.0f; return; }
    if (a.pad0) { md.cbh = cbh; md.cbh_on = true; }
    const float y = v41_bf16r(v41_vq_row_dot(md, r, h + (ulong)pair * a.MID, gov + (ulong)e * a.OUT, a.has_gov, lane));
    if (lane == 0u) partial[(ulong)pair * a.OUT + r] = y;
}
// ---- 解码快路(2026-10-17 M5): 只在 v3 + 12 位主流码号 + f16 层码本 + 行 4 B 对齐时走, 否则逐矩阵退回上面的原行点积。
// 8 码字一组 = 12 B = 3 次对齐 uint 读 → 8 个 12 位号, 8 次码本取料互不依赖(gather 是延迟瓶颈, 原核一次一个码字)。
// 一 simdgroup 管多行、每行 4 lane: gate/up = 4 行 × {gate, up} 8 组, down = 8 行; 同一时刻 8 组 lane 读同一段 x ⇒ 硬件合并成广播。
// 累加结合序与原核不同 ⇒ 门是 NLL(PR 说明里的 HumanEval+ teacher-force 门)。
#define V41_VQ_Q12_XMAX 6144u
inline bool v41_vq_q12_ok(const thread v41_vq_mat &m, device const uchar *b, ulong blob_off) {
    return m.v3 && m.mb == 12u && m.cbh_on && (m.nidx_row & 7u) == 0u && m.nidx_row * 8u <= V41_VQ_Q12_XMAX &&
           ((blob_off + (ulong)(m.ix - b)) & 3ul) == 0ul;
}
// 一行的部分点积: 本 lane 管第 sub, sub+subw, … 组(每组 8 码字 = 64 元素), 调用方在子组内 simd_shuffle_xor 规约。
inline float v41_vq_row_part(const thread v41_vq_mat &m, uint r, device const float *xs, uint sub, uint subw) {
    const uint ng = m.nidx_row >> 3;
    device const uint *rowp = (device const uint *)(m.ix + (((ulong)r * m.nidx_row * 12u) >> 3));
    device const uchar *exr = m.ex + (ulong)r * ((m.nidx_row + 7u) >> 3);
    float acc = 0.0f;
    for (uint gi = sub; gi < ng; gi += subw) {
        const uint w0 = rowp[3u * gi], w1 = rowp[3u * gi + 1u], w2 = rowp[3u * gi + 2u];
        uint v[8];
        v[0] = w0 & 0xFFFu;                  v[1] = (w0 >> 12) & 0xFFFu;
        v[2] = ((w0 >> 24) | (w1 << 8)) & 0xFFFu; v[3] = (w1 >> 4) & 0xFFFu;
        v[4] = (w1 >> 16) & 0xFFFu;          v[5] = ((w1 >> 28) | (w2 << 4)) & 0xFFFu;
        v[6] = (w2 >> 8) & 0xFFFu;           v[7] = (w2 >> 20) & 0xFFFu;
        if (m.has_ex) { const uint eb = exr[gi]; for (uint k = 0; k < 8u; k++) v[k] |= ((eb >> k) & 1u) << 12; }
        device const float4 *xg = (device const float4 *)(xs + gi * 64u);
        float sacc = 0.0f;
        for (uint k = 0; k < 8u; k++) {
            device const half4 *cp = (device const half4 *)(m.cbh + (ulong)v[k] * 8u);
            sacc += dot(float4(cp[0]), xg[2u * k]) + dot(float4(cp[1]), xg[2u * k + 1u]);
        }
        acc += sacc;
    }
    return acc;
}
// gate/up: grid (ceil(MID/32), n_tok·K), 256 线程 = 8 simdgroup × 4 行
kernel void kernel_v41_vq_gateup_q12(constant v41_vq_args &a [[buffer(0)]], device float *h [[buffer(1)]], device const uchar *blob [[buffer(2)]],
                                  device const int *sel [[buffer(3)]], device const float *x [[buffer(4)]], constant ulong &blob_off [[buffer(5)]],
                                  device const half *cbh [[buffer(6)]],
                                  uint2 tg [[threadgroup_position_in_grid]], uint sgitg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint pair = tg.y, t = pair / a.K, grp = lane >> 2, sub = lane & 3u;
    const uint rbase = (tg.x * 8u + sgitg) * 4u, r = rbase + (grp >> 1);
    const bool up = (grp & 1u) != 0u, live = r < a.MID;
    const int e = sel[pair];
    if (e < 0 || rbase >= a.MID) return;
    device const float *xt = x + (ulong)t * a.IN;
    device const uchar *b = blob + blob_off;
    v41_vq_mat mg = v41_vq_open(b, (uint)e, 0u, a.MID, a.IN, a.v3), mu = v41_vq_open(b, (uint)e, 1u, a.MID, a.IN, a.v3);
    if (!mg.ok || !mu.ok) return;
    mg.cbh = cbh; mg.cbh_on = true; mu.cbh = cbh; mu.cbh_on = true;
    if (v41_vq_q12_ok(mg, blob, blob_off) && v41_vq_q12_ok(mu, blob, blob_off)) {
        float p = live ? v41_vq_row_part(up ? mu : mg, r, xt, sub, 4u) : 0.0f;
        p += simd_shuffle_xor(p, 2u); p += simd_shuffle_xor(p, 1u);
        const float pu = simd_shuffle_down(p, 4u);
        if (!up && sub == 0u && live) {
            const float gv = v41_bf16r(p * v41_vq_gain(mg, r, x, 0u)), ui = v41_bf16r(pu * v41_vq_gain(mu, r, x, 0u));
            h[(ulong)pair * a.MID + r] = v41_vq_swiglu(gv, ui, a.clamp_v);
        }
    } else {
        for (uint q = 0; q < 4u; q++) {
            const uint rr = rbase + q;
            if (rr >= a.MID) break;
            const float gv = v41_bf16r(v41_vq_row_dot(mg, rr, xt, x, 0u, lane));
            const float ui = v41_bf16r(v41_vq_row_dot(mu, rr, xt, x, 0u, lane));
            if (lane == 0u) h[(ulong)pair * a.MID + rr] = v41_vq_swiglu(gv, ui, a.clamp_v);
        }
    }
}
// down: grid (ceil(OUT/64), np), 256 线程 = 8 simdgroup × 8 行, 每行 4 lane
kernel void kernel_v41_vq_down_q12(constant v41_vq_args &a [[buffer(0)]], device float *partial [[buffer(1)]], device const uchar *blob [[buffer(2)]],
                                device const int *sel [[buffer(3)]], device const float *h [[buffer(4)]], constant ulong &blob_off [[buffer(5)]],
                                device const float *gov [[buffer(6)]], device const half *cbh [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],
                                uint sgitg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint pair = tg.y, sub = lane & 3u, rbase = (tg.x * 8u + sgitg) * 8u, r = rbase + (lane >> 2);
    const int e = sel[pair];
    const bool live = r < a.OUT;
    device const float *hp = h + (ulong)pair * a.MID;
    if (e < 0) { if (live && sub == 0u) partial[(ulong)pair * a.OUT + r] = 0.0f; return; }
    v41_vq_mat md = v41_vq_open(blob + blob_off, (uint)e, 2u, a.OUT, a.MID, a.v3);
    if (!md.ok) { if (live && sub == 0u) partial[(ulong)pair * a.OUT + r] = 0.0f; return; }
    md.cbh = cbh; md.cbh_on = true;
    if (v41_vq_q12_ok(md, blob, blob_off)) {
        float p = live ? v41_vq_row_part(md, r, hp, sub, 4u) : 0.0f;
        p += simd_shuffle_xor(p, 2u); p += simd_shuffle_xor(p, 1u);
        if (live && sub == 0u) partial[(ulong)pair * a.OUT + r] = v41_bf16r(p * v41_vq_gain(md, r, gov + (ulong)e * a.OUT, a.has_gov));
    } else {
        for (uint q = 0; q < 8u; q++) {
            const uint rr = rbase + q;
            if (rr >= a.OUT) break;
            const float y = v41_bf16r(v41_vq_row_dot(md, rr, hp, gov + (ulong)e * a.OUT, a.has_gov, lane));
            if (lane == 0u) partial[(ulong)pair * a.OUT + rr] = y;
        }
    }
}
// out[t][o] = Σ_k w[t][k]·partial[t·K+k][o]; tail 版再 + so 并舍 bf16(MoE 尾巴四发合一)
kernel void kernel_v41_vq_reduce(constant v41_vq_args &a [[buffer(0)]], device float *out [[buffer(1)]], device const float *partial [[buffer(2)]],
                                 device const float *w [[buffer(3)]], device const float *so [[buffer(4)]], constant uint &tail [[buffer(5)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    const uint o = gid.x, t = gid.y;
    if (o >= a.OUT) return;
    float s = 0.0f;
    for (uint k = 0; k < a.K; k++) s += w[(ulong)t * a.K + k] * partial[((ulong)t * a.K + k) * a.OUT + o];
    out[(ulong)t * a.OUT + o] = tail ? v41_bf16r(s + so[(ulong)t * a.OUT + o]) : s;
}

// ---- 预填(n > 8, 按专家分组): meta = [act(n_act) | off(n_act) | cnt(n_act)], perm[排序位置] = 对号(对号/K = token)。
// 一 threadgroup 8 行(一 simdgroup 一行), grid (ceil(rows/8), n_act); 一行的码字每 16 个 token 解一遍。
#define V41_VQP_MT 16u
inline void v41_vq_rowdot_grp(const thread v41_vq_mat &m, uint r, device const float *x, uint xstride, uint K, device const int *perm, uint p0, uint mcnt,
                              device float *outrows, uint ostride, device const float *gov, uint has_gov, uint lane, uint mode) {
    const float gain = v41_vq_gain(m, r, gov, has_gov);
    for (uint t0 = 0; t0 < mcnt; t0 += V41_VQP_MT) {
        const uint mt = mcnt - t0 < V41_VQP_MT ? mcnt - t0 : V41_VQP_MT;
        float acc[V41_VQP_MT];
        for (uint t = 0; t < V41_VQP_MT; t++) acc[t] = 0.0f;
        for (uint j = lane; j < m.nidx_row; j += 32u) {
            float c[8];
            v41_vq_cw(m, v41_vq_code(m, r, j), c);
            for (uint t = 0; t < V41_VQP_MT; t++) {
                if (t < mt) {
                    const uint pos = p0 + t0 + t;
                    device const float *xr = mode == 0u ? x + (ulong)(perm[pos] / (int)K) * xstride : x + (ulong)pos * xstride;
                    acc[t] += v41_vq_dot8(c, xr + (ulong)j * 8u);
                }
            }
        }
        for (uint t = 0; t < V41_VQP_MT; t++) {
            const float v = simd_sum(acc[t]);
            if (lane == 0u && t < mt) {
                const uint pos = p0 + t0 + t;
                device float *o = outrows + (ulong)pos * ostride + r;
                const float val = v41_bf16r(v * gain);
                if (mode == 0u) o[0] = val;                                   /* gate: 先存 */
                else if (mode == 1u) o[0] = v41_vq_swiglu(o[0], val, 0.0f);   /* up: 与存下的 gate 合成 h(clamp 由调用方折进 mode 2) */
                else o[0] = val;                                              /* down: ys */
            }
        }
    }
}
kernel void kernel_v41_vq_gateup_grp(constant v41_vq_args &a [[buffer(0)]], device float *h [[buffer(1)]], device const uchar *blob [[buffer(2)]],
                                     device const uint *meta [[buffer(3)]], device const int *perm [[buffer(4)]], device const float *x [[buffer(5)]],
                                     constant ulong &blob_off [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                     uint sgitg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint k = tg.y, e = meta[k], p0 = meta[a.n_act + k], mcnt = meta[2u * a.n_act + k], r = tg.x * 8u + sgitg;
    if (r >= a.MID) return;
    device const uchar *b = blob + blob_off;
    const v41_vq_mat mg = v41_vq_open(b, e, 0u, a.MID, a.IN, a.v3), mu = v41_vq_open(b, e, 1u, a.MID, a.IN, a.v3);
    if (!mg.ok || !mu.ok) return;
    v41_vq_rowdot_grp(mg, r, x, a.IN, a.K, perm, p0, mcnt, h, a.MID, x, 0u, lane, 0u);
    /* up: 与存下的 gate 合成 swiglu(含 clamp) */
    const float gain = v41_vq_gain(mu, r, x, 0u);
    for (uint t0 = 0; t0 < mcnt; t0 += V41_VQP_MT) {
        const uint mt = mcnt - t0 < V41_VQP_MT ? mcnt - t0 : V41_VQP_MT;
        float acc[V41_VQP_MT];
        for (uint t = 0; t < V41_VQP_MT; t++) acc[t] = 0.0f;
        for (uint j = lane; j < mu.nidx_row; j += 32u) {
            float c[8];
            v41_vq_cw(mu, v41_vq_code(mu, r, j), c);
            for (uint t = 0; t < V41_VQP_MT; t++) if (t < mt) acc[t] += v41_vq_dot8(c, x + (ulong)(perm[p0 + t0 + t] / (int)a.K) * a.IN + (ulong)j * 8u);
        }
        for (uint t = 0; t < V41_VQP_MT; t++) {
            const float v = simd_sum(acc[t]);
            if (lane == 0u && t < mt) { device float *o = h + (ulong)(p0 + t0 + t) * a.MID + r; o[0] = v41_vq_swiglu(o[0], v41_bf16r(v * gain), a.clamp_v); }
        }
    }
}
kernel void kernel_v41_vq_down_grp(constant v41_vq_args &a [[buffer(0)]], device float *ys [[buffer(1)]], device const uchar *blob [[buffer(2)]],
                                   device const uint *meta [[buffer(3)]], device const float *h [[buffer(4)]], constant ulong &blob_off [[buffer(5)]],
                                   device const float *gov [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                   uint sgitg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint k = tg.y, e = meta[k], p0 = meta[a.n_act + k], mcnt = meta[2u * a.n_act + k], r = tg.x * 8u + sgitg;
    if (r >= a.OUT) return;
    const v41_vq_mat md = v41_vq_open(blob + blob_off, e, 2u, a.OUT, a.MID, a.v3);
    if (!md.ok) { if (lane == 0u) for (uint t = 0; t < mcnt; t++) ys[(ulong)(p0 + t) * a.OUT + r] = 0.0f; return; }
    /* mode 2: h 按排序位置存([pos][MID]), 不经 perm 取行, perm 参数不用(传 meta 占位) */
    v41_vq_rowdot_grp(md, r, h, a.MID, a.K, (device const int *)meta, p0, mcnt, ys, a.OUT, gov + (ulong)e * a.OUT, a.has_gov, lane, 2u);
}
// out[t][o] = Σ_pk rw[t][pk]·ys[inv[t·K+pk]][o](pick 序固定); expand: dst[pk][o] = ys[inv[pk]][o](缺席 0, 反修取料)
kernel void kernel_v41_vqp_reduce(constant v41_vq_args &a [[buffer(0)]], device float *out [[buffer(1)]], device const float *ys [[buffer(2)]],
                                  device const int *inv [[buffer(3)]], device const float *rw [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    const uint o = gid.x, t = gid.y;
    if (o >= a.OUT) return;
    float s = 0.0f;
    for (uint pk = 0; pk < a.K; pk++) {
        const int i = inv[(ulong)t * a.K + pk];
        if (i < 0) continue;
        s += rw[(ulong)t * a.K + pk] * ys[(ulong)i * a.OUT + o];
    }
    out[(ulong)t * a.OUT + o] = s;
}
kernel void kernel_v41_vqp_expand(constant v41_vq_args &a [[buffer(0)]], device float *dst [[buffer(1)]], device const float *ys [[buffer(2)]],
                                  device const int *inv [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint o = gid.x, pk = gid.y;
    if (o >= a.OUT) return;
    const int i = inv[pk];
    dst[(ulong)pk * a.OUT + o] = i < 0 ? 0.0f : ys[(ulong)i * a.OUT + o];
}

// ---- 草稿塔 dense MoE: 逐专家一个 fp4x32 张量, offs[3·n_expert] = 各专家 gate/up/down 在视图里的字节偏移, vid[n_expert] = 住哪个视图
struct v41_mtp_args { uint IN, MID, OUT, K, n_expert, cur_view, pad0, pad1; float clamp_v; float pad2, pad3, pad4; };
inline float v41_fp4_row_dot(device const uchar *w, ulong off, uint r, device const float *x, uint cols, uint lane) {
    float acc = 0.0f;
    for (uint c0 = lane * 8u; c0 < cols; c0 += 256u) {
        float wv[8];
        v41_w8(V41_WT_FP4X32, w, off, w, 0ul, 0u, r, c0, cols, wv);
        for (uint q = 0; q < 8u; q++) acc += wv[q] * v41_bf16r(x[c0 + q]);
    }
    return simd_sum(acc);
}
kernel void kernel_v41_mtp_gateup(constant v41_mtp_args &a [[buffer(0)]], device float *h [[buffer(1)]], device const uchar *w [[buffer(2)]],
                                  device const ulong *offs [[buffer(3)]], device const uint *vid [[buffer(4)]], device const int *sel [[buffer(5)]],
                                  device const float *x [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                  uint sgitg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint pair = tg.y, t = pair / a.K, r = tg.x * 8u + sgitg;
    const int e = sel[pair];
    if (e < 0 || (uint)e >= a.n_expert || vid[e] != a.cur_view || r >= a.MID) return;
    device const float *xs = x + (ulong)t * a.IN;
    float gv = v41_bf16r(v41_fp4_row_dot(w, offs[e], r, xs, a.IN, lane));
    float uv = v41_bf16r(v41_fp4_row_dot(w, offs[a.n_expert + e], r, xs, a.IN, lane));
    if (a.clamp_v > 0.0f) { if (gv > a.clamp_v) gv = a.clamp_v; if (uv > a.clamp_v) uv = a.clamp_v; if (uv < -a.clamp_v) uv = -a.clamp_v; }
    if (lane == 0u) h[(ulong)pair * a.MID + r] = v41_bf16r((gv / (1.0f + exp(-gv))) * uv);
}
kernel void kernel_v41_mtp_down(constant v41_mtp_args &a [[buffer(0)]], device float *partial [[buffer(1)]], device const uchar *w [[buffer(2)]],
                                device const ulong *offs [[buffer(3)]], device const uint *vid [[buffer(4)]], device const int *sel [[buffer(5)]],
                                device const float *h [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                uint sgitg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint pair = tg.y, r = tg.x * 8u + sgitg;
    const int e = sel[pair];
    if (r >= a.OUT) return;
    if (e < 0 || (uint)e >= a.n_expert) { if (lane == 0u && a.cur_view == 0u) partial[(ulong)pair * a.OUT + r] = 0.0f; return; }
    if (vid[e] != a.cur_view) return;
    const float y = v41_fp4_row_dot(w, offs[2u * a.n_expert + e], r, h + (ulong)pair * a.MID, a.MID, lane);
    if (lane == 0u) partial[(ulong)pair * a.OUT + r] = v41_bf16r(y);
}

