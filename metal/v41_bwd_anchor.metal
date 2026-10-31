// v41_bwd_anchor.metal — 参数锚(EWC)核(2026-10-31): facc += g²; g += lam·F·(p − p0); 每个 threadgroup 落两个部分和(Σ F·(p−p0)², Σ (lam·F·(p−p0))²)。
// 单独一个文件是因为 v41_bwd.metal 已到 500 行守卫。契约 ds4_gpu_bwd.h ds4_gpu_bwd_anchor_tensor; 发射器 src/metal/metal_bwd_dense.m; 数值与
// CUDA 的 bwd_anchor_kernel 同式(cuda_bwd_dense.inc.cu)。has_f = 0 时 F 槽绑的是哑缓冲, 只攒 facc 不碰 g。
struct v41_anchor_args { ulong n; float lam; uint has_f, pad; };
kernel void kernel_v41_bwd_anchor(constant v41_anchor_args &a [[buffer(0)]], device float *part [[buffer(1)]], device float *g [[buffer(2)]],
                                  device float *facc [[buffer(3)]], device const float *p [[buffer(4)]], device const float *p0 [[buffer(5)]],
                                  device const float *F [[buffer(6)]],
                                  uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint ntg [[threadgroups_per_grid]]) {
    threadgroup float sh[512];
    float sp = 0.0f, sg = 0.0f;
    for (ulong i = (ulong)tg * 512u + tid; i < a.n; i += (ulong)ntg * 512u) {
        const float gi = g[i];
        facc[i] += gi * gi;
        if (a.has_f != 0u) { const float d = p[i] - p0[i], fd = F[i] * d, ad = a.lam * fd; g[i] = gi + ad; sp += fd * d; sg += ad * ad; }
    }
    const float rp = v41_tg_sum(sp, sh, tid, 512u);
    const float rg = v41_tg_sum(sg, sh, tid, 512u);
    if (tid == 0u) { part[2u * tg] = rp; part[2u * tg + 1u] = rg; }
}
// 解耦拉回(ds4_gpu_bwd_pull_tensor): p -= coef·F·(p − p0), 部分和同上(Σ F·d² 拉回前, Σ 位移²); 与 CUDA 的 bwd_pull_kernel 同式
struct v41_pull_args { ulong n; float coef; uint pad0, pad1; };
kernel void kernel_v41_bwd_pull(constant v41_pull_args &a [[buffer(0)]], device float *part [[buffer(1)]], device float *p [[buffer(2)]],
                                device const float *p0 [[buffer(3)]], device const float *F [[buffer(4)]],
                                uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint ntg [[threadgroups_per_grid]]) {
    threadgroup float sh[512];
    float sp = 0.0f, sg = 0.0f;
    for (ulong i = (ulong)tg * 512u + tid; i < a.n; i += (ulong)ntg * 512u) {
        const float d = p[i] - p0[i], fd = F[i] * d, s = min(a.coef * F[i], 1.0f) * d;   // 系数封顶 1(与 CUDA 同: 不封顶重尾 F 会发散)
        p[i] -= s; sp += fd * d; sg += s * s;
    }
    const float rp = v41_tg_sum(sp, sh, tid, 512u);
    const float rg = v41_tg_sum(sg, sh, tid, 512u);
    if (tid == 0u) { part[2u * tg] = rp; part[2u * tg + 1u] = rg; }
}
