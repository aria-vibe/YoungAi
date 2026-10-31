/* core_restoreb_init.c — 段 B 还原训练的起点(2026-10-31): 读起点段 B(rb_load_init, 从 core_restoreb.c 挪来) + 参数锚(EWC)。
 *
 * 参数锚是什么(用户 10-31: "之前指标也没有维持, 按方案 3 改造"): 每轮都从现役段 B 续训, 新料把参数拉向新料, 上一轮学到的东西没有东西按着就漂
 *   (t001 修好的 wordy 落笔样式, t002 不带它的料就得重学; 编程尺 85.03 → 84.93 → 84.89 每轮蹭掉 0.1)。回放旧料能按住, 但料随轮数无限长, 用户不要。
 *   锚 = 一份数据都不带、成本固定的保持: 损失加 λ/2·Σ_i F_i·(θ_i − θ*_i)², θ* = 起点(rb_init 读进来的那份), F_i = 以前训练时第 i 个参数
 *   梯度的平方均值(经验 Fisher 对角: 以前的料在乎哪些坐标, 这一轮就别动哪些坐标)。EWC(Kirkpatrick 2017)的在线形式; 一轮多存两张和 amp 同形状
 *   的表(每层 F_A / F_B), 不随轮数长。
 * 怎么跑: 每轮训练都攒本轮数据梯度的 g²(rb_anchor_step 在 Adam 之前调: 攒的是数据梯度, 锚梯度不算进去), 落盘 <ckpt>/fisher_Lnn.bin
 *   (头 int32 {E, K, 攒过的轮数}, 然后 F_A[K][E]、F_B[K][E] f32, 与 amp_Lnn.bin 同序)。存的是历来各轮的逐坐标最大值 F = max(F_old, F_本轮):
 *   有界(不像求和随轮数涨到训不动), 也不稀释(不像平均把早先的轮越摊越薄) —— 某个坐标只要有一轮在乎过, 以后就按着。
 *   rb_anchor=λ > 0 要起点目录有 fisher_Lnn.bin(没有 = 停车: 先用 lr=0 跑一趟只攒 F, 轮脚本 --fisher-only); 扩秩多出来的秩 F=0、不拉。
 * 怎么拉(rb_anchor=S, 10-31 晚第三版, 解耦拉回): 每个 Adam 步之后直接按位移拉 θ_i ← θ_i − (F_i / (F̄·S))·(θ_i − θ*_i), F̄ = 起点 F 的均值。
 *   含义: 一般重要(F_i = F̄)的坐标每步被拉回当前位移的 1/S, 等于"漂出去 S 步就被按住"; 更重要的坐标(F_i = 10 F̄)漂 S/10 步就按住; 旧料不在乎的(F_i ≈ 0)不管。
 *   为什么不走梯度(前两版): ①EWC 原味 λ=1: t002 料攒出 F̄ 1.23e-9, 一轮漂 4e-4, 锚梯度 5e-13 对数据梯度 4e-5 差 8 个量级, 等于没拉;
 *   ②按数据定标 λ = g_rms/(F̄·lr·S) 加进梯度: Adam 按坐标归一化, 锚梯度和数据梯度一起被归掉 —— F 大的坐标里锚占主导也只走 lr, F 小的坐标里数据占主导也走 lr,
 *     λ 翻倍位移几乎不变(t006 S=10 → t007 S=5: 训练料 KLD 降幅 8.7% → 7.7%, 尺上没任何变化)。拉回必须在 Adam 之外做, 像 AdamW 对权重衰减那样解耦。
 *   train.log 每步 "anchor step i pen P pull D"(P = Σ F·Δθ² 拉回前, D = 本步拉回位移的逐坐标 rms, 和 lr 比: D ≈ lr 时锚与数据势均力敌)。
 * 出错会怎样: fisher 文件缺/形状不对 → 返回 false 停车; 不给 rb_anchor 只攒不拉, 训练与原来逐位同(锚梯度 0)。 */
#include "core_restoreb.h"
#include "../common/ds4_gr_fnv.h"
#ifndef DS4_NO_GPU

/* 起点: 秩 K0 ≤ 本趟 K。A 的多出行置 0、B 的多出行保留 pt_run_alloc 给的随机值 ⇒ 起点输出与起点目录逐位同 */
bool rb_load_init(const pt_cfg *c, pt_run *r, const char *dir) {
    const uint32_t E = DS4_N_EMBD, K = r->K;
    const uint64_t nel = (uint64_t)K * E;
    float *buf = xmalloc((size_t)nel * 4);
    bool ok = true; int32_t k0 = 0;
    for (uint32_t il = c->layer_lo; il <= c->layer_hi && ok; il++) {
        char p[1200]; snprintf(p, sizeof p, "%s/amp_L%02u.bin", dir, il);
        FILE *f = fopen(p, "rb");
        int32_t hd[3] = { 0, 0, 0 };
        ok = f && fread(hd, 4, 3, f) == 3 && hd[0] == (int32_t)E && hd[1] > 0 && hd[1] <= (int32_t)K && hd[2] == 1;
        const uint64_t n0 = ok ? (uint64_t)hd[1] * E : 0;
        if (ok) { memset(buf, 0, nel * 4); ok = fread(buf, 4, n0, f) == n0 && ds4_gpu_tensor_write(r->st.ampA[il], 0, buf, nel * 4); }
        if (ok) ok = ds4_gpu_tensor_read(r->st.ampB[il], 0, buf, nel * 4) && fread(buf, 4, n0, f) == n0 && ds4_gpu_tensor_write(r->st.ampB[il], 0, buf, nel * 4);
        if (f) fclose(f);
        if (!ok) fprintf(stderr, "ds4: --restore-b 起点 %s 读不了或形状不对(要 E %u、K ≤ %u、f32)\n", p, E, K);
        k0 = hd[1];
    }
    free(buf);
    if (ok) {   /* 起点是解在哪一份段 A 上的: 与现挂的段 A 对不上就停车(换了段 A 不许叠加) */
        char p[1200]; snprintf(p, sizeof p, "%s/base.fnv", dir);
        FILE *f = fopen(p, "r"); unsigned long long want = 0; unsigned wn = 0, hn = 0;
        if (f && fscanf(f, "%llx %u", &want, &wn) == 2) {
            const uint64_t have = (g_ds4_v41_amp_dir && g_ds4_v41_amp_dir[0]) ? ds4_gr_dir_fnv(g_ds4_v41_amp_dir, DS4_N_LAYER, &hn) : DS4_GR_FNV_SEED;
            if (have != (uint64_t)want || hn != wn) { fprintf(stderr, "ds4: ★--restore-b 起点 %s 解在另一份段 A 上, 不许叠加★\n", dir); ok = false; }
        } else fprintf(stderr, "ds4: --restore-b 起点 %s 没有 base.fnv, 核对不了段 A(放行)\n", dir);
        if (f) fclose(f);
    }
    if (ok) fprintf(stderr, "ds4: [restore-b] 起点 %s(秩 %d → 本趟 %u%s)\n", dir, k0, K, (uint32_t)k0 < K ? ", 多出的秩 A=0 起步" : "");
    return ok;
}

/* 起点目录的 fisher_Lnn.bin 读进 FA/FB(前 K0 行, 其余 0); 文件不在 → *have = false 且返回 true(由调用方决定要不要停车) */
static bool rb_fisher_load(rb_anchor *an, const pt_cfg *c, const pt_run *r, const char *dir, bool *have) {
    const uint32_t E = DS4_N_EMBD, K = r->K;
    const uint64_t nel = (uint64_t)K * E;
    float *buf = xmalloc((size_t)nel * 4);
    bool ok = true; *have = false; double fsum = 0;
    for (uint32_t il = c->layer_lo; il <= c->layer_hi && ok; il++) {
        char p[1200]; snprintf(p, sizeof p, "%s/fisher_L%02u.bin", dir, il);
        FILE *f = fopen(p, "rb");
        if (!f) { if (*have) { fprintf(stderr, "ds4: --restore-b 起点 %s 的 fisher 文件不齐(L%02u 缺)\n", dir, il); ok = false; } break; }
        *have = true;
        int32_t hd[3] = { 0, 0, 0 };
        ok = fread(hd, 4, 3, f) == 3 && hd[0] == (int32_t)E && hd[1] > 0 && hd[1] <= (int32_t)K && hd[2] >= 1;
        const uint64_t n0 = ok ? (uint64_t)hd[1] * E : 0;
        for (int ab = 0; ab < 2 && ok; ab++) {
            memset(buf, 0, nel * 4); ok = fread(buf, 4, n0, f) == n0 && ds4_gpu_tensor_write(ab ? an->FB[il] : an->FA[il], 0, buf, nel * 4);
            for (uint64_t i = 0; ok && i < n0; i++) fsum += buf[i];
        }
        fclose(f);
        if (!ok) fprintf(stderr, "ds4: --restore-b 起点 %s 读不了或形状不对(要 E %u、K ≤ %u)\n", p, E, K);
        an->nold = (uint32_t)hd[2];
    }
    free(buf);
    an->fmean = fsum / (2.0 * (c->layer_hi - c->layer_lo + 1) * (double)nel);   /* 均值按本趟的秩算(扩秩多出的秩 F = 0 也摊进去, 与核里按坐标拉的口径一致) */
    return ok;
}

bool rb_anchor_alloc(rb_anchor *an, const pt_cfg *c, const pt_run *r, const char *init, float lam) {
    memset(an, 0, sizeof *an);
    an->lam = lam;
    const uint64_t nel = (uint64_t)r->K * DS4_N_EMBD;
    bool ok = true;
    float *buf = xmalloc((size_t)nel * 4);
    for (uint32_t il = c->layer_lo; il <= c->layer_hi && ok; il++) {   /* p0 = 此刻的 A/B(rb_load_init 之后调, 就是起点); acc 从 0 攒; F 先清零 */
        an->p0A[il] = v41_alloc(nel * 4, &ok); an->p0B[il] = v41_alloc(nel * 4, &ok);
        an->accA[il] = v41_alloc(nel * 4, &ok); an->accB[il] = v41_alloc(nel * 4, &ok);
        an->FA[il] = v41_alloc(nel * 4, &ok); an->FB[il] = v41_alloc(nel * 4, &ok);
        if (ok) ok = ds4_gpu_tensor_read(r->st.ampA[il], 0, buf, nel * 4) && ds4_gpu_tensor_write(an->p0A[il], 0, buf, nel * 4) &&
                     ds4_gpu_tensor_read(r->st.ampB[il], 0, buf, nel * 4) && ds4_gpu_tensor_write(an->p0B[il], 0, buf, nel * 4) &&
                     ds4_gpu_tensor_fill_f32(an->accA[il], 0.f, nel) && ds4_gpu_tensor_fill_f32(an->accB[il], 0.f, nel) &&
                     ds4_gpu_tensor_fill_f32(an->FA[il], 0.f, nel) && ds4_gpu_tensor_fill_f32(an->FB[il], 0.f, nel);
    }
    free(buf);
    if (!ok) { fprintf(stderr, "ds4: [restore-b] 参数锚的表分配失败(每层 6 × %.1f MB)\n", nel * 4 / 1048576.0); return false; }
    bool have = false;
    if (init && init[0] && !rb_fisher_load(an, c, r, init, &have)) return false;
    an->have_f = have;
    if (lam > 0.f && !have) { fprintf(stderr, "ds4: ★--restore-b rb_anchor=%.3g 但起点 %s 没有 fisher_Lnn.bin: 先用 lr=0 跑一趟只攒 F(轮脚本 --fisher-only)★\n", (double)lam, init ? init : "(无)"); return false; }
    if (lam > 0.f && !(an->fmean > 0)) { fprintf(stderr, "ds4: ★--restore-b 起点 %s 的 F 全零, 定不了拉力★\n", init); return false; }
    if (lam > 0.f) an->coef = (float)(1.0 / (an->fmean * (double)lam));
    fprintf(stderr, "ds4: [restore-b] 参数锚: %s; 本轮攒 F 落 <ckpt>/fisher_Lnn.bin(6 张表 %.0f MB)\n",
            lam > 0.f ? "S > 0, Adam 之后按位移解耦拉回" : have ? "S = 0 只攒不拉(起点的 F 带着往下传)" : "起点没有 F, 只攒不拉", 6.0 * (c->layer_hi - c->layer_lo + 1) * nel * 4 / 1048576.0);
    if (have) fprintf(stderr, "ds4: [restore-b] 起点 F 攒过 %u 轮, 均值 %.3g, S %.3g ⇒ 每步拉回 (F_i/%.3g)/%.3g 的位移\n", an->nold, an->fmean, (double)lam, an->fmean, (double)lam);
    return true;
}

/* 每个优化步 Adam 之前: 攒数据梯度的 g²(每层 A / B 各一次; F 传 NULL = 只攒不碰 g)。pen / gn2 在这一版恒为 0(拉回挪到了 rb_anchor_pull) */
bool rb_anchor_step(rb_anchor *an, const pt_cfg *c, pt_run *r, double *pen, double *gn2) {
    const uint64_t nel = (uint64_t)r->K * DS4_N_EMBD;
    *pen = 0; *gn2 = 0;
    for (uint32_t il = c->layer_lo; il <= c->layer_hi; il++)
        if (!ds4_gpu_bwd_anchor_tensor(r->gA[il], an->accA[il], r->st.ampA[il], an->p0A[il], NULL, nel, 0.f, pen, gn2) ||
            !ds4_gpu_bwd_anchor_tensor(r->gB[il], an->accB[il], r->st.ampB[il], an->p0B[il], NULL, nel, 0.f, pen, gn2)) return false;
    an->steps++;
    return true;
}

/* 每个优化步 Adam 之后: 解耦拉回(见文件头)。pen = Σ F·Δθ²(拉回前), gn2 = 本步拉回位移²(也累进 an 的总账); S = 0 / 起点没 F 时什么都不做 */
bool rb_anchor_pull(rb_anchor *an, const pt_cfg *c, pt_run *r, double *pen, double *gn2) {
    const uint64_t nel = (uint64_t)r->K * DS4_N_EMBD;
    *pen = 0; *gn2 = 0;
    if (!(an->lam > 0.f && an->have_f)) return true;
    for (uint32_t il = c->layer_lo; il <= c->layer_hi; il++)
        if (!ds4_gpu_bwd_pull_tensor(r->st.ampA[il], an->p0A[il], an->FA[il], nel, an->coef, pen, gn2) ||
            !ds4_gpu_bwd_pull_tensor(r->st.ampB[il], an->p0B[il], an->FB[il], nel, an->coef, pen, gn2)) return false;
    an->pen += *pen; an->gn2 += *gn2;
    return true;
}

/* 落盘 <dir>/fisher_Lnn.bin: F = max(起点的 F, 本轮 acc/步数) */
bool rb_anchor_save(const rb_anchor *an, const pt_cfg *c, const char *dir) {
    if (!an->steps) return true;   /* 一步没走(没训), 没有本轮的 F 可落 */
    const uint32_t E = DS4_N_EMBD, K = c->rank;
    const uint64_t nel = (uint64_t)K * E;
    float *acc = xmalloc((size_t)nel * 4), *old = xmalloc((size_t)nel * 4);
    const float inv = 1.0f / (float)an->steps;
    bool ok = true; double snew = 0, sold = 0;
    for (uint32_t il = c->layer_lo; il <= c->layer_hi && ok; il++) {
        char p[1200]; snprintf(p, sizeof p, "%s/fisher_L%02u.bin", dir, il);
        FILE *f = fopen(p, "wb");
        const int32_t hd[3] = { (int32_t)E, (int32_t)K, (int32_t)an->nold + 1 };
        ok = f && fwrite(hd, 4, 3, f) == 3;
        for (int ab = 0; ab < 2 && ok; ab++) {
            ok = ds4_gpu_tensor_read(ab ? an->accB[il] : an->accA[il], 0, acc, nel * 4) && ds4_gpu_tensor_read(ab ? an->FB[il] : an->FA[il], 0, old, nel * 4);
            for (uint64_t i = 0; ok && i < nel; i++) { const float v = acc[i] * inv; snew += v; sold += old[i]; acc[i] = v > old[i] ? v : old[i]; }
            if (ok) ok = fwrite(acc, 4, nel, f) == nel;
        }
        if (f) fclose(f);
        if (!ok) fprintf(stderr, "ds4: [restore-b] 写不了 %s\n", p);
    }
    free(acc); free(old);
    const double den = 2.0 * (c->layer_hi - c->layer_lo + 1) * (double)nel;
    if (ok) fprintf(stderr, "ds4: [restore-b] F 落 %s/fisher_Lnn.bin(本轮均值 %.3g, 起点均值 %.3g, 攒过 %u 轮; 每步 Σ F·Δθ² 均值 %.4g, 每步拉回位移 rms %.3g)\n",
                    dir, snew / den, sold / den, an->nold + 1, an->steps ? an->pen / an->steps : 0.0, an->steps ? sqrt(an->gn2 / an->steps / den) : 0.0);
    return ok;
}

void rb_anchor_free(rb_anchor *an) {
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor **pl[] = { &an->p0A[il], &an->p0B[il], &an->accA[il], &an->accB[il], &an->FA[il], &an->FB[il] };
        for (size_t k = 0; k < sizeof pl / sizeof *pl; k++) if (*pl[k]) { ds4_gpu_tensor_free(*pl[k]); *pl[k] = NULL; }
    }
}
#endif /* !DS4_NO_GPU */
