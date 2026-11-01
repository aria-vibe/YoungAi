/* v41_amp_hook.inc.c — 反修取料的两件事: 拟合料按 .layout 窗分层 held-out, 与引擎每层 MoE 出口的取料钩子
 * (2026-10-21 从 v41_amp_run.c 原样搬出, 只为守 500 行; v41_amp_run.c 单 TU include)。 */

/* RNE 舍到 bf16 再回 f32 —— 与引擎 v41_bf16r(cuda_v41_1.inc.cu)逐位同式。取料自检要重建引擎的
 * y = round_bf16(routed + shared), 舍入口径差一位就对不上。 */
static float bf16r(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    if ((u & 0x7F800000u) == 0x7F800000u) return x;
    u += 0x7FFFu + ((u >> 16) & 1u); u &= 0xFFFF0000u;
    float y; memcpy(&y, &u, 4); return y;
}

/* <ids>.layout 的窗分层 held-out(= v41_amp_split.layout_split): 每域内窗号 k%4==3 的窗作 val, 其余拟合; 不满一窗的尾巴归拟合。
 * 为什么不前缀切: idshalf 把各域连续铺进 8192 行, 前缀切的 val 全是最后一个域而拟合段没见过它 —— 量的是跨域泛化, 假账。 */
static int layout_split(const char *ids_path, int ntok, ctx_t *c) {
    char p[4300]; snprintf(p, sizeof p, "%s.layout", ids_path);
    FILE *f = fopen(p, "r");
    if (!f) { fprintf(stderr, "★%s 缺: 择优路要 layout 分层 held-out★\n", p); return -1; }
    int win = 128, nf = 0, nv = 0;
    int *fit = malloc(sizeof(int) * (size_t)ntok), *val = malloc(sizeof(int) * (size_t)ntok);
    char line[512], desc[2048] = "";
    while (fgets(line, sizeof line, f)) {
        char dom[128]; int off, cnt;
        if (line[0] == '#' || line[0] == '\n') continue;
        if (sscanf(line, "win %d", &win) == 1) continue;
        if (sscanf(line, "%127s %d %d", dom, &off, &cnt) != 3) continue;
        const int nw = cnt / win; int dv = 0;
        for (int k = 0; k < nw; k++) {
            const int isval = (k % 4 == 3); if (isval) dv++;
            for (int r = off + k * win; r < off + (k + 1) * win; r++) { if (r >= ntok) continue; if (isval) val[nv++] = r; else fit[nf++] = r; }
        }
        for (int r = off + nw * win; r < off + cnt; r++) if (r < ntok) fit[nf++] = r;
        char d1[192]; snprintf(d1, sizeof d1, "%s%s %d+%d窗", desc[0] ? ", " : "", dom, nw - dv, dv); strncat(desc, d1, sizeof desc - strlen(desc) - 1);
    }
    fclose(f);
    if (nf + nv != ntok) { fprintf(stderr, "★layout 覆盖 %d 行 ≠ ntok %d★\n", nf + nv, ntok); return -1; }
    c->perm = malloc(sizeof(int) * (size_t)ntok);
    memcpy(c->perm, fit, sizeof(int) * (size_t)nf); memcpy(c->perm + nf, val, sizeof(int) * (size_t)nv);
    c->nfit = nf; c->n = ntok;
    free(fit); free(val);
    printf("[held-out·分层] 窗宽 %d: 拟合 %d / val %d 行 (%s)\n", win, nf, nv, desc);
    return 0;
}

/* 取料钩子: 只认本遍的层, 原序收进 rx/ry/rsel/rrw, 收完本块即让引擎停下(返回 1) */
static int hook(void *ud, int il, int pos0, int n, int D, int n_used, float clamp, const float *x, const float *y, const int *sel, const float *rw, const float *alpha, const float *ye, const float *ysh) {
    ctx_t *c = ud;
    if (il != c->layer) return 0;
    if (D != c->D || n_used > MAXU || pos0 < 0 || pos0 + n > c->n) { fprintf(stderr, "★钩子 pos0=%d n=%d D=%d n_used=%d 与拟合 %d/%d 不符★\n", pos0, n, D, n_used, c->n, c->D); return -1; }
    c->n_used = n_used; c->clamp = clamp;
    /* rx/ry 为空 = 这条路不要它们(第七版只用 ye/rw/sel/alpha, 两份 [n][D] f32 在 35k 行上是 1.4 GB 白占) */
    if (c->rx) memcpy(c->rx + (size_t)pos0 * D, x, (size_t)n * D * 4);
    if (c->ry) memcpy(c->ry + (size_t)pos0 * D, y, (size_t)n * D * 4);
    memcpy(c->rsel + (size_t)pos0 * n_used, sel, (size_t)n * n_used * 4); memcpy(c->rrw + (size_t)pos0 * n_used, rw, (size_t)n * n_used * 4);
    memcpy(c->ralpha + (size_t)pos0, alpha, (size_t)n * 4);
    if (c->want_ye && !(c->lowrank_k && !ye)) {   /* 低秩形态只要 x(上面已存), 解码 GEMV 路没有逐专家输出也放行(10-01: 按行拆的事实料一行 7 token) */
        /* 取不到就硬停: 逐专家输出只有 prefill GEMM 路物化, 拿不到说明走了解码路或形状对不上 ——
         * 这时候继续跑会解出一个没有料支撑的东西, 比报错难查得多。 */
        if (!ye || !ysh) { fprintf(stderr, "★L%02d 拿不到逐专家 down 输出/shared(ye=%p ysh=%p): 块 n=%d 可能走了解码 gemv 路★\n", il, (const void *)ye, (const void *)ysh, n); return -1; }
        if (c->rye_map) {
            /* 第七版取料槽: 只搬 rye_map 里有槽号的行(第一遍榜单表定的行集), f32 原值。 */
            if (!c->rye) {
                c->rye_nu = n_used;
                c->rye = malloc((size_t)c->rye_slots * n_used * D * 4);
                if (!c->rye) { fprintf(stderr, "★取料槽 %d × %d × %d f32(%.2f GB)分配失败★\n", c->rye_slots, n_used, D, (double)c->rye_slots * n_used * D * 4 / 1e9); return -1; }
            }
            if (c->rye_nu != n_used) { fprintf(stderr, "★取料槽 n_used 变了: %d → %d★\n", c->rye_nu, n_used); return -1; }
            for (int i = 0; i < n; i++) {
                const int s = c->rye_map[pos0 + i];
                if (s < 0) continue;
                if (s >= c->rye_slots) { fprintf(stderr, "★取料槽号 %d ≥ 槽数 %d★\n", s, c->rye_slots); return -1; }
                memcpy(c->rye + (size_t)s * n_used * D, ye + (size_t)i * n_used * D, (size_t)n_used * D * 4);
            }
        } else {
            memcpy(c->rye + (size_t)pos0 * n_used * D, ye, (size_t)n * n_used * D * 4);
            memcpy(c->rysh + (size_t)pos0 * D, ysh, (size_t)n * D * 4);
        }
    }
    c->got += n;
    /* ★取完本层要不要继续跑到出口★
     * 蒸馏靶要同一趟的学生 logits, 必须跑完。后训练也要(榜单 top-K 与 rms 就在出口那一步算),
     * ★两样本来就在同一趟里★ —— 第一版让它半路停车、另跑一趟专门打分, 于是同一批 token 前向了
     * 两遍, 8 条样本白烧 22 分钟, 外加 8 次独立进程各加载一遍模型(107s×8 ≈ 14 分钟)。
     * 当时停车的理由是"跑到出口会写一份全词表 logits"(517 KB/位置, 统一内存上写它=掏 GPU 内存),
     * 但现在有 --score-no-logits + --score-topk/--score-nll/--score-rms 三个小出口, 出口那一段
     * 只落几十 MB。所以只在【表已经在盘上】(c->reuse_tab)时才提前停车。 */
    return (c->klt && !c->reuse_tab) ? 0 : 1;
}
