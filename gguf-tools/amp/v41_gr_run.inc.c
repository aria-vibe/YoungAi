/* v41_gr_run.inc.c — 权重侧逐专家反修的驱动侧编排(2026-09-13)。v41_amp_run.c 单 TU include,
 * 拆出来只为守单文件 ≤500 行; 数值全在 v41_gr_solve.cu, 这里只做缓冲、λ 择优、落盘、manifest。
 *
 * 产物 gr_Lnn.bin: 头 <i32 n_expert><i32 D><i32 类型 43=fp4x32>, 接每个专家每个输出通道的
 * 【增益缩放因子】的偏离量 s−1(4.25 bpw, 17 B/32 个数, 1.04 MB/层; f32 是 7.86, f16 是 3.93)。
 * 引擎加载时 +1 还原, 乘进 VQ 载荷里那一行的 g_r。
 * ★解算器直接在 fp4 格点上解, 码字是 GPU 产出的, 这里一个字节都不转换★(2026-09-14, 用户令
 * "反修输出直接就是 fp4, 而不是转换数据")。为什么必须这样而不是"解完再压": 4 bit 的相对精度就是
 * 10% 量级(E2M1 八个幅值格点), 解完再压等于白扔 11% 的修正量且无人接管; 格点直解让每个专家的
 * 量化误差当场进残差, 由后面的专家在同一轮 Gauss-Seidel 里补偿。机理与判据见 fp4.md, 实现见
 * v41_gr_solve.cu 的 gr_fp4_quant_kernel。落盘前拿 src/common 的唯一解码器回读逐位自证。
 * ★存缩放因子而不是新增益★: 盘上原 g 一个字节不动, 插件不挂就是裸底座; 且缩放因子恒在 1 附近,
 * 万一某层解崩了, 看一眼分布就能发现(新增益的绝对值没有这个参照)。 */

/* λ 网格: 岭把解锚在"不改"上, λ 无量纲(尺度已用本通道能量归一)。大 λ ⇒ 解被压回 1。
 * 与低秩路的 SEL_LAMS 不同数量级是应该的 —— 那边是 XᵀX 的正则, 这边是一维带能量归一的岭。
 *
 * ★2026-09-19/20 往下加一格 0.01(用户令"r 正则范围新增支持 0.01、0.1 这两个", 0.1 原网格已有)★
 * 依据不是直觉, 是现役产物自己的账: `gr-fin-40-fp4/manifest.txt` 37 个挂上的层里 ——
 *     λ=0.1(旧下界) 9 层 | λ=1 26 层 | λ=10 2 层 | λ=100 **0 层** | λ=1000 **0 层**
 * 撞下界的那 9 层(L00~L05 + L13 + L19 + L39)恰好是最肥的一批(L00 层内 val +26.7%)。
 * ★选到边界 = 网格在那一侧是错的★: 这些层的最优 λ 很可能在 0.1 以下, 岭把它们的修正量压小了,
 * 而旧网格量不到这件事 —— 它只能报"0.1 最好", 报不了"还想更小"。
 * 网格按用户点名的值走: 只加 0.01, 上界 1000 原样留着(没被选中不等于要砍, 用户没令砍)。
 *
 * ★代价要认★: 每层"N 选 1 取最大"是有选择偏差的(纯噪声层也会挑出一个正的 val), N 从 5 变 6 会让
 * 这个偏差略增, 也就是 GR_GATE(0.5%) 的显著性略降。★不因此动闸★ —— 那个数 09-15 用端到端尺
 * 扫过(见下), 凭直觉调它是拿判决尺的噪声换数字。要判新网格值不值, 看的是端到端主尺, 不是层内 val。
 * ★下一轮的检查点★: 跑完看 manifest 的 λ 分布, 若又有一批层撞新下界 0.01, 再往下扩(0.001)要用户点头。 */
#define GR_NL 6
static const float GR_LAMS[GR_NL] = {0.01f, 0.1f, 1.0f, 10.f, 100.f, 1000.f};
#if GR_NL > GR_MAXL
#error "GR_NL 超过 v41_amp_run.c 的 GR_MAXL: 每个 λ 的解都要留主机副本给端到端择优"
#endif
#define GR_ITER 3            /* Gauss-Seidel 轮数: 专家间耦合只来自同 token 的 6 个 pick, 很稀疏 */
/* ★层内闸(0.5 百分点)与"λ 取 val 最大"都已退役★(2026-10-21): 赢家与挂不挂由端到端尺定(v41_e2e_select.inc.c) ——
 * 层内 val 只用来排候选次序(val 最高的 E2E_NCAND 个进端到端), val ≤ 0 的解不进(本层 held-out 上已经变坏)。
 * 为什么: "N 个 λ 挑 val 最大"对无信号层也会交出一个正数(选择偏差), 而层内 val 到出口要穿过后面所有层 ——
 * 08-31 / 09-13 / 10-20 三次撞到"层内赢、端到端不赢"。09-15 用金融主尺扫的 0 / 0.5 / 1.0 三点闸曲线留在 fable5。 */

/* 量化往返: hs(缩放因子, 锚 1) → fp4x32 码字, 并把 hs 原地换成【解回来的那一份】。
 * ★这是后训练(第三件)专用的路★ —— 它解的是"在上一版 ② 之上再乘多少", 落盘的是乘积, 所以只能
 * 先解完再压。权重侧反修(gr 路)★不走这里★: 它的解算器直接在 fp4 格点上解, 码字由 GPU 产出,
 * 见 v41_gr_solve.cu 的"格点直解"。留一泛化尺(尺 L)每折都要往返但不落盘, 所以往返与写盘是两个函数。 */
static uint8_t *gr_fp4_roundtrip(float *hs, size_t nsD) {
    const size_t nblk = nsD / 32u;
    float *dev = malloc(nsD * 4);
    uint8_t *pk = malloc(nblk * 17u);
    if (!dev || !pk) { free(dev); free(pk); fprintf(stderr, "★fp4 缓冲分配失败★\n"); return NULL; }
    for (size_t t = 0; t < nsD; t++) dev[t] = hs[t] - 1.0f;
    ds4_quant_fp4x32(dev, nblk, pk);
    ds4_deq_fp4x32(pk, nblk, dev);
    for (size_t t = 0; t < nsD; t++) hs[t] = 1.0f + dev[t];
    free(dev);
    return pk;
}

/* ★产物落盘: 写的就是【解算器产出的那份码字】, 中间没有任何转换★(2026-09-14)。
 * 头 <i32 n_expert><i32 D><i32 43=fp4x32>, 接 [n_expert][D/32] 个 17 字节块(存的是 s−1)。
 * 写 .part 再改名, 半成品永不被下一遍挂上。
 *
 * ★落盘前必须解码自证★: 码字是 GPU 上编出来的(v41_gr_solve.cu 的 gr_fp4_quant_kernel), 解码用的
 * 是 src/common 的唯一解码器 ds4_deq_fp4x32。两边逐位相等才落盘 —— 不等说明 GPU 编码与全仓格式
 * 基元漂开了(舍入、饱和、块缩放择优任一处), 这种漂移不会让程序报错, 只会让盘上的插件与解算器
 * 以为的那一份不是同一个东西, 判决尺量出来就是一笔无从追查的"莫名其妙掉了"。 */
static int gr_write_bin(const float *hs, const uint8_t *pk, int ne, int D, const char *dir, int il) {
    const size_t nsD = (size_t)ne * D;
    if (nsD % 32u) { fprintf(stderr, "  [L%02d] ★%zu 不是 32 的整数倍★\n", il, nsD); return -1; }
    const size_t nblk = nsD / 32u, nby = nblk * 17u;
    float *back = malloc(nsD * 4);
    if (!back) { fprintf(stderr, "  [L%02d] ★自证缓冲分配失败★\n", il); return -1; }
    ds4_deq_fp4x32(pk, nblk, back);
    for (size_t t = 0; t < nsD; t++) {
        const float want = 1.0f + back[t];
        if (want != hs[t]) {
            fprintf(stderr, "  [L%02d] ★码字解回来与解算态对不上(第 %zu 个: 盘 %.9g vs 解 %.9g), 停车★\n",
                    il, t, want, hs[t]);
            free(back); return -1;
        }
    }
    free(back);
    char p[4300]; snprintf(p, sizeof p, "%s/gr_L%02d.bin.part", dir, il);
    FILE *f = fopen(p, "wb");
    if (!f) { fprintf(stderr, "  [L%02d] ★写不了 %s★\n", il, p); return -1; }
    const int32_t hd[3] = { ne, D, (int32_t)DS4_GGT_FP4X32 };
    const int okw = fwrite(hd, 4, 3, f) == 3 && fwrite(pk, 1, nby, f) == nby;
    if (fclose(f) || !okw) { fprintf(stderr, "  [L%02d] ★gr_L%02d.bin 写盘截断★\n", il, il); remove(p); return -1; }
    char q[4300]; snprintf(q, sizeof q, "%s/gr_L%02d.bin", dir, il);
    if (rename(p, q)) { fprintf(stderr, "  [L%02d] ★改名失败★\n", il); return -1; }
    return 0;
}

/* 后训练(第三件)那条路仍是"解完再压", 它落盘的是与上一版相乘之后的 s ⇒ 先往返再走这个写盘函数。 */
static int gr_write_bin_roundtrip(float *hs, int ne, int D, const char *dir, int il) {
    uint8_t *pk = gr_fp4_roundtrip(hs, (size_t)ne * D);
    if (!pk) return -1;
    const int rc = gr_write_bin(hs, pk, ne, D, dir, il);
    free(pk);
    return rc;
}

/* ★取料/解算工作区按层开、端到端前向之前放★(2026-10-21): rx/ry/hx/hy(各 n×D f32)、rye/hye(各 n×6×D f32, 1.9 GB)、rysh/hysh、
 * 设备 dye/dysh/drw/dsel/dX/dYq/dYfp/dYtmp、val 副本 hvf/hvt —— 15360 行合计 ~12 GB, 全前向那几趟一个都不用。统一内存机器上它们
 * 占着就是页缓存少这么多: 权重缓存封顶 88 GB 之外的 9 层(25 GB)靠页缓存才不每块重读盘(探针实测: 起点态 66 s, 解完一层后 179 s)。
 * 放掉 + FP 教师的设备缓冲也放掉(v41_fp_moe_release), 下一层取料前再开。rrw/rsel/ralpha(几十 MB)与 λ 候选副本不动。 */
static void gr_work_free(ctx_t *c) {
    free(c->rx); free(c->ry); free(c->hx); free(c->hy); free(c->hrw); free(c->hsel); free(c->halpha); free(c->hvf); free(c->hvt);
    free(c->rye); free(c->rysh); free(c->hye); free(c->hysh);
    c->rx = c->ry = c->hx = c->hy = c->hrw = c->halpha = c->hvf = c->hvt = c->rye = c->rysh = c->hye = c->hysh = NULL; c->hsel = NULL;
    cudaFree(c->dye); cudaFree(c->dysh); cudaFree(c->drw); cudaFree(c->dsel); cudaFree(c->dX); cudaFree(c->dYq); cudaFree(c->dYfp); cudaFree(c->dYtmp);
    c->dye = c->dysh = c->drw = c->dX = c->dYq = c->dYfp = c->dYtmp = NULL; c->dsel = NULL;
    v41_fp_moe_release();
}
static int gr_work_alloc(ctx_t *c) {
    const int ntok = c->n;
    const size_t nD = (size_t)ntok * c->D, nU = (size_t)ntok * MAXU, nyD = nU * (size_t)c->D;
    c->rx = malloc(nD * 4); c->ry = malloc(nD * 4); c->hx = malloc(nD * 4); c->hy = malloc(nD * 4);
    c->hrw = malloc(nU * 4); c->hsel = malloc(nU * 4); c->halpha = malloc((size_t)ntok * 4);
    c->hvf = malloc((size_t)(ntok - c->nfit + 1) * c->D * 4); c->hvt = malloc((size_t)(ntok - c->nfit + 1) * c->D * 4);
    c->rye = malloc(nyD * 4); c->rysh = malloc(nD * 4); c->hye = malloc(nyD * 4); c->hysh = malloc(nD * 4);
    if (!c->rx || !c->ry || !c->hx || !c->hy || !c->hrw || !c->hsel || !c->halpha || !c->hvf || !c->hvt || !c->rye || !c->rysh || !c->hye || !c->hysh) {
        fprintf(stderr, "★取料工作区(主机 %.1f GB)分配失败★\n", (double)(nD * 16 + nyD * 8) / 1e9); return -1; }
    if (cudaMalloc((void **)&c->dye, nyD * 4) || cudaMalloc((void **)&c->dysh, nD * 4) || cudaMalloc((void **)&c->drw, nU * 4) || cudaMalloc((void **)&c->dsel, nU * 4) ||
        cudaMalloc((void **)&c->dX, nD * 4) || cudaMalloc((void **)&c->dYq, nD * 4) || cudaMalloc((void **)&c->dYfp, nD * 4) || cudaMalloc((void **)&c->dYtmp, nD * 4)) {
        fprintf(stderr, "★取料工作区(设备 %.1f GB)分配失败★\n", (double)(nyD * 4 + nD * 20) / 1e9); return -1; }
    return 0;
}

/* 在当前取料上解 λ 网格, 全部解留进 sol[b](b=0 无本层 Δb 的态 / 1 挂着 Δb 的态)。返回 <0 失败。
 * ★这里报的 train/val 就是落地态★(2026-09-14 改格点直解之后): 解算器在 fp4 格点上解, 最终残差
 * 是拿格点值算的。以前是"按解算态 val 选 λ, 再把赢家压成 fp4", 两件事口径不一样 —— 大 λ 的解
 * 幅度小、相对量化误差更大, 两种口径可能选出不同的 λ, 而部署里跑的永远是压过之后的那一份。 */
static int gr_solve_branch(ctx_t *c, int il, int b) {
    const int n = c->nsel > 0 ? c->nsel : c->n, D = c->D, nu = c->n_used, ne = c->n_expert;   /* 行数用 nsel(--fit-rows/--val-rows 的子集) */
    const size_t nsD = (size_t)ne * D, nby = nsD / 32u * 17u;
    gr_sols *S = &c->sol[b];
    double t0, t1;
    if (prep_layer(c, il, &t0, &t1)) return -1;   /* 取料自检 → 置换 → 上传 → FP 靶 */
    for (int l = 0; l < GR_NL; l++) {
        if (v41_gr_solve_layer_gpu(c->dye, c->drw, c->dsel, c->dYfp, c->dysh,
                                   n, c->nfit, nu, D, ne, GR_LAMS[l], GR_ITER, c->dgr, c->dgrpk, &S->tr[l], &S->va[l])) {
            fprintf(stderr, "  [L%02d] ★增益解算失败(λ=%.3g), 停车★\n", il, GR_LAMS[l]);
            return -1;
        }
        printf("  [L%02d] %s λ=%-5.3g train %+6.2f%%  val %+6.2f%% (fp4 格点上的落地读数)\n", il, b ? "挂Δb" : "无Δb", GR_LAMS[l], S->tr[l], S->va[l]);
        fflush(stdout);
        /* 每个 λ 的 s 与【它那一份码字】一起留下来 —— 两者必须是同一次解算的产物, 否则落盘自证会炸 */
        CK(cudaMemcpy(S->hgr[l], c->dgr, nsD * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(S->pk[l], c->dgrpk, nby, cudaMemcpyDeviceToHost));
    }
    S->n = GR_NL;
    const double t2 = now_s(); c->t_solve += t2 - t1;
    return 0;
}

/* 一个分支里层内 val 前 E2E_NCAND 个解的下标(val ≤ 0 不进), 返回个数 */
static int gr_top(const gr_sols *S, int *ord) {
    int o[GR_MAXL], no = 0;
    for (int l = 0; l < S->n; l++) if (S->va[l] > 0.f) { int k = no++; while (k > 0 && S->va[o[k - 1]] < S->va[l]) { o[k] = o[k - 1]; k--; } o[k] = l; }
    const int ng = no < E2E_NCAND ? no : E2E_NCAND;
    for (int k = 0; k < ng; k++) ord[k] = o[k];
    return ng;
}

/* 本层增益的候选端到端裁; 赢家定本层盘上留不留增益。Δb 的去留不在这里裁(见下), 返回 <0 失败, 0 本层不挂增益, 1 挂了增益。
 * ★Δb 只由层内闸定, 端到端不撤它★(2026-10-21 第二趟 40 层后的两次诊断, 账在 fable5):
 *   旧做法每层把"带 Δb"与"不带 Δb"两支候选放一起裁, 带 Δb 的那支每层只输 0.02~0.78% KLD, 18 层被撤 —— 按逐层读数全挂该亏 ~5%,
 *   实测却是赚: ① 10-18 旧侧车拿掉它那 29 层 Δb, 判决份 KLD 0.31794 → 0.32500(落到第二趟的 0.32457); ② 第二趟补上旧侧车有它没有的
 *   19 层 Δb(增益是在没 Δb 的料上解的, 还失配着), 判决份 KLD 0.32457 → 0.31666, 五项四项赢旧侧车。Δb 把路由推向 FP 的选专家集合,
 *   单层推、下游还是量化路由时互相不搭; 一串层一起推才走到 FP 那条轨迹上 —— 逐层贪心看不见这种"合起来才赚"。
 *   所以: 层内闸过了(rb_seq_solve 返回 1, 已落盘)就留着; 本层的比较基线换成"只挂 Δb、不挂增益"(多跑一趟前向), 只在挂着 Δb 重取的料上
 *   解的增益里选; 层内闸没过的层与以前一样只有无 Δb 一支, 基线是上一层赢家的读数(不另跑)。 */
static int gr_e2e_select(ctx_t *c, int il, int has_rb) {
    const int D = c->D, ne = c->n_expert; const size_t nsD = (size_t)ne * D;
    v41_st_release_all(&c->S);   /* 全前向之前把 FP 权重分片全卸掉(页缓存也丢): 让给流式层的权重页 */
    gr_work_free(c);             /* 同理: 取料/解算工作区与 FP 教师设备缓冲 ~16 GB 也还回去, 下一层取料前再开 */
    char gp[4300]; snprintf(gp, sizeof gp, "%s/gr_L%02d.bin", c->out_dir, il);
    const int b = has_rb;        /* 只有一支: 有 Δb 就是挂着它重取的那支 */
    const gr_sols *S = &c->sol[b];
    e2e_cand cs[E2E_NCAND]; int nc = 0, cl[E2E_NCAND];   /* 候选 → λ 下标 */
    memset(cs, 0, sizeof cs);   /* m.hit 从 NULL 起, e2e_run 才分配 */
    const double t2 = now_s();
    if (has_rb) {   /* 基线 = 只挂本层 Δb: 盘上此刻有 rb_Lnn 没有 gr_Lnn */
        e2e_m m = {0};
        if (e2e_run(c, "只挂Δb", &m)) return -1;
        fprintf(c->mf, "L%02d e2e 上层赢家 KLD %.5f RMS %.4f Same %.2f → 只挂Δb KLD %.5f RMS %.4f Same %.2f(Δb 层内闸过, 留着; 下面的 base = 只挂Δb)\n",
                il, c->e2e_base.kld, c->e2e_base.rms, c->e2e_base.same, m.kld, m.rms, m.same);
        free(c->e2e_base.hit); c->e2e_base = m;
    }
    int ord[E2E_NCAND]; const int ng = gr_top(S, ord);
    for (int k = 0; k < ng; k++) {
        const int l = ord[k];
        if (gr_write_bin(S->hgr[l], S->pk[l], ne, D, c->out_dir, il)) return -1;
        snprintf(cs[nc].name, sizeof cs[nc].name, "%sλ=%g", b ? "rb+" : "", (double)GR_LAMS[l]);
        if (e2e_run(c, cs[nc].name, &cs[nc].m)) return -1;
        cl[nc] = l; nc++;
    }
    if (!nc) printf("  [L%02d] %s层内没有正 val 的增益解\n", il, b ? "挂Δb " : "无Δb ");
    const int w = nc ? e2e_pick(c, il, cs, nc) : -1;
    const double t3 = now_s();
    for (int k = 0; k < nc; k++) if (k != w) free(cs[k].m.hit);   /* 输家的逐行命中表不要了; 赢家的随读数一起成为下一层的基线 */
    if (w < 0) {   /* 没候选, 或基线赢: 本层不留增益(Δb 有就留着, 基线读数已是它的) */
        remove(gp);
        c->skip_layers++;
        const float *va = S->va, *tr = S->tr; int lb = 0; for (int l = 1; l < S->n; l++) if (va[l] > va[lb]) lb = l;
        fprintf(c->mf, "L%02d skip gr - - %+.2f %+.2f - (%s%s)\n", il, tr[lb], va[lb], nc ? "端到端基线赢" : "层内无正 val 候选", has_rb ? ", 只挂 Δb" : ""); fflush(c->mf);
        printf("  [L%02d] 不挂增益: %s%s  (端到端 %.1fs)\n", il, nc ? "端到端基线赢" : "层内无正 val 候选", has_rb ? ", 只挂 Δb" : "", t3 - t2);
        return 0;
    }
    free(c->e2e_base.hit); c->e2e_base = cs[w].m;   /* 赢家的读数(含逐行命中表) = 下一层的基线 */
    const int l = cl[w];
    const float *hs = S->hgr[l], lam = GR_LAMS[l], tr = S->tr[l], va = S->va[l];
    /* 盘上此刻是最后一个候选的增益文件: 不是赢家的就重写(写的仍是解算器那份码字, 落盘前解码自证) */
    if (w != nc - 1 && gr_write_bin(S->hgr[l], S->pk[l], ne, D, c->out_dir, il)) return -1;
    /* 缩放因子的分布也报出来 —— 全是 1 说明岭把解压死了(那时 val 也该接近 0, 两者对不上就是 bug) */
    double smin = 1e30, smax = -1e30, sabs = 0;
    for (size_t t = 0; t < nsD; t++) { const double v = hs[t]; if (v < smin) smin = v; if (v > smax) smax = v; sabs += fabs(v - 1.0); }
    c->ok_layers++; c->val_sum += va; c->dz_sum += sabs / (double)nsD;
    fprintf(c->mf, "L%02d gr %.3g %.2f %+.2f %+.2f [%.4f,%.4f] %s\n", il, lam, sabs / (double)nsD, tr, va, smin, smax, b ? "带Δb" : "");
    fflush(c->mf);
    printf("  [L%02d] ★挂上★ %s λ=%.3g train %+.2f%% val %+.2f%% | 缩放 |s−1| 均 %.4f 范围 [%.4f, %.4f] | e2e KLD %.5f RMS %.4f%% Same %.2f%%  (端到端 %.1fs)\n",
           il, b ? "带Δb" : "无Δb", lam, tr, va, sabs / (double)nsD, smin, smax, cs[w].m.kld, cs[w].m.rms, cs[w].m.same, t3 - t2);
    return 1;
}
