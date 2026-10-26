/* anchor_metrics_pair.inc.c — 两份 --row-out 逐行文件的配对差与按窗标准误差(2026-10-21)。
 * 被 anchor_metrics.c 单 TU include(拆文件只为守 500 行), 用法:
 *     anchor_metrics --pair A.rows B.rows [--win 128]
 * 出 B − A 的 ΔKLD / ΔΣmin / ΔSame top, 各带标准误差和 σ 倍数。
 *
 * 为什么按窗分块, 不按行: 同一个 128 token 窗里相邻位置的误差强相关(同一段上下文、同一处漂移),
 * 按 3840 或 15360 个独立行算 SE 会小好几倍, 0.3σ 的差能被说成 3σ。判决份是 120 个 128 窗拼的,
 * 窗才是独立的抽样单位。块号 = 行号 / win, 所以行号必须是锚行号(--row-out 写的就是)。
 * 为什么是配对差: 两个学生在同一批行上打分, 难的行两边都难; 先逐行相减再求 SE, 把"行有多难"这份
 * 方差消掉, 剩下的才是两者真实的差 —— 10-21 两份侧车直接比 1σ ≈ 0.33%, 只差一类张量的两个学生会小得多。
 * 出错会怎样: 两份行号对不上(不同判决料 / 不同 --rows)直接拒, 不会拿错位的行相减出一个"看着合理"的数。 */
typedef struct { int n, cap; int *row; double *kld, *smin; int *same; } pair_rows;

static int pair_read(const char *path, pair_rows *r) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "--pair %s 打不开\n", path); return 2; }
    memset(r, 0, sizeof *r);
    char line[256];
    while (fgets(line, sizeof line, f)) {
        int row, same; double kld, smin;
        if (sscanf(line, "%d %lf %lf %d", &row, &kld, &smin, &same) != 4) continue;
        if (r->n == r->cap) {
            r->cap = r->cap ? r->cap * 2 : 16384;
            r->row = realloc(r->row, (size_t)r->cap * sizeof(int)); r->same = realloc(r->same, (size_t)r->cap * sizeof(int));
            r->kld = realloc(r->kld, (size_t)r->cap * sizeof(double)); r->smin = realloc(r->smin, (size_t)r->cap * sizeof(double));
            if (!r->row || !r->same || !r->kld || !r->smin) { fprintf(stderr, "--pair 内存\n"); fclose(f); return 2; }
        }
        r->row[r->n] = row; r->kld[r->n] = kld; r->smin[r->n] = smin; r->same[r->n] = same; r->n++;
    }
    fclose(f);
    if (!r->n) { fprintf(stderr, "--pair %s: 一行都没读到(不是 --row-out 的产物?)\n", path); return 2; }
    return 0;
}

/* 比值估计的按块 SE: 块大小不等时也成立(判决行最后一窗可能不满)。v[] = 逐行的 B−A。 */
static void pair_stat(const pair_rows *a, const double *v, int win, double *mean, double *se, int *nb_out) {
    int nb = 0; for (int i = 0; i < a->n; i++) if (a->row[i] / win + 1 > nb) nb = a->row[i] / win + 1;
    double *bs = calloc((size_t)nb, sizeof(double)); int *bn = calloc((size_t)nb, sizeof(int));
    double tot = 0; for (int i = 0; i < a->n; i++) { bs[a->row[i] / win] += v[i]; bn[a->row[i] / win]++; tot += v[i]; }
    const double m = tot / a->n; double ss = 0; int used = 0;
    for (int b = 0; b < nb; b++) if (bn[b]) { const double w = (double)bn[b] / a->n, d = bs[b] / bn[b] - m; ss += w * w * d * d; used++; }
    *mean = m; *se = used > 1 ? sqrt(ss * used / (used - 1)) : NAN; *nb_out = used;
    free(bs); free(bn);
}

static int pair_report(const char *pa, const char *pb, int win) {
    pair_rows a, b;
    if (pair_read(pa, &a) || pair_read(pb, &b)) return 2;
    if (a.n != b.n) { fprintf(stderr, "--pair 行数不同 %d vs %d: 不是同一份判决料\n", a.n, b.n); return 2; }
    for (int i = 0; i < a.n; i++) if (a.row[i] != b.row[i]) { fprintf(stderr, "--pair 第 %d 行行号 %d vs %d 对不上\n", i, a.row[i], b.row[i]); return 2; }
    double *v = malloc((size_t)a.n * sizeof(double)), ka = 0, sa = 0, ta = 0;
    for (int i = 0; i < a.n; i++) { ka += a.kld[i]; sa += a.smin[i]; ta += a.same[i]; }
    ka /= a.n; sa /= a.n; ta /= a.n;
    printf("配对 B − A: A=%s\n           B=%s\n  n=%d 行", pa, pb, a.n);
    const char *nm[3] = {"ΔKLD", "ΔΣmin", "ΔSame top"};
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < a.n; i++) v[i] = k == 0 ? b.kld[i] - a.kld[i] : k == 1 ? b.smin[i] - a.smin[i] : (double)(b.same[i] - a.same[i]);
        double m, se; int nb; pair_stat(&a, v, win, &m, &se, &nb);
        if (k == 0) printf(", %d 块 × %d 行\n", nb, win);
        if (k == 0) printf("  %-10s %+.5f (相对 A %+.2f%%)  SE %.5f  ⇒ %+.1fσ   (A 均值 %.5f)\n", nm[k], m, 100 * m / ka, se, m / se, ka);
        if (k == 1) printf("  %-10s %+.5f                 SE %.5f  ⇒ %+.1fσ   (A 均值 %.4f)\n", nm[k], m, se, m / se, sa);
        if (k == 2) printf("  %-10s %+.3f pp                SE %.3f pp ⇒ %+.1fσ   (A %.2f%%)\n", nm[k], 100 * m, 100 * se, m / se, 100 * ta);
    }
    free(v);
    return 0;
}
