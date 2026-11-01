/* v41_e2e_select.inc.c — 反修逐层候选的【端到端】择优(2026-10-21, 用户: "行内指标好但最后指标不行; 指标换成标准的 PPL/KLD/Same top 选最优";
 * 10-24 用户令"反修的过程指标换成 Same top / KLD / RMS": 三把尺就是判决器五指标里的三把, NLL/PPL 退出过程尺)。
 * v41_amp_run.c 单 TU include, 拆出只为守 500 行。
 *
 * 【做什么】每层解完 λ 网格之后, 赢家不再按层内 held-out 能量挽回率定, 而是: 把候选(层内 val 最高的 E2E_NCAND 个增益解)逐个挂上,
 * 整条引擎前向跑完拟合料, 在 val 行(与层内 held-out 同一批窗, 拟合从没见过)上对 FP 教师锚算判决器那三把尺 Same top / KLD / RMS Δp(top-32 联合窗),
 * 与"本层不挂增益"(本层 Δb 层内闸过了 = 只挂 Δb, 多跑一趟; 否则 = 上一层赢家的读数, 不另跑)一起裁: 三项全优者赢; 没有全优者, Same top
 * 比基线退得超过噪声 2σ 的候选先否掉, 剩下的 KLD 最低者赢, KLD 差在趟间噪声内比 RMS Δp(10-21 用户: "没有旧车好, 说明选择指标有问题"); 基线赢就本层不挂增益。
 * Δb 的去留不在这里裁(逐层裁会错杀, 账在 v41_gr_run.inc.c gr_e2e_select)。
 * 【为什么】层内 held-out 是本层 MoE 出口的能量域读数, 它与出口分布的关系要穿过后面所有层; 08-31 / 09-13 / 10-20 三次都撞到
 * "层内正收益、端到端小或负", 10-20 的思考模板趟层内 val 更高、判决份反退。出口上直接量就没有这层换算。代价是每层多跑
 * 候选数趟全前向(拟合料 15360 行)。
 * 【尺】逐位式子与 bench/anchor_metrics 同一份(bench/anchor_row.c), 教师锚 = v41_judge.sh 落的 teacher_<料>_n<N>[_<底座>].bin
 * (<i32 S><i32 V><f32[S][V]>, 拟合料自己的锚, 不是判决份的); 行按需 pread, 读完即 fadvise 丢页 —— 7.9 GB 的锚整份进页缓存会把
 * 流式层的权重页挤出去, 每块多读几 GB 盘(10-18 解算日志: 后 9 层每块 3~4 s, 判决器同路只要 0.3 s)。
 * 【不是判决份】这里的 val 行是拟合料自己的 held-out 窗; 判决份 judge.ids 仍只在 v41_judge.sh 的五指标里用, 两份语料不串岗。 */
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include "../bench/anchor_row.h"

#define E2E_NCAND 2      /* 每层进端到端的增益候选数: 层内 val 最高的几个(val ≤ 0 的不进 —— 本层 held-out 上已经变坏的解没必要再跑一趟)。
                          * 10-20 探针(3 个): 两层里第三名从没赢过, 而每个候选就是一趟 35~40 s 的全前向; 用户嫌慢, 减到 2。 */
/* ★为什么 KLD 当主裁、Same top 只当否决闸(10-21 第一趟 40 层的账, 在 fable5)★: Same top 是 3840 个 0/1 的计数, 两个候选只在"一个中一个不中"
 * 的行上有差, 这种不一致行有 d 个, 差 D = 中多的行数 − 中少的行数, 没有真差别时 D 的标准差 ≈ √d ≈ 8~13 行(0.2~0.3 pp); 而单层的真实效应
 * 只有 1~15 行 —— 信噪比 0.1~1.5, 拿它当主裁等于掷硬币, 且平手基线赢 ⇒ 15 层不挂、14 层挂错, 判决份五项全输 10-18 旧侧车(旧侧车 40 层全挂,
 * 在同一批 val 行上 KLD 也低 5.8%)。KLD 是 3840 行的均值, 单层效应 0.1~2.3%, 趟间噪声 ~1e-4(0.03%), 信噪比 4~80, 分得清。
 * 所以: 主裁 = KLD; Same top 只在候选比基线退超过 E2E_VETO_SIGMA·√d 时否决(防 KLD 降了 top-1 坏了); "一样"的门槛全从数据量, 不写死:
 * Same top 用配对噪声 √d(地板 = 趟间噪声行数), KLD 用趟间噪声(e2e_noise_check 起点态连跑两趟的差, 10-20 探针同进程内 0)。
 * 否决用 2σ 不用 1σ: 1σ 下没有真差别时单侧 16% 会被误否, 一层 4 个候选就有过半概率误否掉一个好解; 2σ 单侧 2.3%。
 * 趟间噪声超过 E2E_NOISE_MAX_ROWS 就停: 同一个态两趟都不一样, 尺本身坏了。 */
#define E2E_NOISE_MAX_ROWS 2
#define E2E_VETO_SIGMA 2.0
#define E2E_THREADS 8    /* 逐行 log-softmax 两遍 129280 个数 ~1 ms, 一块 512 个 val 行单线程 0.5 s; 分 8 路 */

struct e2e_s {
    int fd, S, V; const int *ids; int n;
    int *vidx; int nval; int tol_rows;   /* vidx[行] = 该行在 val 里的序号, 非 val 行 −1; tol_rows = 趟间噪声(行), Same top 容差的地板 */
    double kld_tol;                      /* 趟间噪声(KLD): 两个候选 KLD 差不超过它算一样, 再比 RMS Δp */
    uint8_t *hit;                        /* 本趟每个 val 行 Same top 中没中, e2e_run 末尾拷进读数的 hit */
    double kl_sum, rms_sum; long same, rows; int bad;   /* 本趟累加(rms_sum = top-32 联合窗均方的和, 报数时 sqrt(均值)×100 = 判决器的 RMS Δp) */
};
typedef struct { struct e2e_s *z; int row0; const float *lg; int t0, t1; double kl, rms; long same, rows; int bad; } e2e_job;

/* 非有限数按位模式判, 不用 isnan(引擎对象开着 fast-math, 这个 TU 没开, 但不赌): 10-22 01:xx 实撞 L31 一臂 KLD=nan,
 * 被当成"输给基线"悄悄否决 —— 换页压力下流式层经 UVM 缺页读盘、内核同时报 NVRM memdesc OOM, 一块 512 行尾部 logits 成了垃圾。
 * 垃圾读数不是"差", 是尺坏了: 停车比继续裁强, 继续裁等于拿坏尺选层。 */
static int e2e_nonfinite(double v) { uint64_t b; memcpy(&b, &v, sizeof b); return ((b >> 52) & 0x7ffu) == 0x7ffu; }

static void *e2e_worker(void *a) {
    e2e_job *j = a; struct e2e_s *z = j->z; const int V = z->V;
    float *tr = malloc((size_t)V * 4); double *w = malloc((size_t)V * 4 * sizeof(double));
    if (!tr || !w) { j->bad = 1; free(tr); free(w); return NULL; }
    for (int i = j->t0; i < j->t1 && !j->bad; i++) {
        const int row = j->row0 + i, vi = z->vidx[row];
        if (vi < 0) continue;
        const off_t off = 8 + (off_t)row * V * 4;
        if (pread(z->fd, tr, (size_t)V * 4, off) != (ssize_t)((size_t)V * 4)) { j->bad = 1; break; }
        posix_fadvise(z->fd, off, (off_t)V * 4, POSIX_FADV_DONTNEED);
        anchor_row_t o;
        anchor_row_metrics(tr, j->lg + (size_t)i * V, V, w, w + V, w + 2 * (size_t)V, w + 3 * (size_t)V, &o);
        if (e2e_nonfinite(o.kld) || e2e_nonfinite(o.rms)) {
            fprintf(stderr, "★端到端尺: 第 %d 行(val 第 %d 行) KLD/RMS 非有限数 —— 引擎这趟前向出了垃圾 logits, 读数不能用, 停车★\n", row, vi);
            j->bad = 1; break;
        }
        j->kl += o.kld; j->rms += o.rms; j->same += o.same; j->rows++; z->hit[vi] = (uint8_t)(o.same != 0);   /* 各线程写各自的行, 不重叠 */
    }
    free(tr); free(w);
    return NULL;
}

/* 引擎每块 logits 读回主机后回调(ds4_engine_v41_set_logits_hook): 只算 val 行 */
static int e2e_logits_hook(void *ud, int row0, int nrow, const float *lg, int V) {
    struct e2e_s *z = ud;
    if (V != z->V || row0 + nrow > z->S) { fprintf(stderr, "★端到端尺: 引擎 V=%d 行 %d+%d 与教师锚 V=%d S=%d 不符★\n", V, row0, nrow, z->V, z->S); return -1; }
    e2e_job jobs[E2E_THREADS]; pthread_t th[E2E_THREADS]; int nt = 0;
    const int per = (nrow + E2E_THREADS - 1) / E2E_THREADS;
    for (int t = 0; t < E2E_THREADS; t++) {
        const int a = t * per, b = a + per > nrow ? nrow : a + per;
        if (a >= b) break;
        memset(&jobs[nt], 0, sizeof jobs[nt]);
        jobs[nt].z = z; jobs[nt].row0 = row0; jobs[nt].lg = lg; jobs[nt].t0 = a; jobs[nt].t1 = b;
        if (pthread_create(&th[nt], NULL, e2e_worker, &jobs[nt])) { e2e_worker(&jobs[nt]); th[nt] = 0; }
        nt++;
    }
    for (int t = 0; t < nt; t++) {
        if (th[t]) pthread_join(th[t], NULL);
        z->kl_sum += jobs[t].kl; z->rms_sum += jobs[t].rms; z->same += jobs[t].same; z->rows += jobs[t].rows;
        if (jobs[t].bad) z->bad = 1;
    }
    return z->bad ? -1 : 0;
}

/* 打开教师锚并核对: 行数必须 = 拟合料行数, 文件大小必须 = 头 + S×V×4(半截锚不会报错, 只会 pread 到 0 行) */
static int e2e_open(ctx_t *c, const char *ref, const int *ids, int ntok) {
    struct e2e_s *z = calloc(1, sizeof *z);
    if (!z) return -1;
    z->fd = open(ref, O_RDONLY);
    if (z->fd < 0) { fprintf(stderr, "★端到端尺: 打不开教师锚 %s★\n", ref); return -1; }
    int32_t hd[2];
    if (pread(z->fd, hd, 8, 0) != 8) { fprintf(stderr, "★教师锚 %s 头截断★\n", ref); return -1; }
    z->S = hd[0]; z->V = hd[1];
    struct stat st;
    if (fstat(z->fd, &st) || st.st_size != (off_t)8 + (off_t)z->S * z->V * 4) {
        fprintf(stderr, "★教师锚 %s 大小 %lld ≠ 8 + %d×%d×4★\n", ref, (long long)st.st_size, z->S, z->V); return -1; }
    if (z->S != ntok) { fprintf(stderr, "★教师锚 %s 有 %d 行, 拟合料 %d 行 —— 不是同一份料的锚★\n", ref, z->S, ntok); return -1; }
    z->ids = ids; z->n = ntok;
    z->nval = c->n - c->nfit;
    z->vidx = malloc(sizeof(int) * (size_t)ntok); z->hit = calloc((size_t)z->nval, 1);
    if (!z->vidx || !z->hit) return -1;
    for (int i = 0; i < ntok; i++) z->vidx[i] = -1;
    for (int i = c->nfit; i < c->n; i++) z->vidx[c->perm[i]] = i - c->nfit;
    c->e2e = z; c->e2e_base.hit = NULL;
    printf("[端到端择优] 教师锚 %s (S=%d V=%d); val %d 行; 每层候选 = 层内 val 前 %d 个增益解(本层 Δb 层内闸过了就挂着它重取再解, Δb 端到端不撤); 尺 = Same top / KLD / RMS Δp; 裁决 = 三项全优者赢, 否则 Same top 比基线退超过 %.0f√不一致行数 的候选否决, 剩下 KLD 最低者赢(差在趟间噪声内比 RMS Δp 再比 Same top; 基线=不挂增益也参赛)\n",
           ref, z->S, z->V, z->nval, E2E_NCAND, E2E_VETO_SIGMA);
    return 0;
}

/* 目录里现在有没有插件文件: 下一趟前向挂不挂它。按盘上真实文件判, 不靠计数 —— 候选被撤掉、路由偏置被撤掉都要立刻反映。 */
static int dir_has_plugins(const ctx_t *c) {
    char p[4300]; struct stat st;
    for (int il = 0; il < 64; il++) {
        snprintf(p, sizeof p, "%s/gr_L%02d.bin", c->out_dir, il); if (!stat(p, &st)) return 1;
        snprintf(p, sizeof p, "%s/rb_L%02d.bin", c->out_dir, il); if (!stat(p, &st)) return 1;
    }
    return 0;
}

/* 挂上目录现状跑一趟全前向, 出 val 行三指标。钩子卸掉(整条跑完), 结束后由调用方重设。 */
static int e2e_run(ctx_t *c, const char *label, e2e_m *out) {
    struct e2e_s *z = c->e2e;
    z->kl_sum = z->rms_sum = 0; z->same = z->rows = 0; z->bad = 0;
    ds4_engine_v41_set_amp_dir(dir_has_plugins(c) ? c->out_dir : NULL);
    ds4_engine_v41_set_moe_hook(NULL, NULL);
    ds4_engine_v41_set_logits_hook(e2e_logits_hook, z);
    const double t = now_s();
    const int rc = ds4_engine_v41_score_ids(c->eng, c->ids_all, c->ntok_all, NULL, c->no_engram_all, 0);
    ds4_engine_v41_set_logits_hook(NULL, NULL);
    c->t_e2e += now_s() - t;
    if (rc || z->bad) { fprintf(stderr, "★端到端前向失败(%s) rc=%d★\n", label, rc); return -1; }
    if (z->rows != z->nval) { fprintf(stderr, "★端到端尺只收到 %ld/%d 个 val 行★\n", z->rows, z->nval); return -1; }
    out->kld = z->kl_sum / (double)z->rows;
    out->rms = 100.0 * sqrt(z->rms_sum / (double)z->rows);   /* 与 bench/anchor_metrics 报的 RMS Δp 同式: sqrt(逐行均方的均值)×100 */
    out->same = 100.0 * (double)z->same / (double)z->rows;
    if (!out->hit && !(out->hit = malloc((size_t)z->nval))) return -1;
    memcpy(out->hit, z->hit, (size_t)z->nval);
    printf("  [e2e] %-9s KLD %.5f  RMS Δp %.4f%%  Same top %.2f%%  (%.0fs)\n", label, out->kld, out->rms, out->same, now_s() - t);
    fflush(stdout);
    return 0;
}

typedef struct { char name[24]; e2e_m m; int veto; } e2e_cand;   /* veto = Same top 比基线退超过 2σ, 不许赢 */
/* ★裁决规则(10-21 用户: "没有旧车好, 说明选择指标有问题, 怎么改" ⇒ KLD 主裁; 之前的 Same top 优先是第一趟输给旧侧车的原因, 见文件头)★
 * 参赛者 = 基线(本层不挂) + 全部候选。①有谁三项(Same top / KLD / RMS Δp)都不劣于其他所有人, 它赢(全优者; Same top 的"不劣" = 差在 √d 内);
 * ②没有全优者: 先把 Same top 比基线退超过 E2E_VETO_SIGMA·√d 的候选否掉; 剩下的 KLD 最低者赢, KLD 差在趟间噪声内算一样, 一样比 RMS Δp 小,
 * 再一样比 Same top(超 √d 才算高)。基线赢 = 本层不挂。Same top 的本职回到终判(判决份五指标 + wt2)。 */
/* a 对 b 的 Same top 配对账: D = a 中 b 不中的行数 − b 中 a 不中的行数, d = 两者不一致的行数 */
static void e2e_pair(const ctx_t *c, const e2e_m *a, const e2e_m *b, int *D, int *d) {
    int ab = 0, ba = 0;
    for (int i = 0; i < c->e2e->nval; i++) { ab += a->hit[i] && !b->hit[i]; ba += b->hit[i] && !a->hit[i]; }
    *D = ab - ba; *d = ab + ba;
}
static double e2e_tol(const ctx_t *c, int d) { const double s = sqrt((double)d); return s > c->e2e->tol_rows ? s : (double)c->e2e->tol_rows; }
static int e2e_same_cmp(const ctx_t *c, const e2e_m *a, const e2e_m *b) {   /* +1 = a 的 Same top 更高(超噪声), −1 = 更低, 0 = 一样 */
    int D, d; e2e_pair(c, a, b, &D, &d);
    const double tol = e2e_tol(c, d);
    return D > tol ? 1 : (-D > tol ? -1 : 0);
}
/* x 对 y: 三项都不劣 */
static int e2e_dominates(const ctx_t *c, const e2e_m *x, const e2e_m *y) {
    return x->kld <= y->kld && x->rms <= y->rms && e2e_same_cmp(c, x, y) >= 0;
}
/* 返回赢家下标, -1 = 基线(本层不挂增益) */
static int e2e_pick(ctx_t *c, int il, e2e_cand *cs, int nc) {
    const e2e_m *b = &c->e2e_base;
    const e2e_m *all[1 + 2 * E2E_NCAND]; int na = 0;
    all[na++] = b;
    for (int k = 0; k < nc; k++) {
        all[na++] = &cs[k].m;
        int D, d; e2e_pair(c, &cs[k].m, b, &D, &d);
        cs[k].veto = -D > E2E_VETO_SIGMA * e2e_tol(c, d);
    }
    int best = 0; const char *how = "全优";
    for (int i = 0; i < na; i++) {   /* ① 全优者(全优者对基线 Same top 不劣, 不可能被否决) */
        int dom = 1;
        for (int j = 0; j < na && dom; j++) if (j != i && !e2e_dominates(c, all[i], all[j])) dom = 0;
        if (dom) { best = i; goto done; }
    }
    how = "KLD 最低";
    for (int i = 1; i < na; i++) {   /* ② 没被否决的里 KLD 最低; 差在趟间噪声内比 RMS Δp, 再比 Same top; 基线排第一, 平手不让位 */
        if (cs[i - 1].veto) continue;
        const double dk = all[i]->kld - all[best]->kld;
        if (dk < -c->e2e->kld_tol) best = i;
        else if (fabs(dk) <= c->e2e->kld_tol && (all[i]->rms < all[best]->rms || (all[i]->rms == all[best]->rms && e2e_same_cmp(c, all[i], all[best]) > 0))) best = i;
    }
done:
    fprintf(c->mf, "L%02d e2e base      KLD %.5f RMS %.4f Same %.2f%s\n", il, b->kld, b->rms, b->same, best == 0 ? " ★赢家(不挂)" : "");
    for (int k = 0; k < nc; k++) {   /* 每个候选对基线的配对账也落盘: 差几行 / 噪声几行 / 否没否决, 事后能复核裁决 */
        int D, d; e2e_pair(c, &cs[k].m, b, &D, &d);
        fprintf(c->mf, "L%02d e2e %-9s KLD %.5f RMS %.4f Same %.2f 对基线 %+d 行/噪声 %.1f%s%s\n", il, cs[k].name, cs[k].m.kld, cs[k].m.rms, cs[k].m.same, D, e2e_tol(c, d),
                cs[k].veto ? " 否决(Same top 退超 2σ)" : "", k + 1 == best ? " ★赢家" : "");
    }
    fprintf(c->mf, "L%02d e2e 裁决=%s\n", il, how);
    fflush(c->mf);
    printf("  [e2e] L%02d 基线 KLD %.5f RMS %.4f%% Same %.2f%% → %s(%s)\n", il, b->kld, b->rms, b->same, best ? cs[best - 1].name : "本层不挂增益", how);
    return best - 1;
}
/* 起点再跑一趟: 同一个态两趟读数的差 = 这把尺的趟间噪声, 落 manifest; Same top 差超过容差就停 —— 尺分不清容差以下的差别, 跑下去全是掷硬币。 */
static int e2e_noise_check(ctx_t *c) {
    e2e_m m = {0};
    if (e2e_run(c, "起点(复)", &m)) return -1;
    const double dk = fabs(m.kld - c->e2e_base.kld), dr = fabs(m.rms - c->e2e_base.rms), ds = fabs(m.same - c->e2e_base.same);
    int D, d; e2e_pair(c, &m, &c->e2e_base, &D, &d);   /* 两趟不一致的行数(同一个态, 理应 0) */
    free(m.hit);
    c->e2e->tol_rows = d; c->e2e->kld_tol = dk;
    fprintf(c->mf, "# 趟间噪声(起点两趟) ΔKLD %.2e ΔRMS %.2e ΔSame %.3f pp, 不一致 %d 行/%d ⇒ KLD 一样的门槛 %.2e, Same top 容差地板 %d 行(上限 %d); 候选对基线另按配对噪声 √不一致行数, 退超 %.0fσ 否决\n",
            dk, dr, ds, d, c->e2e->nval, dk, c->e2e->tol_rows, E2E_NOISE_MAX_ROWS, E2E_VETO_SIGMA);
    fflush(c->mf);
    printf("[端到端择优] 趟间噪声: ΔKLD %.2e ΔRMS %.2e ΔSame %.3f pp, 不一致 %d 行 ⇒ KLD 一样的门槛 %.2e, Same top 容差地板 %d 行\n", dk, dr, ds, d, dk, c->e2e->tol_rows);
    if (c->e2e->tol_rows > E2E_NOISE_MAX_ROWS) { fprintf(stderr, "★趟间噪声 %d 行 > 上限 %d 行: 这把尺分不清候选, 停车★\n", c->e2e->tol_rows, E2E_NOISE_MAX_ROWS); return -1; }
    return 0;
}
