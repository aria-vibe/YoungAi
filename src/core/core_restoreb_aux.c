/* core_restoreb_aux.c — 反修 B 段的逐层靶(2026-10-29, 配置 rb_aux=<目录> rb_auxw=<相对权重>)。
 *
 * 为什么(fable5 10-29): 只用出口 KL 训, 秩 64 的段 B 在练过的文字上推到 Same top 93%, 没练过的另外 15 种语言纹丝不动 ——
 *   出口 KL 每位只给一个分布, 几千位料对 2,620 万参数, 富余的自由度只能拿去背文字。逐层靶是权重的性质:
 *   t_l = y_fp(x_q) − y_q(x_q)(同一个 x 进 FP 原件和量化专家, 差 = 这一层量化丢的), 每位每层一整条 5120 维向量, 换哪篇文字都是同一个函数。
 *   训练损失 = 出口 KL + w·(1/层数)·Σ_l ‖xn·B·A − t_l‖² / ‖t_l‖²; KL 定往哪修, 逐层靶钉住"修的是权重的账, 不是这段文字"。
 * 靶从哪来: rb_dump=<目录> 一趟(rb_aux_dump)落当前链的 x/y_q → FP 原件 v41_teacher.py --force-x 同目录 --dump-moe 落 y_fp(x_q)。
 *   x 也用取料那一刻的(多层训练不存 MoE 入口, 反传是逐层重算的, 见 core_ptrain_bwd.c): 逐层项是固定 x 上的回归, 段 B 动了以后
 *   真实 x 会漂一点, 这一项不跟着漂(lr 5e-5、十几遍的量级上是一阶近似)。
 * 出错会怎样: 目录缺文件 / 长度对不上就停车; 样本不是靶序列的前缀(料里的短行)只走 KL。 */
#include "core_restoreb.h"
#include <sys/stat.h>
#ifndef DS4_NO_GPU

typedef struct { int fx[DS4_MAX_LAYER], fy[DS4_MAX_LAYER]; uint64_t rows[DS4_MAX_LAYER]; uint32_t n; int bad; } rb_dump_ctx;

static bool rb_pwrite_all(int fd, const void *p, size_t len, off_t off) {
    const char *b = p;
    while (len) { const ssize_t k = pwrite(fd, b, len, off); if (k <= 0) return false; b += k; len -= (size_t)k; off += k; }
    return true;
}
static bool rb_pread_all(int fd, void *p, size_t len, off_t off) {
    char *b = p;
    while (len) { const ssize_t k = pread(fd, b, len, off); if (k <= 0) return false; b += k; len -= (size_t)k; off += k; }
    return true;
}

static int rb_dump_hook(void *ud, int il, int pos0, int n, int D, int n_used, float clamp, const float *x, const float *y, const int *sel,
                        const float *rw, const float *alpha, const float *ye, const float *ysh) {
    rb_dump_ctx *d = ud; (void)n_used; (void)clamp; (void)sel; (void)rw; (void)alpha; (void)ye; (void)ysh;
    if (il < 0 || il >= (int)DS4_N_LAYER) return 0;
    if (D != (int)DS4_N_EMBD || pos0 < 0 || (uint64_t)pos0 + (uint64_t)n > d->n) { fprintf(stderr, "ds4: ★[restore-b 取料] L%02d 形状不符(D %d pos0 %d n %d)★\n", il, D, pos0, n); d->bad = 1; return -1; }
    const size_t len = (size_t)n * D * 4; const off_t off = (off_t)pos0 * D * 4;
    if (!rb_pwrite_all(d->fx[il], x, len, off) || !rb_pwrite_all(d->fy[il], y, len, off)) { fprintf(stderr, "ds4: ★[restore-b 取料] L%02d 写盘失败★\n", il); d->bad = 1; return -1; }
    d->rows[il] += (uint64_t)n;
    return 0;
}

bool rb_aux_dump(ds4_engine *e, pt_run *r, const int32_t *ids, uint32_t n, const char *dir) {
    rb_dump_ctx d; memset(&d, 0, sizeof d); d.n = n;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) d.fx[il] = d.fy[il] = -1;
    mkdir(dir, 0755);
    bool ok = true;
    char p[1200];
    for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {
        snprintf(p, sizeof p, "%s/x_L%02u.bin", dir, il); d.fx[il] = open(p, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        snprintf(p, sizeof p, "%s/yq_L%02u.bin", dir, il); d.fy[il] = open(p, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        ok = d.fx[il] >= 0 && d.fy[il] >= 0;
    }
    snprintf(p, sizeof p, "%s/ids.txt", dir);
    FILE *f = ok ? fopen(p, "w") : NULL;
    for (uint32_t i = 0; f && i < n; i++) fprintf(f, "%d\n", ids[i]);
    if (f) fclose(f); else ok = false;
    if (!ok) fprintf(stderr, "ds4: ★[restore-b 取料] 建不了 %s 下的文件★\n", dir);
    ds4_v41_state *st = &r->st;
    if (ok) {   /* 与 rb_step 的前缀块同口径: 不存档、出口只算末行、解码器回放关掉(每层每位都走预填核) */
        v41_tsave *keep = st->tsave;
        const int rp = g_ds4_v41_replay;
        g_ds4_v41_replay = 0; st->tsave = NULL; st->head_last_only = 1; st->ced_skip = 0;
        pt_state_reset(st);
        ds4_engine_v41_set_moe_hook_layer(-1);
        ds4_engine_v41_set_moe_hook(rb_dump_hook, &d);
        const double t0 = now_sec();
        for (uint32_t c0 = 0; ok && c0 < n; ) {
            const uint32_t nc = n - c0 < st->cap_tok ? n - c0 : st->cap_tok;
            ok = v41_forward(e, st, ids + c0, nc) && !d.bad;
            c0 += nc;
            fprintf(stderr, "ds4: [restore-b 取料] %u / %u 位, %.0f s\n", c0, n, now_sec() - t0);
        }
        ds4_engine_v41_set_moe_hook(NULL, NULL);
        g_ds4_v41_replay = rp; st->tsave = keep; st->head_last_only = 0;
        pt_state_reset(st);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (ok && d.rows[il] != n) { fprintf(stderr, "ds4: ★[restore-b 取料] L%02u 只落了 %llu / %u 行★\n", il, (unsigned long long)d.rows[il], n); ok = false; }
        if (d.fx[il] >= 0) close(d.fx[il]);
        if (d.fy[il] >= 0) close(d.fy[il]);
    }
    if (ok) fprintf(stderr, "ds4: [restore-b 取料] %u 位 × %u 层 → %s(x_Lnn / yq_Lnn / ids.txt)\n", n, DS4_N_LAYER, dir);
    return ok;
}

struct rb_aux {
    int fx[DS4_MAX_LAYER], fq[DS4_MAX_LAYER], ff[DS4_MAX_LAYER];
    int32_t *ids; uint32_t n, cap, mcap;   /* cap = 状态一块的行数(评估分块); mcap = maxlen(一道样本答案位最多几行, 投影矩阵按它开) */
    float *hx, *hq, *hf;                     /* 主机暂存 [cap][D] */
    ds4_gpu_tensor *X, *R, *F, *sA, *sB, *T, *gT, *Tm;   /* Tm [cap][cap]: 敏感度投影 (·)·Gᵀ */
};

void rb_aux_close(rb_aux *a) {
    if (!a) return;
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) { if (a->fx[il] >= 0) close(a->fx[il]); if (a->fq[il] >= 0) close(a->fq[il]); if (a->ff[il] >= 0) close(a->ff[il]); }
    ds4_gpu_tensor *t[] = { a->X, a->R, a->F, a->sA, a->sB, a->T, a->gT, a->Tm };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) if (t[i]) ds4_gpu_tensor_free(t[i]);
    free(a->hx); free(a->hq); free(a->hf); free(a->ids); free(a);
}

rb_aux *rb_aux_open(const char *dir, const pt_cfg *c, const pt_run *r) {
    rb_aux *a = xmalloc_zeroed(1, sizeof *a);
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) a->fx[il] = a->fq[il] = a->ff[il] = -1;
    const uint32_t D = DS4_N_EMBD, K = r->K;
    char p[1200];
    snprintf(p, sizeof p, "%s/ids.txt", dir);
    FILE *f = fopen(p, "r");
    uint32_t cap = 0; int v;
    while (f && fscanf(f, "%d", &v) == 1) { if (a->n == cap) { cap = cap ? cap * 2u : 4096u; a->ids = realloc(a->ids, (size_t)cap * 4); } a->ids[a->n++] = v; }
    if (f) fclose(f);
    bool ok = a->n > 1;
    if (!ok) fprintf(stderr, "ds4: ★[restore-b] 逐层靶 %s 没有 ids.txt★\n", p);
    const off_t want = (off_t)a->n * D * 4;
    for (uint32_t il = c->layer_lo; ok && il <= c->layer_hi; il++) {
        const char *nm[3] = { "x", "yq", "y" }; int *fd[3] = { &a->fx[il], &a->fq[il], &a->ff[il] };
        for (int k = 0; ok && k < 3; k++) {
            snprintf(p, sizeof p, "%s/%s_L%02u.bin", dir, nm[k], il);
            struct stat sb;
            *fd[k] = open(p, O_RDONLY);
            ok = *fd[k] >= 0 && !fstat(*fd[k], &sb) && sb.st_size == want;
            if (!ok) fprintf(stderr, "ds4: ★[restore-b] 逐层靶 %s 缺或长度不是 %u 位 × %u★\n", p, a->n, D);
        }
    }
    a->cap = r->st.cap_tok;
    if (ok) {
        const uint64_t rb = (uint64_t)a->cap * D * 4;
        a->hx = xmalloc(rb); a->hq = xmalloc(rb); a->hf = xmalloc(rb);
        a->X = v41_alloc(rb, &ok); a->R = v41_alloc(rb, &ok); a->F = v41_alloc(rb, &ok);
        a->sA = v41_alloc((uint64_t)K * D * 4, &ok); a->sB = v41_alloc((uint64_t)K * D * 4, &ok);
        a->T = v41_alloc((uint64_t)a->cap * K * 4, &ok); a->gT = v41_alloc((uint64_t)a->cap * K * 4, &ok);
        a->mcap = c->maxlen < a->cap ? c->maxlen : a->cap;
        a->Tm = v41_alloc((uint64_t)a->mcap * a->mcap * 4, &ok);
    }
    if (!ok) { rb_aux_close(a); return NULL; }
    fprintf(stderr, "ds4: [restore-b] 逐层靶 %s: %u 位, L%u-L%u\n", dir, a->n, c->layer_lo, c->layer_hi);
    return a;
}

uint32_t rb_aux_len(const rb_aux *a) { return a ? a->n : 0; }

bool rb_aux_covers(const rb_aux *a, const pt_sample *s) {
    return a && s->sids && s->sn <= a->n && !memcmp(s->sids, a->ids, (size_t)s->sn * 4);
}

bool rb_aux_rows(rb_aux *a, const pt_cfg *c, pt_run *r, uint32_t p0, uint32_t m, double w, int grad, double *res, double *tgt, double *gn2) {
    const uint32_t D = DS4_N_EMBD, K = r->K, nl = c->layer_hi - c->layer_lo + 1u;
    if (!m || m > a->cap || p0 + m > a->n) { fprintf(stderr, "ds4: ★[restore-b] 逐层靶行 [%u, %u) 越界(靶 %u 位, 暂存 %u 行)★\n", p0, p0 + m, a->n, a->cap); return false; }
    const size_t len = (size_t)m * D * 4; const off_t off = (off_t)p0 * D * 4;
    const uint64_t me = (uint64_t)m * D, ke = (uint64_t)K * D;
    for (uint32_t il = c->layer_lo; il <= c->layer_hi; il++) {
        ds4_gpu_tensor *xv = a->X;
        bool ok = rb_pread_all(a->fx[il], a->hx, len, off) && rb_pread_all(a->fq[il], a->hq, len, off) && rb_pread_all(a->ff[il], a->hf, len, off) &&
                  ds4_gpu_tensor_write(a->X, 0, a->hx, len) && ds4_gpu_tensor_write(a->R, 0, a->hq, len) && ds4_gpu_tensor_write(a->F, 0, a->hf, len) &&
                  ds4_gpu_bwd_axpy_tensor(a->R, a->F, -1.f, me);   /* R = y_q − y_fp = −t */
        double tl = 0, rl = 0;
        if (ok) ok = ds4_gpu_bwd_sumsq_tensor(a->R, me, &tl) &&
                     ds4_gpu_v41_amp_apply_tensor(a->R, xv, r->st.ampA[il], r->st.ampB[il], a->T, m, D, K) &&   /* R = xn·B·A − t */
                     ds4_gpu_bwd_sumsq_tensor(a->R, me, &rl);
        if (ok && grad && tl > 0) {   /* ∂/∂Z [w/nl · ‖Z−t‖²/‖t‖²] = (2w/(nl·‖t‖²))·R; 先进暂存量范数, 再按系数加进 gA/gB */
            const double coef = 2.0 * w / ((double)nl * tl);
            double sa = 0, sb = 0;
            ok = ds4_gpu_tensor_fill_f32(a->sA, 0.f, ke) && ds4_gpu_tensor_fill_f32(a->sB, 0.f, ke) &&
                 ds4_gpu_bwd_amp_tensor(a->sA, a->sB, NULL, a->R, xv, r->st.ampA[il], r->st.ampB[il], a->T, a->gT, m, D, K) &&
                 ds4_gpu_bwd_sumsq_tensor(a->sA, ke, &sa) && ds4_gpu_bwd_sumsq_tensor(a->sB, ke, &sb);
            if (ok) *gn2 += coef * coef * (sa + sb);
            if (ok && grad == 1) ok = ds4_gpu_bwd_axpy_tensor(r->gA[il], a->sA, (float)coef, ke) && ds4_gpu_bwd_axpy_tensor(r->gB[il], a->sB, (float)coef, ke);
        }
        if (!ok) { fprintf(stderr, "ds4: ★[restore-b] 逐层靶 L%02u 行 [%u, %u) 计算失败★\n", il, p0, p0 + m); return false; }
        *res += rl; *tgt += tl;
    }
    return true;
}

bool rb_aux_layer(rb_auxstep *s, const pt_cfg *c, pt_run *r, uint32_t il, const ds4_gpu_tensor *gy, const ds4_gpu_tensor *xn) {
    rb_aux *a = s->ax;
    const uint32_t D = DS4_N_EMBD, K = r->K, m = s->m, nl = c->layer_hi - c->layer_lo + 1u;
    if (!m || m > a->mcap || s->p0 + m > a->n || a->fq[il] < 0) { fprintf(stderr, "ds4: ★[restore-b] 逐层靶 L%02u 行 [%u, %u) 越界或没开★\n", il, s->p0, s->p0 + m); return false; }
    const size_t len = (size_t)m * D * 4; const off_t off = (off_t)s->p0 * D * 4;
    const uint64_t me = (uint64_t)m * D, ke = (uint64_t)K * D, mm = (uint64_t)m * m;
    ds4_gpu_tensor *G = pt_rows(gy, s->r0, m, D), *X = pt_rows(xn, s->r0, m, D);
    double tl = 0, rl = 0, tw = 0, pw = 0;
    /* (·)·Gᵀ 与 (·)·Gᵀ·G 借放大器应用核算: 把 G 当成秩 m 的 A 与 B, 中间量 T[m][m] = (·)·Gᵀ 正是要的投影, y += T·G 是梯度要的那一项 */
    bool ok = G && X && rb_pread_all(a->fq[il], a->hq, len, off) && rb_pread_all(a->ff[il], a->hf, len, off) &&
              ds4_gpu_tensor_write(a->R, 0, a->hq, len) && ds4_gpu_tensor_write(a->F, 0, a->hf, len) &&
              ds4_gpu_bwd_axpy_tensor(a->R, a->F, -1.f, me) &&                                   /* R = −t */
              ds4_gpu_bwd_sumsq_tensor(a->R, me, &tl) &&
              ds4_gpu_tensor_fill_f32(a->F, 0.f, me) && ds4_gpu_v41_amp_apply_tensor(a->F, a->R, G, G, a->Tm, m, D, m) &&
              ds4_gpu_bwd_sumsq_tensor(a->Tm, mm, &tw) &&                                         /* ‖t·Gᵀ‖² */
              ds4_gpu_v41_amp_apply_tensor(a->R, X, r->st.ampA[il], r->st.ampB[il], a->T, m, D, K) &&   /* R = Z − t */
              ds4_gpu_bwd_sumsq_tensor(a->R, me, &rl) &&
              ds4_gpu_tensor_fill_f32(a->F, 0.f, me) && ds4_gpu_v41_amp_apply_tensor(a->F, a->R, G, G, a->Tm, m, D, m) &&   /* Tm = P = R·Gᵀ, F = P·G */
              ds4_gpu_bwd_sumsq_tensor(a->Tm, mm, &pw);
    if (ok && s->grad && tw > 0) {   /* ∂/∂Z [w/nl · ‖R·Gᵀ‖²/tw] = (2w/(nl·tw))·P·G */
        const double coef = 2.0 * s->w / ((double)nl * tw);
        double sa = 0, sb = 0;
        ok = ds4_gpu_tensor_fill_f32(a->sA, 0.f, ke) && ds4_gpu_tensor_fill_f32(a->sB, 0.f, ke) &&
             ds4_gpu_bwd_amp_tensor(a->sA, a->sB, NULL, a->F, X, r->st.ampA[il], r->st.ampB[il], a->T, a->gT, m, D, K) &&
             ds4_gpu_bwd_sumsq_tensor(a->sA, ke, &sa) && ds4_gpu_bwd_sumsq_tensor(a->sB, ke, &sb);
        if (ok) s->gn2 += coef * coef * (sa + sb);
        if (ok && s->grad == 1) ok = ds4_gpu_bwd_axpy_tensor(r->gA[il], a->sA, (float)coef, ke) && ds4_gpu_bwd_axpy_tensor(r->gB[il], a->sB, (float)coef, ke);
    }
    if (G) ds4_gpu_tensor_free(G);
    if (X) ds4_gpu_tensor_free(X);
    if (!ok) { fprintf(stderr, "ds4: ★[restore-b] 逐层靶 L%02u(反传内)计算失败★\n", il); return false; }
    s->res += rl; s->tgt += tl; s->wres += pw; s->wtgt += tw;
    return true;
}

bool rb_aux_eval(rb_aux *a, const pt_cfg *c, pt_run *r, uint32_t p_lo, uint32_t p_hi, uint32_t stride, double *res, double *tgt) {
    double g = 0;
    uint32_t k = 0;
    for (uint32_t p = p_lo; p < p_hi; p += a->cap, k++) {
        if (stride > 1 && k % stride) continue;
        const uint32_t m = p_hi - p < a->cap ? p_hi - p : a->cap;
        if (!rb_aux_rows(a, c, r, p, m, 0.0, 0, res, tgt, &g)) return false;
    }
    return true;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_restoreb_aux_nonempty_tu;
