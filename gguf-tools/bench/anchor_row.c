/* anchor_row.c — 逐位置指标的实现(2026-10-21 从 anchor_metrics.c 原样搬出, 式子一个字没改; 见 anchor_row.h 为什么要单独一份)。 */
#include "anchor_row.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

void anchor_log_softmax_row(const float *x, double *out, int V) {
    float mx = x[0];
    for (int v = 1; v < V; v++) if (x[v] > mx) mx = x[v];
    double s = 0;
    for (int v = 0; v < V; v++) s += exp((double)x[v] - mx);
    double lse = log(s);
    for (int v = 0; v < V; v++) out[v] = (double)x[v] - mx - lse;
}

double anchor_nll_at(const float *row, int V, int tgt) {
    float mx = row[0];
    for (int v = 1; v < V; v++) if (row[v] > mx) mx = row[v];
    double s = 0;
    for (int v = 0; v < V; v++) s += exp((double)row[v] - mx);
    return -((double)row[tgt] - mx - log(s));
}

/* top-K 索引(无序), 简单选择: 维护 K 槽最小值替换 */
static void topk_idx(const double *p, int V, int K, int *idx) {
    double vals[64]; int n = 0;
    for (int v = 0; v < V; v++) {
        if (n < K) { idx[n] = v; vals[n] = p[v]; n++; continue; }
        int mi = 0;
        for (int i = 1; i < K; i++) if (vals[i] < vals[mi]) mi = i;
        if (p[v] > vals[mi]) { vals[mi] = p[v]; idx[mi] = v; }
    }
}
static int cmp_int(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }

void anchor_row_metrics(const float *r, const float *s, int V, double *lr, double *ls, double *pr, double *ps, anchor_row_t *o) {
    anchor_log_softmax_row(r, lr, V); anchor_log_softmax_row(s, ls, V);
    double kl = 0; int ir = 0, iu = 0;
    for (int v = 0; v < V; v++) {
        pr[v] = exp(lr[v]); ps[v] = exp(ls[v]);
        kl += pr[v] * (lr[v] - ls[v]);
        if (r[v] > r[ir]) ir = v;
        if (s[v] > s[iu]) iu = v;
    }
    o->kld = kl;
    { double sm = 0; for (int v = 0; v < V; v++) sm += pr[v] < ps[v] ? pr[v] : ps[v];
      o->smin = sm; }   /* 分布还原率 Σmin(还原率铁律的主尺) */
    o->same = (ir == iu); o->ir = ir; o->iu = iu;
    o->dtop = fabs(pr[ir] - ps[ir]);
    /* top32 联合窗均方 */
    int ia[32], ib[32], u[64];
    topk_idx(pr, V, 32, ia); topk_idx(ps, V, 32, ib);
    memcpy(u, ia, sizeof ia); memcpy(u + 32, ib, sizeof ib);
    qsort(u, 64, sizeof(int), cmp_int);
    double d2 = 0; int cnt = 0;
    for (int i = 0; i < 64; i++) {
        if (i && u[i] == u[i - 1]) continue;
        double d = pr[u[i]] - ps[u[i]];
        d2 += d * d; cnt++;
    }
    o->rms = d2 / cnt;
}
