/* anchor_row.h — 锚(FP 参考 logits)对学生 logits 的【逐位置】指标, 全仓唯一的一份实现(2026-10-21)。
 * 两个使用方: 判决器 bench/anchor_metrics(五指标 = 这些逐位量的平均)与反修解算器 amp/v41_amp_run 的端到端择优
 * (每层候选挂上跑全前向, 在拟合料 val 行上算 KLD / NLL / Same top 选赢家)。解算器选的尺与判决器判的尺必须逐式同源 ——
 * 抄第二份的下场是两边各改各的, 解算器按一把尺选、判决器按另一把尺判, 而差异藏在小数点后面谁也看不出来。 */
#ifndef DS4_ANCHOR_ROW_H
#define DS4_ANCHOR_ROW_H

typedef struct {
    double kld;    /* Σ_v p_ref·(ln p_ref − ln p_stu) */
    double smin;   /* Σ_v min(p_ref, p_stu): 分布还原率 */
    double dtop;   /* |p_ref(ref top) − p_stu(ref top)| */
    double rms;    /* top-32 联合窗内 (p_ref − p_stu)² 的均值(判决器再开方×100 报 RMS Δp) */
    int same;      /* argmax ref == argmax stu */
    int ir, iu;    /* ref / stu 的 argmax */
} anchor_row_t;

/* 行内 log-softmax → out[V](f64) */
void anchor_log_softmax_row(const float *x, double *out, int V);
/* teacher-forced NLL: −ln p(tgt), 只需要目标列, 免整行展开 */
double anchor_nll_at(const float *row, int V, int tgt);
/* 一行的全部逐位量。lr/ls/pr/ps 是 4 个 V 长的 f64 工作区, 调用方分配一次反复用(每行 4×129280×8 = 4 MB, 不在这里 malloc)。 */
void anchor_row_metrics(const float *ref, const float *stu, int V, double *lr, double *ls, double *pr, double *ps, anchor_row_t *o);

#endif
