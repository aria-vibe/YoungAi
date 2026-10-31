/* core_restoreb.h — 反修 B 段(core_restoreb.c)与它的逐层靶(core_restoreb_aux.c)之间的声明。只被这两个文件包含。 */
#ifndef DS4_CORE_RESTOREB_H
#define DS4_CORE_RESTOREB_H
#include "core_ptrain.h"
#ifndef DS4_NO_GPU

/* 取料: 用当前链(段 A + 起点段 B)把 ids 整条前向一遍, 每层 MoE 入口 xn 与放大器之前的输出 y 落 <dir>/x_Lnn.bin / yq_Lnn.bin
 * (f32 [n][5120], 第 p 行 = 位置 p), 序列落 <dir>/ids.txt(一行一个 id)。x_Lnn.bin 是给 FP 原件 --force-x 用的(v41_teacher.py:
 * 每层 MoE 入口钉成这份 x, 同目录 --dump-moe 落 y_Lnn.bin = y_fp(x_q))。 */
bool rb_aux_dump(ds4_engine *e, pt_run *r, const int32_t *ids, uint32_t n, const char *dir);

typedef struct rb_aux rb_aux;
rb_aux *rb_aux_open(const char *dir, const pt_cfg *c, const pt_run *r);
void rb_aux_close(rb_aux *a);
uint32_t rb_aux_len(const rb_aux *a);
/* 这道样本的序列是不是靶目录那条的前缀(不是 = 这道没有逐层靶, 只走 KL) */
bool rb_aux_covers(const rb_aux *a, const pt_sample *s);
/* 一段位置 [p0, p0+m) 的逐层残差: res += Σ_l ‖xn·B·A − t‖², tgt += Σ_l ‖t‖²(t = y_fp(x_q) − y_q, 放大器之前; x 读靶目录的 x_Lnn.bin)。
 * grad 0 = 只量; 1 = 梯度 ∇[w/层数 · Σ_l ‖·‖²/‖t‖²] 加进 gA/gB; 2 = 只量这份梯度的范数(gn2 += ‖·‖², 不加)。 */
bool rb_aux_rows(rb_aux *a, const pt_cfg *c, pt_run *r, uint32_t p0, uint32_t m, double w, int grad, double *res, double *tgt, double *gn2);
/* 评估: [p_lo, p_hi) 按 cap 分块、每 stride 块取一块, 残差与靶的平方和累加进 res / tgt(比值 = 逐层靶还剩几成没补上) */
bool rb_aux_eval(rb_aux *a, const pt_cfg *c, pt_run *r, uint32_t p_lo, uint32_t p_hi, uint32_t stride, double *res, double *tgt);

/* 一道样本反传时的逐层项(rb_layers_bwd 每层调一次 rb_aux_layer): 窗口第 r0 行起 m 行 = 靶序列第 p0 位起 m 位。
 * 敏感度加权: F = Σ_j g_j g_jᵀ(g_j = 本题第 j 位出口 KL 对本层 MoE 输出的梯度, 本题自己估的下游在乎哪些方向), 逐层项
 *   J_l = Σ_i (Z_i − t_i)ᵀ F (Z_i − t_i) / Σ_i t_iᵀ F t_i = ‖(Z − T)·Gᵀ‖² / ‖T·Gᵀ‖²; 不加权的 ‖Z − T‖²/‖T‖² 只记账。
 * grad: 0 = 只量; 1 = ∇[w/层数 · J_l] 加进 gA/gB; 2 = 只量这份梯度的范数(gn2, 第 1 步定标用)。x 用本步重算的 xn(当前链), t 用取料那一刻的。 */
typedef struct {
    rb_aux *ax; uint32_t r0, m, p0; double w; int grad;
    double res, tgt, wres, wtgt, gn2;   /* 累加: 不加权残差/靶, 加权残差/靶, 逐层项梯度范数² */
} rb_auxstep;
bool rb_aux_layer(rb_auxstep *s, const pt_cfg *c, pt_run *r, uint32_t il, const ds4_gpu_tensor *gy, const ds4_gpu_tensor *xn);
/* 带路由的段 B(10-29, rb_route=1): y += ((xn·B) ⊙ u)·A, u = 1 + Σ_j rw_j·S[e_j](src/cuda/cuda_amp_route.inc.cu)。S 住 r->st.ampS(训练态自己的,
 * v41_amp_free 一起放), 梯度/Adam 状态住这里; 起点 rb_init 目录有 ampr_Lnn.bin 就读, 没有 = 0(与现役段 B 逐位同)。落盘 <ckpt>/ampr_Lnn.bin。 */
typedef struct { int on; float lr; ds4_gpu_tensor *gS[DS4_MAX_LAYER], *mS[DS4_MAX_LAYER], *vS[DS4_MAX_LAYER]; } rb_route;
bool rb_route_alloc(rb_route *rt, const pt_cfg *c, pt_run *r, const char *init);
bool rb_route_step(rb_route *rt, const pt_cfg *c, pt_run *r, uint32_t step, double *gn);   /* Adam(核里清零梯度), gn = ‖∇S‖ */
bool rb_route_save(const pt_cfg *c, const pt_run *r, const char *dir);
void rb_route_free(rb_route *rt);
/* core_restoreb_init.c: 起点 + 参数锚(EWC 的解耦版, 10-31; 为什么与文件格式见该文件头)。rb_anchor_alloc 要在 rb_load_init 之后调(p0 = 此刻的 A/B);
 * rb_anchor_step 每个优化步在 Adam 之前调(攒数据梯度的 g²); rb_anchor_pull 在 Adam 之后调(θ ← θ − (F/(F̄·S))·(θ−θ*)); rb_anchor_save 随 ckpt 落 fisher_Lnn.bin。 */
bool rb_load_init(const pt_cfg *c, pt_run *r, const char *dir);
typedef struct {
    float lam, coef; bool have_f; uint32_t nold, steps; double fmean, pen, gn2;   /* lam = rb_anchor 的 S, coef = 1/(F̄·S); 起点有没有 F / 攒过几轮 / 本轮步数 / F 均值 / 本轮累计 Σ F·Δθ² 与拉回位移² */
    ds4_gpu_tensor *p0A[DS4_MAX_LAYER], *p0B[DS4_MAX_LAYER], *FA[DS4_MAX_LAYER], *FB[DS4_MAX_LAYER], *accA[DS4_MAX_LAYER], *accB[DS4_MAX_LAYER];
} rb_anchor;
bool rb_anchor_alloc(rb_anchor *an, const pt_cfg *c, const pt_run *r, const char *init, float lam);
bool rb_anchor_step(rb_anchor *an, const pt_cfg *c, pt_run *r, double *pen, double *gn2);
bool rb_anchor_pull(rb_anchor *an, const pt_cfg *c, pt_run *r, double *pen, double *gn2);
bool rb_anchor_save(const rb_anchor *an, const pt_cfg *c, const char *dir);
void rb_anchor_free(rb_anchor *an);
/* core_restoreb_bwd.c: 段 B 自己的逐层反传(= pt_layers_bwd 单题路 + 每层 g_y 出来后调 rb_aux_layer; as 为 NULL 不调; rt 非 NULL = 带路由) */
bool rb_layers_bwd(ds4_engine *e, const pt_cfg *c, pt_run *r, uint32_t n, rb_auxstep *as, rb_route *rt);

#endif /* !DS4_NO_GPU */
#endif
