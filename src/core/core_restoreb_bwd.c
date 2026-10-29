/* core_restoreb_bwd.c — 反修 B 段自己的逐层反传(2026-10-29)。
 *
 * 为什么另写一份(用户 10-29: "新写一段 b 段后训练代码就可以了"): 逐层靶要按下游敏感度加权, 要的是每层 MoE 出口的梯度 g_y ——
 *   它只在逐层反传里活着(推完这一层就被下一层盖掉), 后训练 ③ 的 core_ptrain_layer.c 不许动, 就照抄一份到这里, 在 g_y 刚算出来、
 *   放大器反传之前多调一次 rb_aux_layer。其余每一步与 core_ptrain_layer.c 的 pt_layer_bwd / pt_layers_bwd 同一串核、同一个次序
 *   (重算 → ffn 半层 → 注意力半层 → engram 门), 只删了诊断件(dbglayer / 重算对拍 / prof / hc 边界检查 / 毒值)。
 *   ③ 那边改了反传(加了要求导的东西、换了次序)这里要跟着改, 否则段 B 是对着另一个函数求的导 —— 两份对拍: 同一题 rb_aux 不挂时,
 *   这里与 pt_layers_bwd 累出来的 gA/gB 应逐位同。
 * 缓冲全用 ③ 分好的(pt_run.lb / save, pt_layer_alloc), 不另开。 */
#include "core_restoreb.h"
#ifndef DS4_NO_GPU

#define RB_CP(dst, src, bytes) ds4_gpu_tensor_copy((dst), 0, (src), 0, (bytes))
#define RB_MT(gx, w, in, out, gy, acc) ds4_gpu_bwd_matmul_t_tensor((gx), m->map, m->size, (w)->type, (w)->abs_offset, (in), (out), (gy), n, (acc))

static bool rb_has_engram(const ds4_v41_state *st, uint32_t il) { return !st->no_engram && g_ds4_v41.engram_index_of[il] >= 0; }
static void rb_free_views(ds4_gpu_tensor **v, size_t n) { for (size_t i = 0; i < n; i++) if (v[i]) ds4_gpu_tensor_free(v[i]); }

/* 段 B 逐题过(nopack), 注意力只有一段: 行 0 起整段, 组数/topk 等存档在 r->save(pt_segs 的单题路) */
typedef struct { uint32_t ng, topk, iratio; ds4_gpu_tensor *win, *comp, *idx; } rb_seg;
static void rb_seg_of(const pt_run *r, uint32_t il, uint32_t n, rb_seg *g) {
    const ds4_v41_state *st = &r->st; const v41_tsave *sv = &r->save;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    const int16_t src = ratio ? g_ds4_v41.kv_source_of[il] : -1;
    g->ng = sv->ng[il]; g->topk = sv->topk[il]; g->iratio = sv->iratio[il]; g->win = st->win[il]; g->comp = src >= 0 ? st->comp_kv[src] : NULL;
    g->idx = ratio && g->ng && g->topk ? ds4_gpu_tensor_view(sv->idx[il], 0, (uint64_t)n * g->topk * 4) : NULL;
}

/* = pt_layer_recompute 的单题路: 从存档的入口把第 il 层重算一遍, 中间量抄进 lb; 结束时 st 里留着本层 MoE 的 y/sel/rw/glog/sg/su 与 attn_out */
static bool rb_layer_recompute(ds4_engine *e, pt_run *r, uint32_t il, uint32_t n) {
    ds4_v41_state *st = &r->st; const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    pt_layer_buf *b = &r->lb; const v41_tsave *sv = &r->save;
    const uint64_t E = DS4_N_EMBD, HC = DS4_N_HC, MIX = 2u * HC + HC * HC, HD = DS4_N_HEAD_DIM, QH = (uint64_t)DS4_N_HEAD * HD, SWA = DS4_N_SWA;
    const uint32_t ratio = ds4_layer_compress_ratio(il);
    st->n = n; st->pos0 = r->pfx; st->graph = 0;
    bool ok = il == 0u ? v41_embed(m, st->x, st->tok, e->weights.token_embd, DS4_N_VOCAB, n, E) && ds4_gpu_v41_expand_hc_tensor(b->hcin, st->x, (uint32_t)E, (uint32_t)HC, n)
                       : ds4_gpu_bwd_unpack_bf16_tensor(b->hcin, sv->hc_in[il], n * HC * E);
    if (ok && rb_has_engram(st, il))
        ok = sv->ekv[il] && ds4_gpu_bwd_unpack_bf16_tensor(b->ekvf, sv->ekv[il], n * (HC + 1u) * E) && RB_CP(b->hpre, b->hcin, n * HC * E * 4) &&
             ds4_gpu_v41_engram_gate_tensor(b->hcin, b->ekvf, m->map, m->size, l->engram_q->abs_offset, l->engram_k->abs_offset, (uint32_t)E, (uint32_t)HC, n, DS4_RMS_EPS);
    ok = ok && RB_CP(st->hc, b->hcin, n * HC * E * 4) &&
              RB_CP(st->pre_mix, sv->pm_in[il], n * HC * 4) && v41_hc_half(e, st, l, true) &&
              RB_CP(b->mixa, st->mix, n * MIX * 4) && RB_CP(b->posta, st->post, n * HC * 4) && RB_CP(b->comba, st->comb, n * HC * HC * 4) &&
              RB_CP(b->prea, st->pre_mix, n * HC * 4) && RB_CP(b->xa, st->x, n * E * 4) && RB_CP(b->xna, st->xn, n * E * 4) &&
              v41_attn_in(e, st, il) && RB_CP(b->qr, st->qr, n * DS4_N_LORA_Q * 4) && RB_CP(b->kv, st->kv, n * HD * 4) &&
              ds4_gpu_tensor_copy(st->win[il], SWA * HD * 4, st->kvn, 0, n * HD * 4);
    if (ok && r->pfx && r->ring_snap[il]) ok = ds4_gpu_tensor_copy(st->win[il], 0, r->ring_snap[il], 0, SWA * HD * 4);
    if (ok) {
        rb_seg g; rb_seg_of(r, il, n, &g);
        const bool hasc = g.idx != NULL;
        ok = (!(ratio && g.ng && g.topk) || hasc) &&
             ds4_gpu_v41_sparse_attn_tensor(st->o, st->q, g.win, hasc ? g.comp : NULL, g.idx, m->map, m->size, l->attn_sinks->abs_offset, n, r->pfx, (uint32_t)SWA,
                                            g.ng, hasc ? g.topk : 0u, hasc ? g.iratio : 0u, DS4_N_HEAD, (uint32_t)HD,
                                            (float)(1.0 / sqrt((double)HD)), 0, 1, 0u, NULL, 0u) != 0;
        if (g.idx) ds4_gpu_tensor_free(g.idx);
    }
    if (ok) ok = RB_CP(b->o, st->o, n * QH * 4) && v41_attn_out(e, st, il) &&
                 ds4_gpu_v41_hc_post_tensor(st->hc2, st->attn_out, st->hc, st->post, st->comb, (uint32_t)E, (uint32_t)HC, n);
    if (ok) { ds4_gpu_tensor *t = st->hc; st->hc = st->hc2; st->hc2 = t; }
    if (ok) ok = v41_hc_half(e, st, l, false) &&
                 RB_CP(b->mixf, st->mix, n * MIX * 4) && RB_CP(b->postf, st->post, n * HC * 4) && RB_CP(b->combf, st->comb, n * HC * HC * 4) &&
                 RB_CP(b->pref, st->pre_mix, n * HC * 4) && RB_CP(b->xf, st->x, n * E * 4) && RB_CP(b->xnf, st->xn, n * E * 4) &&
                 ds4_gpu_bwd_moe_capture(1) && v41_moe(m, l, st, il);
    (void)ds4_gpu_bwd_moe_capture(0);
    return ok;
}

/* = pt_compress_bwd 的前缀/单题路(r0 = 0, 组号与位置表都从 g0 起) */
static bool rb_compress_bwd(ds4_engine *e, pt_run *r, uint32_t il, uint32_t n, uint32_t g0) {
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    pt_layer_buf *b = &r->lb;
    const uint32_t E = DS4_N_EMBD, HD = DS4_N_HEAD_DIM, ratio = ds4_layer_compress_ratio(il), ng = n / ratio;
    if (!ng) return true;
    ds4_gpu_tensor *v[4] = { pt_rows(b->xna, 0, n, E), pt_rows(b->g_xna, 0, n, E), pt_rows(b->gcomp[il], g0, ng, HD), pt_rows(b->posg[il], g0, ng, 1) };
    ds4_gpu_tensor *xna = v[0], *gxna = v[1], *gc = v[2], *posg = v[3];
    bool ok = xna && gxna && gc && posg;
    if (ok && ratio > 1u) ok = ds4_gpu_v41_matmul_bf16_tensor(b->cckv, m->map, m->size, l->attn_compressor_kv->abs_offset, E, HD, xna, n) &&
                               ds4_gpu_v41_matmul_bf16_tensor(b->ccsc, m->map, m->size, l->attn_compressor_gate->abs_offset, E, HD, xna, n) &&
                               ds4_gpu_v41_compress_pool_tensor(b->cpool, b->cckv, b->ccsc, n, ratio, HD);
    else if (ok) ok = ds4_gpu_v41_matmul_bf16_tensor(b->cpool, m->map, m->size, l->attn_compressor_kv->abs_offset, E, HD, xna, n) &&
                      ds4_gpu_v41_round_bf16_tensor(b->cpool, (uint64_t)n * HD);
    ok = ok && ds4_gpu_bwd_rope_tensor(gc, posg, ng, 1, HD, DS4_N_ROT, DS4_COMPRESS_ROPE_FREQ_BASE, (uint32_t)DS4_ROPE_ORIG_CTX,
                                       DS4_ROPE_SCALE_FACTOR, DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, true) &&
         ds4_gpu_bwd_rms_norm_tensor(b->cgpool, gc, b->cpool, m->map, m->size, l->attn_compressor_norm->abs_offset, HD, ng, DS4_RMS_EPS, 0);
    if (ratio > 1u) ok = ok && ds4_gpu_tensor_fill_f32(b->cgkv, 0.f, (uint64_t)n * HD) && ds4_gpu_tensor_fill_f32(b->cgsc, 0.f, (uint64_t)n * HD) &&
                         ds4_gpu_bwd_compress_pool_tensor(b->cgkv, b->cgsc, b->cgpool, b->cckv, b->ccsc, ng, ratio, HD) &&
                         RB_MT(gxna, l->attn_compressor_kv, E, HD, b->cgkv, 1) && RB_MT(gxna, l->attn_compressor_gate, E, HD, b->cgsc, 1);
    else ok = ok && RB_MT(gxna, l->attn_compressor_kv, E, HD, b->cgpool, 1);
    rb_free_views(v, 4);
    if (!ok) fprintf(stderr, "ds4: [restore-b] L%u 压缩器反向失败(ratio %u, %u 行, %u 组)\n", il, ratio, n, ng);
    return ok;
}

/* = pt_layer_bwd(单题路), 多一处: g_y 算出来之后、放大器反传之前调 rb_aux_layer(as 为 NULL 时与 pt_layer_bwd 同一串核) */
static bool rb_layer_bwd(ds4_engine *e, const pt_cfg *c, pt_run *r, uint32_t il, uint32_t n, rb_auxstep *as, rb_route *rt) {
    ds4_v41_state *st = &r->st; const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    pt_layer_buf *b = &r->lb;
    const uint32_t E = DS4_N_EMBD, HC = DS4_N_HC, HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD, Q = DS4_N_LORA_Q, FF = DS4_N_FF_EXP;
    const uint32_t G = DS4_N_OUT_GROUP, LOW = DS4_N_OUT_GROUP * DS4_N_LORA_O, ratio = ds4_layer_compress_ratio(il);
    const float theta = ratio ? DS4_COMPRESS_ROPE_FREQ_BASE : DS4_ROPE_FREQ_BASE;
    const uint32_t osl = ratio ? (uint32_t)DS4_ROPE_ORIG_CTX : 0u;
    v41_tsave *keep = st->tsave;
    st->tsave = NULL;
    bool ok = rb_layer_recompute(e, r, il, n);
    /* ---- ffn 半层 ---- */
    if (ok) ok = ds4_gpu_tensor_fill_f32(b->g_hcm, 0.f, (uint64_t)n * HC * E) &&
                 ds4_gpu_bwd_hc_post_tensor(b->g_y, b->g_hcm, b->g_postf, b->g_combf, r->ghc, st->y, st->hc, b->postf, b->combf, E, HC, n) &&
                 ds4_gpu_tensor_fill_f32(b->g_xnf, 0.f, (uint64_t)n * E);
    if (ok && as) ok = rb_aux_layer(as, c, r, il, b->g_y, b->xnf);   /* 逐层靶: 下游对本层输出的敏感度此刻就在 g_y 里 */
    if (ok && r->gA[il] && st->ampS[il]) {   /* 带路由: 本层路由 st->sel/rw 是重算刚出的(与前向同一份); rw 当常数, 放大器对它的梯度不回传给路由 */
        ok = rt && rt->gS[il] &&
             ds4_gpu_bwd_amp_route_tensor(r->gA[il], r->gB[il], rt->gS[il], b->g_xnf, b->g_y, b->xnf, st->ampA[il], st->ampB[il], st->ampS[il],
                                          st->sel, st->rw, r->T, r->gT, n, E, r->K, DS4_N_EXPERT_USED, DS4_N_EXPERT);
        if (!ok) fprintf(stderr, "ds4: ★[restore-b] L%u 带路由放大器反向失败%s★\n", il, rt ? "" : "(没给 rb_route)");
    } else if (ok && r->gA[il]) ok = ds4_gpu_bwd_amp_tensor(r->gA[il], r->gB[il], b->g_xnf, b->g_y, b->xnf, st->ampA[il], st->ampB[il], r->T, r->gT, n, E, r->K);
    if (ok) ok = RB_MT(b->g_sh, l->ffn_down_shexp, FF, E, b->g_y, 0) &&
                 ds4_gpu_bwd_swiglu_tensor(b->g_sg, b->g_su, b->g_sh, st->sg, st->su, (uint64_t)n * FF, DS4_SWIGLU_CLAMP_EXP) &&
                 RB_MT(b->g_xnf, l->ffn_gate_shexp, E, FF, b->g_sg, 1) && RB_MT(b->g_xnf, l->ffn_up_shexp, E, FF, b->g_su, 1);
    if (ok) {
        char nm[64]; snprintf(nm, sizeof nm, "blk.%u.ffn_exps_vq.blob", il);
        const ds4_tensor *blob = model_find_tensor(m, nm);
        ok = blob && ds4_gpu_bwd_routed_moe_tensor(b->g_xnf, b->g_rw, b->g_y, b->xnf, st->sel, st->rw, m->map, m->size, blob->abs_offset, blob->bytes,
                                                   E, FF, E, DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, il, n) &&
             ds4_gpu_bwd_router_tensor(b->g_glog, b->g_rw, st->sel, st->glog, n, DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_EXPERT_WEIGHT_SCALE) &&
             RB_MT(b->g_xnf, l->ffn_gate_inp, E, DS4_N_EXPERT, b->g_glog, 1);
    }
    if (ok) ok = ds4_gpu_bwd_rms_norm_tensor(b->g_xf, b->g_xnf, b->xf, m->map, m->size, l->ffn_norm->abs_offset, E, n, DS4_RMS_EPS, 0) &&
                 ds4_gpu_bwd_hc_pre_tensor(b->g_hcm, b->g_pre_a, b->g_xf, st->hc, b->prea, E, HC, n) &&
                 ds4_gpu_bwd_hc_mix_tensor(b->g_hcm, r->gpre, b->g_postf, b->g_combf, st->hc, b->mixf, m->map, m->size, l->hc_ffn_fn->abs_offset,
                                           l->hc_ffn_scale->abs_offset, l->hc_ffn_base->abs_offset, E, HC, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS, DS4_RMS_EPS, n);
    /* ---- 注意力半层 ---- */
    if (ok) ok = ds4_gpu_tensor_fill_f32(b->g_hcin, 0.f, (uint64_t)n * HC * E) &&
                 ds4_gpu_bwd_hc_post_tensor(b->g_attn, b->g_hcin, b->g_posta, b->g_comba, b->g_hcm, st->attn_out, b->hcin, b->posta, b->comba, E, HC, n) &&
                 RB_MT(b->g_low, l->attn_output_b, LOW, E, b->g_attn, 0) &&
                 ds4_gpu_bwd_grouped_matmul_t_tensor(b->g_o, m->map, m->size, l->attn_output_a->type, l->attn_output_a->abs_offset, G,
                                                     (uint64_t)(NH / G) * HD, DS4_N_LORA_O, b->g_low, n, 0) &&
                 ds4_gpu_bwd_rope_tensor(b->g_o, st->pos, n, NH, HD, DS4_N_ROT, theta, osl, DS4_ROPE_SCALE_FACTOR, DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, false);
    if (ok) {
        const int16_t src = ratio ? g_ds4_v41.kv_source_of[il] : -1;
        const bool wantc = src >= 0 && (uint32_t)src >= c->layer_lo && r->lb.gcomp[src];
        rb_seg g; rb_seg_of(r, il, n, &g);
        const bool hasc = g.idx != NULL, gcw = hasc && wantc;
        ds4_gpu_tensor *gcv = gcw ? pt_rows(b->gcomp[src], 0, g.ng, HD) : NULL;
        ok = (!(ratio && g.ng && g.topk) || hasc) && (!gcw || gcv) &&
             ds4_gpu_bwd_sparse_attn_tensor(b->g_q, b->g_kvn, gcv, b->g_o, b->o, st->q, g.win, hasc ? g.comp : NULL, g.idx, m->map, m->size,
                                            l->attn_sinks->abs_offset, n, r->pfx, DS4_N_SWA, g.ng, hasc ? g.topk : 0u, NH, HD, (float)(1.0 / sqrt((double)HD)));
        if (gcv) ds4_gpu_tensor_free(gcv);
        if (g.idx) ds4_gpu_tensor_free(g.idx);
    }
    if (ok) ok = ds4_gpu_bwd_rope_tensor(b->g_q, st->pos, n, NH, HD, DS4_N_ROT, theta, osl, DS4_ROPE_SCALE_FACTOR, DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, true) &&
                 RB_MT(b->g_qrn, l->attn_q_b, Q, (uint64_t)NH * HD, b->g_q, 0) &&
                 ds4_gpu_bwd_rms_norm_tensor(b->g_qr, b->g_qrn, b->qr, m->map, m->size, l->attn_q_a_norm->abs_offset, Q, n, DS4_RMS_EPS, 0) &&
                 RB_MT(b->g_xna, l->attn_q_a, E, Q, b->g_qr, 0);
    if (ok) ok = ds4_gpu_bwd_rope_tensor(b->g_kvn, st->pos, n, 1, HD, DS4_N_ROT, theta, osl, DS4_ROPE_SCALE_FACTOR, DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, true) &&
                 ds4_gpu_bwd_rms_norm_tensor(b->g_kv, b->g_kvn, b->kv, m->map, m->size, l->attn_kv_a_norm->abs_offset, HD, n, DS4_RMS_EPS, 0) &&
                 RB_MT(b->g_xna, l->attn_kv, E, HD, b->g_kv, 1);
    if (ok && b->gcomp[il]) ok = rb_compress_bwd(e, r, il, n, r->pfx / ratio);   /* 前缀形态: 只对尾窗自己的组求导, 前缀组的梯度丢(常量) */
    if (ok) ok = ds4_gpu_bwd_rms_norm_tensor(b->g_xa, b->g_xna, b->xa, m->map, m->size, l->attn_norm->abs_offset, E, n, DS4_RMS_EPS, 0) &&
                 ds4_gpu_bwd_hc_pre_tensor(b->g_hcin, b->g_prein, b->g_xa, b->hcin, r->save.pm_in[il], E, HC, n) &&
                 ds4_gpu_bwd_hc_mix_tensor(b->g_hcin, b->g_pre_a, b->g_posta, b->g_comba, b->hcin, b->mixa, m->map, m->size, l->hc_attn_fn->abs_offset,
                                           l->hc_attn_scale->abs_offset, l->hc_attn_base->abs_offset, E, HC, DS4_N_HC_SINKHORN_ITER, DS4_HC_EPS, DS4_RMS_EPS, n);
    if (ok && rb_has_engram(st, il))
        ok = ds4_gpu_bwd_engram_gate_tensor(b->g_hcin, b->hpre, b->ekvf, m->map, m->size, l->engram_q->abs_offset, l->engram_k->abs_offset, E, HC, n, DS4_RMS_EPS);
    if (ok) ok = RB_CP(r->ghc, b->g_hcin, (uint64_t)n * HC * E * 4) && RB_CP(r->gpre, b->g_prein, (uint64_t)n * HC * 4);
    st->tsave = keep;
    if (!ok) fprintf(stderr, "ds4: [restore-b] L%u 逐层反传失败(n %u)\n", il, n);
    return ok;
}

bool rb_layers_bwd(ds4_engine *e, const pt_cfg *c, pt_run *r, uint32_t n, rb_auxstep *as, rb_route *rt) {
    if (r->pk.nb) { fprintf(stderr, "ds4: ★[restore-b] 逐层反传只走单题路(nopack), 却遇到合批态★\n"); return false; }
    for (uint32_t il = c->layer_lo; il <= c->layer_hi; il++)
        if (r->lb.gcomp[il] && !ds4_gpu_tensor_fill_f32(r->lb.gcomp[il], 0.f, ds4_gpu_tensor_bytes(r->lb.gcomp[il]) / 4u)) return false;
    for (uint32_t il = c->layer_hi + 1u; il-- > c->layer_lo; ) {
        (void)ds4_gpu_bwd_wcache(1);   /* 层内 bf16 权重缓存: 只在这一层的"重算 + 反传"里开(与 pt_layers_bwd 同) */
        const bool lok = rb_layer_bwd(e, c, r, il, n, as, rt);
        (void)ds4_gpu_bwd_wcache(0);
        if (!lok) return false;
    }
    return true;
}

bool rb_route_alloc(rb_route *rt, const pt_cfg *c, pt_run *r, const char *init) {
    const uint32_t NE = DS4_N_EXPERT, K = r->K;
    const uint64_t nel = (uint64_t)NE * K;
    float *buf = xmalloc_zeroed(nel, 4);
    bool ok = true; uint32_t nload = 0;
    for (uint32_t il = c->layer_lo; il <= c->layer_hi && ok; il++) {
        if (r->st.ampS[il]) { fprintf(stderr, "ds4: ★[restore-b] L%u 训练态已有路由系数★\n", il); ok = false; break; }
        memset(buf, 0, nel * 4);
        char p[1200]; snprintf(p, sizeof p, "%s/ampr_L%02u.bin", init ? init : "", il);
        FILE *f = init && init[0] ? fopen(p, "rb") : NULL;
        if (f) {   /* 起点带路由: 秩一致才读(扩秩时多出的秩系数 0) */
            int32_t hd[3] = { 0, 0, 0 };
            ok = fread(hd, 4, 3, f) == 3 && hd[0] == (int32_t)NE && hd[1] > 0 && (uint32_t)hd[1] <= K && hd[2] == 1;
            for (uint32_t e2 = 0; ok && e2 < NE; e2++) ok = fread(buf + (uint64_t)e2 * K, 4, (size_t)hd[1], f) == (size_t)hd[1];
            fclose(f);
            if (!ok) fprintf(stderr, "ds4: ★[restore-b] 起点 %s 对不上★\n", p);
            nload++;
        }
        r->st.ampS[il] = v41_alloc(nel * 4, &ok); rt->gS[il] = v41_alloc(nel * 4, &ok); rt->mS[il] = v41_alloc(nel * 4, &ok); rt->vS[il] = v41_alloc(nel * 4, &ok);
        if (ok) ok = ds4_gpu_tensor_write(r->st.ampS[il], 0, buf, nel * 4) && ds4_gpu_tensor_fill_f32(rt->gS[il], 0.f, nel) &&
                     ds4_gpu_tensor_fill_f32(rt->mS[il], 0.f, nel) && ds4_gpu_tensor_fill_f32(rt->vS[il], 0.f, nel);
    }
    free(buf);
    rt->on = ok;
    if (ok) fprintf(stderr, "ds4: [restore-b] 带路由: 每层 %u 专家 × 秩 %u 系数, lr %.3g, 起点 %s\n", NE, K, (double)rt->lr, nload ? "读自 rb_init" : "全 0(= 现役段 B)");
    return ok;
}

bool rb_route_step(rb_route *rt, const pt_cfg *c, pt_run *r, uint32_t step, double *gn) {
    const uint64_t nel = (uint64_t)DS4_N_EXPERT * r->K;
    double ss = 0, q = 0;
    for (uint32_t il = c->layer_lo; il <= c->layer_hi; il++) {
        if (!ds4_gpu_bwd_sumsq_tensor(rt->gS[il], nel, &q)) return false;
        ss += q;
        if (!ds4_gpu_bwd_adam_tensor(r->st.ampS[il], rt->gS[il], rt->mS[il], rt->vS[il], nel, rt->lr, 0.9f, 0.999f, 1e-8f, 1.0f, step)) return false;
    }
    *gn = sqrt(ss);
    return true;
}

bool rb_route_save(const pt_cfg *c, const pt_run *r, const char *dir) {
    const uint32_t NE = DS4_N_EXPERT, K = r->K;
    const uint64_t nel = (uint64_t)NE * K;
    float *buf = xmalloc(nel * 4);
    bool ok = true;
    for (uint32_t il = c->layer_lo; il <= c->layer_hi && ok; il++) {
        char p[1200]; snprintf(p, sizeof p, "%s/ampr_L%02u.bin", dir, il);
        FILE *f = fopen(p, "wb");
        const int32_t hd[3] = { (int32_t)NE, (int32_t)K, 1 };
        ok = f && ds4_gpu_tensor_read(r->st.ampS[il], 0, buf, nel * 4) && fwrite(hd, 4, 3, f) == 3 && fwrite(buf, 4, nel, f) == nel;
        if (f) fclose(f);
        if (!ok) fprintf(stderr, "ds4: ★[restore-b] 写不了 %s★\n", p);
    }
    free(buf);
    return ok;
}

void rb_route_free(rb_route *rt) {
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor **t[3] = { &rt->gS[il], &rt->mS[il], &rt->vS[il] };
        for (int k = 0; k < 3; k++) if (*t[k]) { ds4_gpu_tensor_free(*t[k]); *t[k] = NULL; }
    }
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_restoreb_bwd_nonempty_tu;
