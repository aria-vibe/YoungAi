/* core_restoreb.c — 反修 B 段的还原训练(2026-10-28, ./ds4 --restore-b <配置>)。
 *
 * 为什么另写一份(用户 10-28): "反修 b 段不是新的后训练, 不是把官方的语料再学一遍; 本质还是量化能力还原, 只是形式用了后训练的方式,
 *   要找到以前能量的还原点"; "应该写一份新的反修 b 段的代码, 后训练代码不要动, 那个功能有用的"。
 *   ⇒ 后训练 ③(core_ptrain*.c)一行不改, 这里只调它对外的件: 装料 pt_data_load / 官方 API 教师表 pt_teacher_fp / 训练态 pt_run_alloc /
 *   预热 pt_pack_warm / Adam pt_opt_step / 落盘 pt_save_amp; 逐层反传 10-29 起走自己的一份(core_restoreb_bwd.c, 要在每层 g_y 上接逐层靶)。
 * 和 ③ 不一样的地方:
 *   ① 缺口加权(rb_gap=Q, 0~1): 第 0 步参数下逐位算料的答案位对教师(官方 top-20)的 KL, 全料 Q 分位以上的位权重 1, 其余位权重 rb_rest(缺省 0, 只为跟 10-28 a009~a011 能对上)。
 *      为什么: 10-28 a002~a008 七轮, 训练料自己的 KLD 每轮降 9~17%, 编程尺却在所有位置上均匀地退一点 —— 学生本来就和原件一样的位
 *      (官方自己的文字上中位 KL 0.0025)每位都在出梯度, 合起来就是拉着段 B 复述官方文本; 量化丢掉的能量集中在 KL 大的那一小撮位上。
 *      秩 64 → 128 同一份料同一门(编程 +1.7 SE → +2.1 SE)证明不是容量问题, 是目标问题。
 *   ② 料全训: ③ 的五取一留出清掉(门在外面: 编程尺 + wt2; 10-28 a003 的留出抽走了计划里唯一一条 shell 轨迹, 不训也不评)。
 *   ③ 起点 rb_init=<目录> 的秩可以比本趟 rank 小: 前 K0 秩原样读入, 多出的秩 A=0、B 随机(y += x·B·A, 起点与现役逐位同)。
 *   ④ 逐层靶(10-29, core_restoreb_aux.c): rb_dump=<目录> 只取料不训(当前链把料里最长那条整条前向, 每层 x / 放大器前的 y 落盘, 给 FP 原件 --force-x);
 *      rb_aux=<同目录> 训练损失加 w·(1/层数)·Σ_l ‖xn·B·A − t_l‖²/‖t_l‖²(t = y_fp(x_q) − y_q), rb_auxw = w 相对出口 KL 的梯度范数(第 1 步定标:
 *      rb_auxw=1 ⇒ 第 1 步两份梯度一样大; 0 ⇒ 只量不训, 当对照)。为什么见 core_restoreb_aux.c 头注释。
 *   ⑤ 带路由(10-29, rb_route=1, rb_route_lr=<逐专家系数的 lr>): 段 B 变成 y += ((xn·B) ⊙ u)·A, u = 1 + Σ_j rw_j·S[e_j] —— 每个专家一组秩系数,
 *      为什么见 src/cuda/cuda_amp_route.inc.cu 头注释。S 从 0 起(= 现役段 B), ckpt 多落 ampr_Lnn.bin。不与 ④ 同开(逐层项按不带路由的式子算)。
 *   ⑥ 位置掩码(10-29, rb_mask=<文件>, 每行 "s e" = 目标位置 [s, e), 位置按样本序列的下标、BOS 是 0): 只有目标落在掩码里的答案位出梯度, 其余位权重 0。
 *      为什么: 天花板那把留出是"没见过的 15 种语言", 混了换语言与换文档; 判决份每种语言 4 个窗(窗间在源语料里是跳着取的),
 *      按窗练一半、留一半, 就是"同语言、没见过的上下文"能迁多少(restore_B_ceiling_spark.sh --winmask)。不与 rb_gap 同开。
 *   ⑦ 冻秩(10-29, rb_freeze=<n>): 前 n 个秩(起点 rb_init 读进来的那份)每步更新前梯度清零 ⇒ Adam 更新量恒 0, 与起点逐位同; 只训多出来的秩。
 *      为什么: 要量"参数少了是不是就背不下、只能学跨上下文的修正", 又不能丢掉现役段 B 已有的修正 —— 现役秩 64 冻住, 上面加 8 个秩训(rank=72)。
 * 配置 = ③ 的 key=value(data / out / rank / lr / batch / topk / maxlen / ctxlen / teacher=fp:<表> 同义)+ 本模块的 rb_gap / rb_rest / rb_init / rb_dump / rb_aux / rb_auxw;
 *   本模块的几项摘出来, 其余原样写成 <out>/pt.cfg 交给 pt_cfg_load。teacher=fp: 必须给(教师只认官方表), init= 不许给(用 rb_init)。
 * 产物与 ③ 同格式: <out>/ckpt_e01/amp_Lnn.bin + base.fnv; <out>/train.log 的 "step 0 … train_kl" / "step i … loss … gnorm" / "epoch 1 … train_kl"
 *   与 ③ 同字段(train_kl = 全料答案位逐位平均 KL, 不加权), 另有一行 "gap …" 记阈值与选中的位数。
 * 出错会怎样: 配置/料/教师表/起点任一不对就停车返回 1, 不落盘; 训练中可用内存跌破 RB_MEM_FLOOR_MB(3000 MB)停车(看门狗之前)。 */
#include "core_restoreb.h"
#include "../common/ds4_gr_fnv.h"
#include <sys/stat.h>
#ifndef DS4_NO_GPU

#define RB_MEM_FLOOR_MB 3000   /* 与 ③ 的 PT_MEM_FLOOR_MB 同值: 看门狗 2500 MB 动手, 多留 500 MB 给评估的零碎 */

typedef struct {
    float gap;               /* rb_gap: 缺口分位; 0 = 不加权(每位都出梯度, 与 ③ 的损失同式) */
    float rest;              /* rb_rest: 缺口以外的位的权重(0~1, 缺省 0); 为什么要它见 rb_binarize */
    char init[1024];         /* rb_init: 起点段 B 目录(amp_Lnn.bin + base.fnv); 空 = A=0 从零起 */
    char dump[1024], aux[1024];   /* rb_dump: 只取料; rb_aux: 逐层靶目录(见头注释 ④) */
    float auxw;              /* rb_auxw: 逐层项相对出口 KL 的梯度范数(第 1 步定标); 0 = 只量 */
    int route; float route_lr;   /* rb_route / rb_route_lr: 带路由(见头注释 ⑤) */
    char mask[1024];         /* rb_mask: 位置掩码文件(见头注释 ⑥) */
    int freeze;              /* rb_freeze: 冻住前几个秩(见头注释 ⑦); 0 = 不冻 */
} rb_cfg;

/* 配置分两半: rb_* 自己收, 其余写 <out>/pt.cfg 交 pt_cfg_load(③ 不认的键它会拒, 所以不能整份交过去) */
static bool rb_cfg_load(const char *spec, rb_cfg *b, pt_cfg *c, char *ptpath, size_t ptcap) {
    memset(b, 0, sizeof *b);
    FILE *f = fopen(spec, "r");
    if (!f) { fprintf(stderr, "ds4: --restore-b 读不了配置 %s\n", spec); return false; }
    char ln[4096], out[1024] = "", *rest = NULL; size_t nrest = 0;
    FILE *mem = open_memstream(&rest, &nrest);
    bool ok = mem != NULL;
    while (ok && fgets(ln, sizeof ln, f)) {
        char *p = ln; while (*p == ' ' || *p == '\t') p++;
        if (!strncmp(p, "rb_gap=", 7)) { b->gap = (float)atof(p + 7); continue; }
        if (!strncmp(p, "rb_rest=", 8)) { b->rest = (float)atof(p + 8); continue; }
        if (!strncmp(p, "rb_init=", 8)) { snprintf(b->init, sizeof b->init, "%s", p + 8); b->init[strcspn(b->init, "\r\n")] = 0; continue; }
        if (!strncmp(p, "rb_dump=", 8)) { snprintf(b->dump, sizeof b->dump, "%s", p + 8); b->dump[strcspn(b->dump, "\r\n")] = 0; continue; }
        if (!strncmp(p, "rb_aux=", 7)) { snprintf(b->aux, sizeof b->aux, "%s", p + 7); b->aux[strcspn(b->aux, "\r\n")] = 0; continue; }
        if (!strncmp(p, "rb_auxw=", 8)) { b->auxw = (float)atof(p + 8); continue; }
        if (!strncmp(p, "rb_route=", 9)) { b->route = atoi(p + 9); continue; }
        if (!strncmp(p, "rb_route_lr=", 12)) { b->route_lr = (float)atof(p + 12); continue; }
        if (!strncmp(p, "rb_freeze=", 10)) { b->freeze = atoi(p + 10); continue; }
        if (!strncmp(p, "rb_mask=", 8)) { snprintf(b->mask, sizeof b->mask, "%s", p + 8); b->mask[strcspn(b->mask, "\r\n")] = 0; continue; }
        if (!strncmp(p, "init=", 5)) { fprintf(stderr, "ds4: --restore-b 不收 init=(起点用 rb_init=, 允许扩秩)\n"); ok = false; break; }
        if (!strncmp(p, "out=", 4)) { snprintf(out, sizeof out, "%s", p + 4); out[strcspn(out, "\r\n")] = 0; }
        fputs(ln, mem);
    }
    fclose(f);
    if (mem) fclose(mem);
    if (ok && (!out[0] || b->gap < 0.f || b->gap >= 1.f || b->rest < 0.f || b->rest > 1.f || b->auxw < 0.f || (b->route && (b->route_lr <= 0.f || b->aux[0])) || (b->mask[0] && b->gap > 0.f))) {
        fprintf(stderr, "ds4: --restore-b 配置缺 out= / rb_gap 不在 [0,1) / rb_rest 不在 [0,1] / rb_auxw < 0 / rb_route 没给 rb_route_lr 或与 rb_aux 同开 / rb_mask 与 rb_gap 同开\n"); ok = false; }
    if (ok) {
        mkdir(out, 0755);
        snprintf(ptpath, ptcap, "%s/pt.cfg", out);
        FILE *g = fopen(ptpath, "w");
        ok = g && fwrite(rest, 1, nrest, g) == nrest;
        if (g) fclose(g);
        if (!ok) fprintf(stderr, "ds4: --restore-b 写不了 %s\n", ptpath);
    }
    free(rest);
    if (ok) ok = pt_cfg_load(c, ptpath);
    if (ok && (b->freeze < 0 || (uint32_t)b->freeze >= c->rank || (b->freeze && !b->init[0]))) { fprintf(stderr, "ds4: --restore-b rb_freeze 要 0 ≤ n < rank 且给了 rb_init(冻的是起点那几个秩)\n"); ok = false; }
    if (ok && !c->teacher_fp[0]) { fprintf(stderr, "ds4: --restore-b 要 teacher=fp:<官方表>(还原的靶只认官方分布)\n"); ok = false; }
    if (ok && c->layer_lo >= DS4_N_LAYER - 1u) { fprintf(stderr, "ds4: --restore-b 只接多层(layers 不能只有末层)\n"); ok = false; }
    return ok;
}

/* 起点: 秩 K0 ≤ 本趟 K。A 的多出行置 0、B 的多出行保留 pt_run_alloc 给的随机值 ⇒ 起点输出与起点目录逐位同 */
static bool rb_load_init(const pt_cfg *c, pt_run *r, const char *dir) {
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

/* 一个样本答案位的 KL(与 pt_loss_rows 同一个核、同一个缩放), 多一个逐位权重:
 * mode 0 = 不加权; 1 = 不加权并把逐位 KL 记进 gw[off..](第 0 步定缺口); 2 = 按 gw 加权(训练步) */
static bool rb_loss(const pt_cfg *c, pt_run *r, ds4_gpu_tensor *w, ds4_gpu_tensor *logits, uint32_t row, const pt_sample *s, uint32_t off,
                    float *gw, int mode, double *loss, uint32_t *ntok, ds4_gpu_tensor **glog) {
    const uint32_t m = s->m - off, V = DS4_N_VOCAB, TK = c->topk;
    const float scale = 1.0f / ((float)m * (float)c->batch);
    const bool wt = mode == 2 && gw;
    *glog = pt_rows(logits, row, m, V);
    bool ok = *glog && ds4_gpu_tensor_write(r->tid, 0, s->top_id + (size_t)off * TK, (uint64_t)m * TK * 4) &&
              ds4_gpu_tensor_write(r->tp, 0, s->top_p + (size_t)off * TK, (uint64_t)m * TK * 4) &&
              ds4_gpu_tensor_write(r->trest, 0, s->top_rest + off, (uint64_t)m * 4) &&
              (!wt || ds4_gpu_tensor_write(w, 0, gw + off, (uint64_t)m * 4)) &&
              ds4_gpu_bwd_kl_topk_tensor(*glog, r->loss, logits, row, m, V, r->tid, r->tp, r->trest, wt ? w : NULL, TK, scale) &&
              ds4_gpu_tensor_read(r->loss, 0, r->loss_h, (uint64_t)m * 4);
    if (ok) { double l = 0; for (uint32_t k = 0; k < m; k++) l += r->loss_h[k]; *loss += l; *ntok += m; }
    if (ok && mode == 1) for (uint32_t k = 0; k < m; k++) gw[off + k] = r->loss_h[k];
    return ok;
}

/* 一个样本的前向 + 损失(+ 反传): pt_step_sample 的多层路照抄(前缀形态、按篇接力、环头快照同式), 只把损失换成 rb_loss。
 * 为什么照抄不调用: pt_step_sample 里的损失没有逐位权重口, 而 ③ 的代码不动 */
static bool rb_step(ds4_engine *e, const pt_cfg *c, pt_run *r, ds4_gpu_tensor *w, const pt_sample *s, float *gw, int mode, int grad,
                    double *loss_sum, uint32_t *ntok, rb_auxstep *as, rb_route *rt) {
    ds4_v41_state *st = &r->st;
    const ds4_model *mdl = &e->model;
    const uint32_t E = DS4_N_EMBD, HC = DS4_N_HC, V = DS4_N_VOCAB, cap = st->cap_tok;
    if (as) as->m = 0;
    if (!pt_runnable(r, s)) return true;
    uint32_t pfx = 0, n = s->sn;
    const uint32_t f0 = s->sa0 + s->l0;
    if (s->sn > cap) {
        const uint32_t A = pt_pfx_align(), lo = (s->sn - cap + A - 1u) / A * A, hi = (f0 - 1u) / A * A;
        pfx = hi > lo ? hi : lo;
        n = s->sn - pfx;
    }
    r->pfx = pfx;
    const uint32_t a0 = f0 > pfx + 1u ? f0 : pfx + 1u;
    if (a0 > s->sa0 + s->m - 1u) return true;
    const uint32_t off = a0 - s->sa0, m = s->m - off, r0 = a0 - 1u - pfx;
    /* 逐层靶: 答案位 [a0, a0+m) 由位置 [a0−1, a0−1+m)(窗口第 r0 行起)的隐状态预测, 取后者; 样本不是靶序列的前缀就只走 KL */
    if (as && as->ax && rb_aux_covers(as->ax, s)) { as->r0 = r0; as->m = m; as->p0 = a0 - 1u; }
    const bool cont = pfx && r->cont_s && r->cont_s->sn == pfx && st->n_past == pfx && !memcmp(r->cont_s->sids, s->sids, (size_t)pfx * 4);
    r->cont_s = NULL;
    if (!cont) pt_state_reset(st);
    st->ced_skip = 0; st->head_last_only = 0;
    if (pfx && !cont) {   /* 前缀块: 不存档、出口只算末行、解码器回放关掉(训练的上下文每层每位都算满, 见 pt_step_sample) */
        v41_tsave *keep = st->tsave;
        const int rp = g_ds4_v41_replay;
        g_ds4_v41_replay = 0;
        st->tsave = NULL; st->head_last_only = 1;
        bool pok = true;
        for (uint32_t c0 = 0; pok && c0 < pfx; ) { const uint32_t nc = pfx - c0 < cap ? pfx - c0 : cap; pok = v41_forward(e, st, s->sids + c0, nc); c0 += nc; }
        g_ds4_v41_replay = rp;
        st->tsave = keep; st->head_last_only = 0;
        if (!pok) return false;
    }
    const uint64_t ringb = (uint64_t)DS4_N_SWA * DS4_N_HEAD_DIM * 4;
    if (pfx) for (uint32_t il = 0; il < DS4_N_LAYER; il++)
        if (r->ring_snap[il] && !ds4_gpu_tensor_copy(r->ring_snap[il], 0, st->win[il], 0, ringb)) return false;
    if (!v41_forward(e, st, s->sids + pfx, n)) return false;
    if (pfx) for (uint32_t il = 0; il < DS4_N_LAYER; il++)
        if (r->ring_post[il] && !ds4_gpu_tensor_copy(r->ring_post[il], 0, st->win[il], 0, ringb)) return false;
    ds4_gpu_tensor *glog = NULL;
    bool ok = rb_loss(c, r, w, st->logits, r0, s, off, gw, mode, loss_sum, ntok, &glog);
    if (ok && grad) {   /* 出口梯度铺回整段 → RMSNorm → hc_pre → 逐层往下(③ 的逐层反传原样调用) */
        ds4_gpu_tensor *gv = pt_rows(r->gxn, r0, m, E);
        ok = gv && ds4_gpu_tensor_fill_f32(r->gxn, 0.f, (uint64_t)n * E) &&
             ds4_gpu_bwd_matmul_t_tensor(gv, mdl->map, mdl->size, e->weights.output->type, e->weights.output->abs_offset, E, V, glog, m, 0) &&
             ds4_gpu_bwd_rms_norm_tensor(r->gx, r->gxn, st->x, mdl->map, mdl->size, e->weights.output_norm->abs_offset, E, n, DS4_RMS_EPS, 0) &&
             ds4_gpu_tensor_fill_f32(r->ghc, 0.f, (uint64_t)n * HC * E) &&
             ds4_gpu_bwd_hc_pre_tensor(r->ghc, r->gpre, r->gx, st->hc, st->pre_mix, E, HC, n);
        if (ok) ok = rb_layers_bwd(e, c, r, n, as && as->m ? as : NULL, rt && rt->on ? rt : NULL);
        if (gv) ds4_gpu_tensor_free(gv);
    }
    if (glog) ds4_gpu_tensor_free(glog);
    if (!ok) fprintf(stderr, "ds4: [restore-b] 样本前向/反传失败(长 %u, 答案 %u 行)\n", s->sn, m);
    if (ok && pfx) for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++)
        if (r->ring_post[il]) ok = ds4_gpu_tensor_copy(st->win[il], 0, r->ring_post[il], 0, ringb) != 0;
    if (ok && c->ctxlen > c->maxlen && st->n_past == s->sn) r->cont_s = s;   /* 状态是整段 [0, sn): 留给同篇下一段接力 */
    return ok;
}

/* 全料过一遍(不反传): 返回答案位逐位平均 KL; mode 1 时顺手把逐位 KL 记进各题的 gw */
static bool rb_pass(ds4_engine *e, const pt_cfg *c, pt_run *r, ds4_gpu_tensor *w, const pt_data *d, const uint32_t *ord, uint32_t n,
                    float **gw, int mode, double *kl) {
    double ls = 0; uint32_t nt = 0;
    r->cont_s = NULL;
    for (uint32_t i = 0; i < n; i++)
        if (!rb_step(e, c, r, w, &d->s[ord[i]], gw[ord[i]], mode, 0, &ls, &nt, NULL, NULL)) return false;
    *kl = nt ? ls / nt : 0.0;
    return true;
}

static int rb_cmpf(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }

/* 第 0 步的逐位 KL → 权重: 阈值 = 全料逐位 KL 的 gap 分位, 以上 1, 以下 rest; 打一行账(选中几位、这些位的 KL 占总量多少)。
 * rest 为什么不能是 0(10-28 a010/a011): 缺口位确实修好了(编程尺现役 KL 前 3% 的位 a010 降 0.040), 但其余 90% 的位没梯度钉着、
 *   一起漂坏(编程尺 0~90% 分位每位 +0.002~+0.004; 没练过的官方料上同样 +6.6%), 编程尺上难位只占 KL 的一半, 合起来净退。
 *   同一份 a011 料 rest 0 / 0.3 / 1: 编程尺 +2.9 / +1.4 / −0.1 SE, 最难 3% 的位反而 rest=1 修得最多(−0.025) ⇒ 脚本缺省已回 rb_gap=0(不加权)。 */
static void rb_binarize(const pt_data *d, const uint32_t *ord, uint32_t n, float **gw, float gap, float rest, FILE *lf) {
    uint64_t tot = 0;
    for (uint32_t i = 0; i < n; i++) if (gw[ord[i]]) tot += d->s[ord[i]].m;
    if (!tot) return;
    float *all = xmalloc((size_t)tot * 4); uint64_t k = 0;
    for (uint32_t i = 0; i < n; i++) if (gw[ord[i]]) { memcpy(all + k, gw[ord[i]], (size_t)d->s[ord[i]].m * 4); k += d->s[ord[i]].m; }
    qsort(all, (size_t)tot, 4, rb_cmpf);
    const float thr = all[(size_t)((double)gap * (double)(tot - 1))];
    double sall = 0, ssel = 0; uint64_t nsel = 0;
    for (uint64_t j = 0; j < tot; j++) sall += all[j];
    for (uint32_t i = 0; i < n; i++) {
        float *g = gw[ord[i]];
        if (!g) continue;
        for (uint32_t j = 0; j < d->s[ord[i]].m; j++) {
            const bool on = g[j] >= thr && g[j] > 0.f;
            if (on) { ssel += g[j]; nsel++; }
            g[j] = on ? 1.f : rest;
        }
    }
    free(all);
    fprintf(stderr, "ds4: [restore-b] 缺口: 全料 %llu 位, 阈值 KL ≥ %.4f(%.0f%% 分位), 选中 %llu 位(%.1f%%), 这些位的 KL 占总量 %.1f%%; 其余位权重 %.2f\n",
            (unsigned long long)tot, (double)thr, 100.0 * gap, (unsigned long long)nsel, 100.0 * (double)nsel / (double)tot, sall > 0 ? 100.0 * ssel / sall : 0.0, (double)rest);
    if (lf) { fprintf(lf, "gap q %.3f thr %.5f sel %llu tot %llu klshare %.4f rest %.3f\n", (double)gap, (double)thr, (unsigned long long)nsel, (unsigned long long)tot, sall > 0 ? ssel / sall : 0.0, (double)rest); fflush(lf); }
}

/* 位置掩码(⑥): 读 "s e" 区间表, 每题答案位 j(目标在序列下标 sa0+j)落在某个区间里权重 1, 否则 0; 打一行账 */
static bool rb_apply_mask(const pt_data *d, const uint32_t *ord, uint32_t n, float **gw, const char *path, FILE *lf) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "ds4: ★[restore-b] 读不了掩码 %s★\n", path); return false; }
    uint32_t cap = 0, nr = 0, (*rg)[2] = NULL, a, z;
    while (fscanf(f, "%u %u", &a, &z) == 2) {
        if (z <= a) continue;
        if (nr == cap) { cap = cap ? cap * 2u : 64u; rg = realloc(rg, (size_t)cap * sizeof *rg); }
        rg[nr][0] = a; rg[nr][1] = z; nr++;
    }
    fclose(f);
    if (!nr) { fprintf(stderr, "ds4: ★[restore-b] 掩码 %s 一个区间都没有★\n", path); free(rg); return false; }
    uint64_t on = 0, tot = 0;
    for (uint32_t i = 0; i < n; i++) {
        const pt_sample *s = &d->s[ord[i]];
        float *g = gw[ord[i]];
        if (!g) continue;
        for (uint32_t j = 0; j < s->m; j++) {
            const uint32_t p = s->sa0 + j;
            bool in = false;
            for (uint32_t k = 0; k < nr && !in; k++) in = p >= rg[k][0] && p < rg[k][1];
            g[j] = in ? 1.f : 0.f; on += in; tot++;
        }
    }
    free(rg);
    fprintf(stderr, "ds4: [restore-b] 位置掩码 %s: %u 个区间, 答案位 %llu 里练 %llu 位(%.1f%%), 其余权重 0\n", path, nr, (unsigned long long)tot, (unsigned long long)on, tot ? 100.0 * on / tot : 0.0);
    if (lf) { fprintf(lf, "mask ranges %u on %llu tot %llu\n", nr, (unsigned long long)on, (unsigned long long)tot); fflush(lf); }
    return on > 0;
}

/* 打乱训练序: 按篇打乱、篇内段序不动(前缀形态按篇接力要同篇的段挨着过, 见 pt_run.cont_s) */
static void rb_shuffle(const pt_data *d, uint32_t *ord, uint32_t n, uint64_t *rs) {
    uint32_t *docs = xmalloc((size_t)(n + 1) * 4), nd = 0, *tmp = xmalloc((size_t)(n + 1) * 4), k = 0;
    for (uint32_t i = 0; i < n; i++) { bool seen = false; for (uint32_t j = 0; j < nd && !seen; j++) seen = docs[j] == d->s[ord[i]].doc; if (!seen) docs[nd++] = d->s[ord[i]].doc; }
    for (uint32_t i = nd; i > 1; i--) {
        *rs ^= *rs >> 12; *rs ^= *rs << 25; *rs ^= *rs >> 27;
        const uint32_t j = (uint32_t)((*rs * 2685821657736338717ull) % i), t = docs[i - 1]; docs[i - 1] = docs[j]; docs[j] = t;
    }
    for (uint32_t a = 0; a < nd; a++) for (uint32_t i = 0; i < n; i++) if (d->s[ord[i]].doc == docs[a]) tmp[k++] = ord[i];
    memcpy(ord, tmp, (size_t)n * 4);
    free(docs); free(tmp);
}

/* 逐层靶的读数(④): 练过的位 / 没练过的位各自 Σ‖xn·B·A − t‖² / Σ‖t‖²(还剩几成没补上), 每 3 块量 1 块(一块 cap 位 × 40 层 × 3 份读盘) */
static bool rb_aux_log(rb_aux *ax, const pt_cfg *c, pt_run *r, const uint8_t *mask, const char *tag, FILE *lf) {
    const uint32_t n = rb_aux_len(ax);
    double rs[2] = { 0, 0 }, ts[2] = { 0, 0 };
    for (uint32_t p = 1; p < n; ) {   /* 位置 0 是 BOS, 不预测任何训练位以外的东西也不在尺上 */
        uint32_t q = p; while (q < n && mask[q] == mask[p]) q++;
        if (!rb_aux_eval(ax, c, r, p, q, 3, &rs[mask[p]], &ts[mask[p]])) return false;
        p = q;
    }
    const double tr = ts[1] > 0 ? rs[1] / ts[1] : 0, ho = ts[0] > 0 ? rs[0] / ts[0] : 0;
    fprintf(stderr, "ds4: [restore-b] 逐层靶 %s: 练过的位剩 %.4f, 没练过的位剩 %.4f(1 = 一点没补)\n", tag, tr, ho);
    if (lf) { fprintf(lf, "aux %s train %.5f held %.5f\n", tag, tr, ho); fflush(lf); }
    return true;
}

int ds4_engine_restoreb(ds4_engine *e, const char *spec) {
    if (!e || !spec || !ds4_engine_is_v41(e) || !e->metal_ready) { fprintf(stderr, "ds4: --restore-b 要 V4.1 模型 + GPU 后端\n"); return 1; }
    if (g_ds4_v41_pt_dir) { fprintf(stderr, "ds4: --restore-b 不接 --posttrain\n"); return 1; }
    rb_cfg b; pt_cfg c; char ptpath[1200];
    if (!rb_cfg_load(spec, &b, &c, ptpath, sizeof ptpath)) return 1;
    c.nopack = 1; c.expand = 0;   /* 逐题过(损失要逐位权重, 合批路的损失没有这个口); 不改写问法(料是原文段) */
    pt_data d;
    if (!pt_data_load(e, &c, &d)) { pt_data_free(&d); return 1; }
    for (uint32_t i = 0; i < d.ns; i++) if (d.s[i].src == 0) d.s[i].eval = 0;   /* 料全训(门在外面) */
    if (!pt_teacher_fp(&c, &d)) { pt_data_free(&d); return 1; }
    pt_run r;
    if (!pt_run_alloc(e, &c, &r)) { pt_data_free(&d); return 1; }
    int rc = 1;
    bool ok = true;
    ds4_gpu_tensor *w = v41_alloc((uint64_t)r.st.cap_tok * 4, &ok);
    uint32_t *ord = xmalloc((size_t)d.ns * 4 + 4), ntr = 0;
    float **gw = xmalloc_zeroed(d.ns + 1u, sizeof *gw);
    char lp[1200]; snprintf(lp, sizeof lp, "%s/train.log", c.out);
    FILE *lf = fopen(lp, "w");
    rb_aux *ax = NULL; uint8_t *amask = NULL;
    rb_route rt; memset(&rt, 0, sizeof rt); rt.lr = b.route_lr;
    double aw = -1;   /* 逐层项的绝对权重: 第 1 步按两份梯度范数定(rb_auxw × ‖∇KL‖/‖∇逐层(w=1)‖), 之后不变 */
    if (!ok || (b.init[0] && !rb_load_init(&c, &r, b.init))) goto out;
    if (b.route && !rb_route_alloc(&rt, &c, &r, b.init)) goto out;
    for (uint32_t i = 0; i < d.ns; i++) if (d.s[i].src == 0 && d.s[i].top_id && pt_runnable(&r, &d.s[i])) { ord[ntr++] = i; gw[i] = xmalloc_zeroed(d.s[i].m, sizeof **gw); }
    if (!ntr) { fprintf(stderr, "ds4: [restore-b] 没有能训的题\n"); goto out; }
    if (b.dump[0]) {   /* ④ 只取料: 料里最长那条(前缀形态的整篇)整条前向一遍, 落完就走 —— 不预热、不训 */
        const pt_sample *lg = NULL;
        for (uint32_t i = 0; i < ntr; i++) if (!lg || d.s[ord[i]].sn > lg->sn) lg = &d.s[ord[i]];
        rc = rb_aux_dump(e, &r, lg->sids, lg->sn, b.dump) ? 0 : 1;
        goto out;
    }
    if (b.aux[0]) {   /* 逐层靶: 标出哪些位置被训(样本序列是靶序列的前缀、答案位前一位), 其余是没练过的位 */
        if (!(ax = rb_aux_open(b.aux, &c, &r))) goto out;
        amask = xmalloc_zeroed(rb_aux_len(ax) + 1u, 1);
        uint32_t nc = 0;
        for (uint32_t i = 0; i < ntr; i++) {
            const pt_sample *s = &d.s[ord[i]];
            if (!rb_aux_covers(ax, s)) continue;
            nc++;
            for (uint32_t p = s->sa0 + s->l0; p < s->sa0 + s->m; p++) amask[p - 1u] = 1;
        }
        if (!nc) { fprintf(stderr, "ds4: ★[restore-b] 料里没有一道是逐层靶序列(%s/ids.txt)的前缀★\n", b.aux); goto out; }
        fprintf(stderr, "ds4: [restore-b] 逐层靶接上 %u / %u 段, rb_auxw %.3g%s\n", nc, ntr, (double)b.auxw, b.auxw > 0.f ? "" : "(只量不训, 对照)");
    }
    {   /* 预热(③ 的预热原样调用): 最长一题完整走一趟前向 + 反传, 内存峰值开训前量出来; 这趟的梯度清掉 */
        uint32_t lmax = 0;
        if (!pt_pack_warm(e, &c, &r, &d, &lmax)) goto out;
        for (uint32_t il = c.layer_lo; il <= c.layer_hi; il++)
            if (!ds4_gpu_tensor_fill_f32(r.gA[il], 0.f, (uint64_t)r.K * DS4_N_EMBD) || !ds4_gpu_tensor_fill_f32(r.gB[il], 0.f, (uint64_t)r.K * DS4_N_EMBD)) goto out;
        const long mem = pt_mem_mb();
        fprintf(stderr, "ds4: [restore-b] 预热(最长 %u 行)后可用内存 %ld MB\n", lmax, mem);
        if (mem >= 0 && mem < RB_MEM_FLOOR_MB) { fprintf(stderr, "ds4: ★[restore-b] 可用内存 %ld MB 低于 %d MB, 停车★\n", mem, RB_MEM_FLOOR_MB); goto out; }
    }
    double kl0 = 0, kl1 = 0;
    if (b.freeze) fprintf(stderr, "ds4: [restore-b] 冻住前 %d 个秩(起点 %s), 只训后 %u 个\n", b.freeze, b.init, c.rank - (uint32_t)b.freeze);
    fprintf(stderr, "ds4: [restore-b] 训练 L%u-L%u 秩 %u, lr %.2g, batch %u, %u 轮, 料 %u 段; 第 0 步逐位 KL(定缺口)…\n", c.layer_lo, c.layer_hi, c.rank, (double)c.lr, c.batch, c.epochs, ntr);
    if (!rb_pass(e, &c, &r, w, &d, ord, ntr, gw, 1, &kl0)) goto out;
    if (lf) { fprintf(lf, "step 0 eval_kl 0.00000 train_kl %.5f hold_kl 0.00000\n", kl0); fflush(lf); }
    if (ax && !rb_aux_log(ax, &c, &r, amask, "e00", lf)) goto out;
    {   /* 第 0 步逐位 KL 落 <out>/kl0.tsv(一行"目标 token id  KL"): 拿去和编程尺的缺口按 token 类对照(api_round.py tokclass) ——
         * 料里的缺口和尺上的缺口不是同一类 token 时, 修好料的缺口也搬不到尺上(10-28: 尺的缺口 68% 在标识符上) */
        char kp[1200]; snprintf(kp, sizeof kp, "%s/kl0.tsv", c.out);
        FILE *kf = fopen(kp, "w");
        for (uint32_t i = 0; kf && i < ntr; i++) {
            const pt_sample *s = &d.s[ord[i]];
            for (uint32_t j = 0; j < s->m; j++) fprintf(kf, "%d %.6f\n", s->sids[s->sa0 + j], (double)gw[ord[i]][j]);
        }
        if (kf) fclose(kf); else fprintf(stderr, "ds4: [restore-b] 写不了 %s(只少一份诊断, 训练照常)\n", kp);
    }
    if (b.mask[0]) { if (!rb_apply_mask(&d, ord, ntr, gw, b.mask, lf)) goto out; }
    else if (b.gap > 0.f) rb_binarize(&d, ord, ntr, gw, b.gap, b.rest, lf);
    else for (uint32_t i = 0; i < ntr; i++) { free(gw[ord[i]]); gw[ord[i]] = NULL; }   /* 不加权: 每位都出梯度 */
    uint64_t rs = 0x2545F4914F6CDD1Dull ^ c.seed;
    uint32_t step = 0;
    const double t0 = now_sec();
    for (uint32_t ep = 1; ep <= c.epochs; ep++) {
        rb_shuffle(&d, ord, ntr, &rs);
        r.cont_s = NULL;
        for (uint32_t b0 = 0; b0 < ntr; b0 += c.batch) {
            const uint32_t b1 = b0 + c.batch < ntr ? b0 + c.batch : ntr;
            double ls = 0; uint32_t nt = 0;
            /* 逐层项: rb_auxw = 0 只量; 没定标前(第 1 步)只量梯度范数(w=1), 这一步只有出口 KL; 定标后按 aw 加进梯度 */
            const int cal = b.auxw > 0.f && aw >= 0;
            rb_auxstep as = { ax, 0, 0, 0, cal ? aw / c.batch : 1.0, cal ? 1 : (b.auxw > 0.f ? 2 : 0), 0, 0, 0, 0, 0 };
            for (uint32_t k = b0; k < b1; k++)
                if (!rb_step(e, &c, &r, w, &d.s[ord[k]], gw[ord[k]], gw[ord[k]] ? 2 : 0, 1, &ls, &nt, ax ? &as : NULL, &rt)) goto out;
            if (b.auxw > 0.f && aw < 0 && as.gn2 > 0) {   /* 定标: 此刻 gA/gB 里只有这一批的出口 KL 梯度 */
                double gk = 0, sq = 0;
                for (uint32_t il = c.layer_lo; il <= c.layer_hi; il++) {
                    if (!ds4_gpu_bwd_sumsq_tensor(r.gA[il], (uint64_t)r.K * DS4_N_EMBD, &sq)) goto out;
                    gk += sq;
                    if (!ds4_gpu_bwd_sumsq_tensor(r.gB[il], (uint64_t)r.K * DS4_N_EMBD, &sq)) goto out;
                    gk += sq;
                }
                aw = (double)b.auxw * sqrt(gk / as.gn2);
                fprintf(stderr, "ds4: [restore-b] 逐层项定标: 出口 KL 梯度 %.4g, 逐层项(w=1)梯度 %.4g ⇒ w = %.4g; 起点段 B 在下游敏感方向上剩 %.4f(不加权 %.4f)\n",
                        sqrt(gk), sqrt(as.gn2), aw, as.wtgt > 0 ? as.wres / as.wtgt : 0.0, as.tgt > 0 ? as.res / as.tgt : 0.0);
                if (lf) fprintf(lf, "aux scale gkl %.6g gaux1 %.6g w %.6g\n", sqrt(gk), sqrt(as.gn2), aw);
            }
            for (uint32_t il = c.layer_lo; b.freeze && il <= c.layer_hi; il++) {   /* ⑦ 冻秩: 前 n 个秩的梯度行([K][D] 的前 n 行)清零 */
                const uint64_t fb = (uint64_t)b.freeze * DS4_N_EMBD * 4;
                ds4_gpu_tensor *va = ds4_gpu_tensor_view(r.gA[il], 0, fb), *vb = ds4_gpu_tensor_view(r.gB[il], 0, fb);
                const bool zok = va && vb && ds4_gpu_tensor_fill_f32(va, 0.f, fb / 4) && ds4_gpu_tensor_fill_f32(vb, 0.f, fb / 4);
                if (va) ds4_gpu_tensor_free(va);
                if (vb) ds4_gpu_tensor_free(vb);
                if (!zok) goto out;
            }
            double gn = 0; int clipped = 0;
            if (!pt_opt_step(&c, &r, ++step, &gn, &clipped)) goto out;
            double gns = 0;
            if (rt.on && !rb_route_step(&rt, &c, &r, step, &gns)) goto out;
            if (rt.on && lf) fprintf(lf, "route step %u gnorm %.4g\n", step, gns);
            const long mem = pt_mem_mb();
            if (lf) { fprintf(lf, "step %u ep %u loss %.5f gnorm %.4g t %.0f mem %ld\n", step, ep, nt ? ls / nt : 0.0, gn, now_sec() - t0, mem); fflush(lf); }
            if (lf && as.tgt > 0) { fprintf(lf, "aux step %u resid %.5f wresid %.5f gaux %.4g\n", step, as.res / as.tgt, as.wtgt > 0 ? as.wres / as.wtgt : 0.0, sqrt(as.gn2)); fflush(lf); }
            if (ax && (step <= 3 || step % 10 == 0) && as.tgt > 0)
                fprintf(stderr, "ds4: [restore-b] 步 %u 逐层靶: 敏感方向上剩 %.4f, 不加权剩 %.4f\n", step, as.wtgt > 0 ? as.wres / as.wtgt : 0.0, as.res / as.tgt);
            if (mem >= 0 && mem < RB_MEM_FLOOR_MB) { fprintf(stderr, "ds4: ★[restore-b] 训练中可用内存 %ld MB 低于 %d MB, 停车★\n", mem, RB_MEM_FLOOR_MB); goto out; }
        }
        if (!rb_pass(e, &c, &r, w, &d, ord, ntr, gw, 0, &kl1)) goto out;
        fprintf(stderr, "ds4: [restore-b] ★轮 %u 末: 全料逐位 KL %.4f(第 0 步 %.4f), %u 步, %.0f s★\n", ep, kl1, kl0, step, now_sec() - t0);
        if (lf) { fprintf(lf, "epoch %u eval_kl 0.00000 train_kl %.5f hold_kl 0.00000 gain 0.00000 se 0.00000 nq %u rows 0\n", ep, kl1, ntr); fflush(lf); }
        char et[16]; snprintf(et, sizeof et, "e%02u", ep);
        if (ax && !rb_aux_log(ax, &c, &r, amask, et, lf)) goto out;
        char ck[1200]; snprintf(ck, sizeof ck, "%s/ckpt_e%02u", c.out, ep);
        if (!pt_save_amp(&c, &r, ck) || (rt.on && !rb_route_save(&c, &r, ck))) goto out;
    }
    fprintf(stderr, "ds4: [restore-b] 段 B 已落 %s/ckpt_eNN(amp_L%02u..L%02u.bin + base.fnv)\n", c.out, c.layer_lo, c.layer_hi);
    rc = 0;
out:
    if (lf) fclose(lf);
    rb_aux_close(ax); free(amask); rb_route_free(&rt);
    for (uint32_t i = 0; i < d.ns; i++) free(gw[i]);
    free(gw); free(ord);
    if (w) ds4_gpu_tensor_free(w);
    pt_run_free(&r); pt_data_free(&d);
    return rc;
}
#else
int ds4_engine_restoreb(ds4_engine *e, const char *spec) { (void)e; (void)spec; return 1; }
#endif /* !DS4_NO_GPU */
