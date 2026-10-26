/* v41_amp_run.c — V4.1 反修驱动: 在 ds4 引擎的真前向上逐层【取 → 扫 → 选 → 解】(2026-09-13), 零 Python。
 *
 * 【为什么要这个】解算循环必须跑在部署那条前向上: 引擎钩子 ds4_engine_v41_set_moe_hook 在每层 MoE 出口回调, 本程序拿到
 * 部署态 x_q / y_q / 路由, 用 HF 出厂权重算 y_fp(x_q)(v41_fp_moe.cu), 靶 = y_fp − y_q, 解算走 v41_amp_solve.cu。
 * (09-12 那版挂在 torch 前向上, 解算看到的 x 是 torch 模拟量化传播的, 与部署不是同一笔账: 同档对拍 KL 0.007。)
 *
 * 【多遍序贯】第 k 遍: 引擎挂着本目录里已解的 L0..L(k−1), 按 512 分块正常前向, 每块到第 k 层取料即停; 收齐全部行后解第 k 层
 * 落盘; 末遍全挂跑完整前向打拟合料 PPL。与"一次前向逐层解+挂"数学等价。不整批一次前向的原因: 8192 整批的引擎缓冲把
 * spark avail 压到 6 GB(看门狗线以下)且 L02 前向失败; 多遍用判决同一套 512 分块, 内存账已验证。
 *
 * 【择优规则】held-out 按 <ids>.layout 窗分层(每域窗号 k%4==3 作 val)。低秩路(amp_Lnn): K∈{64,128,256,512} × λ∈{1,3,10,30,100} 只认 val,
 * val ≥ 0.9×最优 的格里取最小 K、同 K 最大 λ; 层最优 val < 0.5% 不挂。★权重侧增益路(gr/grrb, 现役)的赢家由【端到端】定★(2026-10-21,
 * v41_e2e_select.inc.c): 候选挂上跑全前向, val 行上 Same top / KLD / RMS Δp 对教师锚(--e2e-ref)三项全优者赢, 否则 Same top 比基线退超 2√不一致行数 的否决、剩下 KLD 最低者赢,
 * KLD 差在趟间噪声内比 RMS Δp(基线=不挂增益也参赛)。三把尺就是判决器五指标里的三把(10-24 用户令"过程指标换成 Same top / KLD / RMS"; NLL 退出过程尺)。
 * 路由偏置 Δb 只由层内闸定(过了就留, 端到端不撤): 逐层裁会错杀"一串层合起来才赚"的 Δb, 账在 v41_gr_run.inc.c gr_e2e_select。
 *
 * 用法(三条路, 见 main 的 usage): 反修逐层 / 反修-蒸馏靶(--target kl:) / 后训练第三件(--sft-list, 见 v41_sft_run.inc.c)。
 * 编译: make -C gguf-tools v41_amp_run(Linux+CUDA; 要先在仓库根 make cuda-spark 出引擎对象) */
#define _GNU_SOURCE
#include "../../src/common/ds4_st41.h"
#include "../../src/common/ds4_quantfmt.h"
#include "../../src/common/ds4_float.h"
#include "../../ds4.h"
#include "../../ds4_gpu.h"               /* ds4_gpu_set_model_cache_limit_mb(设备权重缓存封顶, --weight-cache-mb) */
#include "v41_fp_moe.h"
#include "v41_kl_target.h"
#include "v41_route_solve.h"             /* 路由偏置侧车的 GPU 数值件(v41_rb_run.inc.c 驱动) */
#include "../../src/common/ds4_gguf.h"    /* 路由侧车要从 GGUF 取路由张量(与引擎同一份字节) */
#include "../../src/common/ds4_etgd.h"   /* --score-topk 产物的唯一读取实现(引擎写/解算器读/判决器读) */
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>

/* v41_amp_solve.cu 的三个入口(Python 路经 libv41amp.so 调的也是这三个) */
int v41_amp_scan_k_gpu(const void *dX, const void *dYfp, const void *dYq, int N, int D, int NFIT, const int *Ks, int nk,
                       const float *lams, int nl, float *out_train, float *out_val, float *out_tgt_rel, int whiten,
                       float *out_train_w, float *out_val_w);
int v41_amp_solve_layer_gpu(const void *dX, const void *dYfp, const void *dYq, int N, int D, int K, float lam,
                            void *dA, void *dB, float *out_ratio, int whiten);
int v41_amp_apply_gpu(const void *dX, void *dY, const void *dA, const void *dB, int N, int D, int K, float *out_dz);
/* v41_gr_solve.cu: 权重侧逐专家逐通道增益(down 行增益)重解 */
int v41_gr_solve_layer_gpu(const float *dye, const float *drw, const int *dsel, const float *dyfp, const float *dysh,
                           int n, int nfit, int nu, int D, int n_expert, float lam, int niter, float *ds, uint8_t *dpk,
                           float *out_train, float *out_val);

#define USAGE "用法: v41_amp_run <model.gguf> <hf-dir> <ids> <ntok上限> <出目录> [--layers N] [--whiten 0|1|2]\n" \
              "      [--mem-budget-mb M] [--weight-cache-mb M] [--no-engram]\n" \
              "  路由侧车: --rb <教师 --dump-moe 目录> [--base-amp <现役侧车>](一趟统计全部层的 Δb, 落 rb_Lnn.bin; 见 v41_rb_run.inc.c)\n" \
              "  合并序贯: --rb <dump> --capture-ye --e2e-ref <拟合料教师锚.bin>  (每层先解 Δb 挂上, 再取料解增益; 候选端到端择优; rb_Lnn + gr_Lnn 同目录)\n" \
              "  蒸馏靶:   --only-layer N --target kl:<教师锚> --eta-rel r\n" \
              "  后训练:   --sft-list <清单> --only-layer 39 --capture-ye --base-amp <②> [--share]\n" \
              "            [--reuse-tables] [--tau-list 1] [--rho-list 0.1,1,10] [--lam-list 0.3,1,3,10]\n" \
              "            [--kappa-list 1e9](专家频率岭) [--nu-list 0](词对重复度权重) [--gates R](x 键控, 1=旧形态)\n" \
              "  验证表:   --sft-list <清单> --verify-tabs g_ --base-pt <候选>(一次加载跑完所有样本)\n" \
              "  第七版:   --adv-list <清单: ids topk rms A_i nprompt 组号> --only-layer 39 --capture-ye --base-amp <②>\n" \
              "            [--eta-list 0.1,0.3,1] [--lam-list 0.3,1,3,10] [--own-pmax 0.95] [--rows-cap N] [--score-chunk 512](见 v41_adv_run.inc.c)\n" \
              "            [--lowrank K]: ③ 解成低秩加性项 amp_Lnn.bin(x·B·A, 与 ② 放大器同构), --lam-list 此时是相对岭(10-01)\n" \
              "  校准取料: --dump-calib  (裸 ① 上一趟, 40 层同趟落 <出目录>/calib_Lnn.bin, 给 v41_quantize --calib; 见 v41_calib_dump.inc.c)\n"

#define NK 4
#define NL 5
static const int SEL_KS[NK] = {64, 128, 256, 512};
static const float SEL_LAMS[NL] = {1.f, 3.f, 10.f, 30.f, 100.f};
#define GATE_PCT 0.5f        /* 层最优 val 低于此(百分点)不挂 */
#define KEEP 0.9f            /* 择优: val ≥ KEEP×最优 的格里挑最便宜的 */
#define MAXK 512
#define MAXU 16              /* n_used 上限(V4.1 = 6) */
#define GR_MAXL 8            /* 增益路 λ 网格上限(v41_gr_run.inc.c 的 GR_NL ≤ 它): 每个 λ 的解都留主机副本, 端到端择优要在它们里挑 */

typedef struct { double kld, rms, same; uint8_t *hit; } e2e_m;   /* 端到端尺的一组读数(val 行均值: KLD / RMS Δp(top-32 联合窗, 百分数) / Same top(百分数); hit[val 行] = 该行 Same top 中没中, 两组读数比 Same top 时算配对噪声用), v41_e2e_select.inc.c */
struct e2e_s;
/* 一个分支(挂/不挂本层路由偏置)上 λ 网格的全部解(v41_gr_run.inc.c): 层内 train/val 与每个解的主机副本(f32 + fp4 码字); 每层只解一支(Δb 层内闸过了是 1, 否则 0), 端到端在那一支里挑 */
typedef struct { float tr[GR_MAXL], va[GR_MAXL]; float *hgr[GR_MAXL]; uint8_t *pk[GR_MAXL]; int n; } gr_sols;

typedef struct {
    v41_st S;                              /* HF 出厂权重索引 */
    int *perm, nfit, n, D, MID, n_expert, n_used;   /* held-out 行置换(拟合行在前) / 形状 */
    FILE *mf; const char *out_dir;
    int layer, whiten;                     /* 本遍取第几层 */
    float clamp;
    float *rx, *ry, *rrw, *ralpha; int *rsel; int got;   /* 本遍原序取料 [n][D] / [n][n_used] / [n], got = 已收到的行数 */
    float *rye, *rysh; int want_ye;        /* --capture-ye: 逐专家 down 输出 [n][n_used][D](未乘 rw) + shared 输出 [n][D] */
    /* ★第七版取料槽★(2026-09-29/30, v41_adv_run.inc.c): rye_map[绝对行] = 槽号或 −1, 钩子只把有槽的行搬进 rye(f32, 槽 × n_used × D),
     * 首次到达时按真实 n_used 分配。为什么: 通用路按 ntok×MAXU(16) 全序列 f32 分配, 一条 CFO 样本 35k 行 = rye/hye/dye 各 11.5 GB,
     * 而 113.6 GB 的模型驻留后整机只剩 ~10 GB —— 不是"省一点", 是根本装不下。中间版本试过"只存生成段 bf16 取料窗"(一份 1.7~2.1 GB),
     * 与打分路状态 + 解码层懒分配叠在一起 30 s 内把 9 GB 吃到 2.2 GB 被看门狗杀(09-30 00:02) ⇒ 改成两遍: 第一遍只出榜单表定行集,
     * 第二遍只存选中的 ≤ rows_cap/N 行(4000 行 = 0.49 GB)。 */
    int *rye_map; int rye_slots, rye_nu;
    int rows_cap;                          /* --rows-cap: 第七版累积行数封顶(0 = 不封); 设备 ye 缓冲 = 它 × n_used × D × 4 */
    int score_chunk;                       /* --score-chunk: 取料趟的分块(0 = 引擎默认 DS4_V41_CHUNK)。分块不改输出(预填尺四档逐字节同), 只改状态内存:
                                            * 2048 行的打分状态(logits 主机+设备各 1.06 GB, hc/q/o…)在模型驻留后的 9 GB 余量里装不下第七版的料 */
    float *hye, *hysh, *hgr;               /* 置换后副本 + 增益缩放因子下载 [n_expert][D] */
    float *dye, *dysh, *drw, *dgr; int *dsel;   /* 权重侧解算的设备缓冲 */
    uint8_t *dgrpk, *hgrpk;                /* ★解算器直接产出的 fp4x32 码字★ [n_expert][D/32][17]: 落盘写的就是它 */
    float *dX, *dYq, *dYfp, *dYtmp, *dA, *dB;     /* 设备 */
    float *hA, *hB, *hx, *hy, *hrw, *halpha; int *hsel;   /* 主机: 放大器下载 / 置换后副本 */
    uint8_t *pkA, *pkB;                    /* fp4x32 码字(落盘的就是它) */
    float *hvf, *hvt;                      /* val 行的 y_fp / 工作副本: fp4 往返后重算 val 用 */
    v41_fp_ffn *ex, sh;
    /* 蒸馏靶(--target kl): 靶不再是"这一层像教师", 而是"让最终 logits 像教师该往哪挪"。见 v41_kl_target.h */
    v41_klt *klt; const char *kl_ref, *kl_stu; float eta_rel;
    /* 后训练第三件(2026-09-13 夜第二版): 靶 = 决策点上两个 token 的 logit 差(第一版拟合损失梯度已判死) */
    const char *sft_list, *base_amp, *base_pt;
    const char *adv_list, *eta_s; float own_pmax;   /* 第七版(09-29): 优势加权自 token 靶的清单 / η 网格 / 行筛阈值(v41_adv_run.inc.c) */
    const char *tau_s, *rho_s, *lam_s;     /* 决策差网格: τ=目标 logit 差 / ρ=约束行权重 / λ=岭 */
    const char *kap_s, *nu_s; float kappa;  /* 泛化先验(back.md §4.7): κ=专家频率岭 / ν=词对重复度权重 */
    int nsmp;                              /* 样本条数: 尺 L 按条留一, 折数 = 它 */
    int reuse_tab; const char *verify_pref;   /* 榜单已在盘上就取完料停车 / 挂候选只出表不解算(gatea 用) */
    int share; float *dgrf;                /* --share: 全体专家共用一组通道增益(未知数 D 个); dgrf = 广播+量化后的落地态 */
    int lowrank_k; float conf_min;         /* --lowrank K(10-01, 只配 --adv-list): ③ 解成低秩加性项 amp_Lnn.bin 而不是增益表; 钩子多存 x 行(rx), 见 v41_adv_run.inc.c adv_lowrank
                                            * --conf-min p: 事实行只推"榜首 p ≥ 阈值且榜首 ≠ 料 token"的位置(复读环正修, 见 adv_take_sample) */
    const char *gate_s; int gates; float *dgate;   /* --gates 列表: x 键控(R=1 = 旧形态; 一次取料跑多个 R);
                                                    * gates = 当前这一档 R; dgate[n][R] = 每行每门的 σ 标量 */
    int nsel;                              /* 解算真正用的行数(fit+val); 取料仍是全部 n 行 */
    const char *rb_dump; ds4_gguf gq;      /* 路由偏置侧车(v41_rb_run.inc.c): 教师 dump 目录 / GGUF(取路由张量) */
    /* 端到端择优(v41_e2e_select.inc.c): 教师锚句柄 / 当前基线(= 已挂层的读数) / 每个 λ 的解的主机副本 / 本层刚落了 rb / 已落盘的 rb 层数 */
    struct e2e_s *e2e; e2e_m e2e_base; gr_sols sol[2]; int nrb;   /* sol[0] = 不挂本层 Δb 的分支, sol[1] = 挂着 Δb 的分支 */
    ds4_engine *eng; const int *ids_all; int ntok_all, no_engram_all;   /* 全前向要的句柄与料(解算在层循环深处, 带不进参数) */
    int ok_layers, skip_layers, kcount[NK]; double val_sum, dz_sum, t_fp, t_scan, t_solve, t_fwd, t_e2e;
} ctx_t;

static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "★CUDA %s @%d: %s★\n", #x, __LINE__, cudaGetErrorString(e_)); return -1; } } while (0)

static int resolve_layer(ctx_t *c, int il);   /* v41_fp_bind.inc.c: HF 出厂权重索引(反修 FP 靶用) */
static float bf16r(float x);                   /* v41_amp_hook.inc.c: 引擎同式的 bf16 舍入(取料自检) */
static int layout_split(const char *ids_path, int ntok, ctx_t *c);   /* 同上: <ids>.layout 窗分层 held-out */
static int hook(void *ud, int il, int pos0, int n, int D, int n_used, float clamp, const float *x, const float *y, const int *sel,
                const float *rw, const float *alpha, const float *ye, const float *ysh);   /* 同上: 取料钩子 */
static int e2e_open(ctx_t *c, const char *ref, const int *ids, int ntok);   /* v41_e2e_select.inc.c: 教师锚 + val 行 */
static int e2e_run(ctx_t *c, const char *label, e2e_m *out);               /* 同上: 挂目录现状跑全前向出三指标 */
static int dir_has_plugins(const ctx_t *c);                                /* 同上: 盘上有没有插件文件(下一趟挂不挂) */
static int e2e_noise_check(ctx_t *c);                                      /* 同上: 起点再跑一趟量趟间噪声 */
static int solve_layer(ctx_t *c, int il);                            /* v41_amp_lowrank.inc.c(低秩放大器路) */
static int gr_work_alloc(ctx_t *c);                                  /* 同上: 取料/解算工作区按层重开(端到端前向之前被放掉) */
static int gr_solve_branch(ctx_t *c, int il, int b);                 /* 同上: 在当前取料上解 λ 网格 → sol[b] */
static int gr_e2e_select(ctx_t *c, int il, int has_rb);              /* 同上: 本层增益候选端到端择优, 盘上只留赢家(Δb 不在这里裁) */
static int rb_fit(ctx_t *c, ds4_engine *e, const int *ids, int ntok, const char *lp, int no_engram,
                  const char *dump, const char *base_amp, int nl);   /* v41_rb_run.inc.c(路由偏置侧车, 一趟模式) */
static int rb_seq_init(ctx_t *c, int ntok, int nl);                  /* 同上, 合并序贯: 统计表分配 / 本层解 Δb / 双职钩子 */
static int rb_seq_solve(ctx_t *c, int il, int *nw);
static void rb_seq_free(void);
static int rbgr_hook(void *ud, int il, int pos0, int n, int D, int n_used, float clamp, const float *x, const float *y, const int *sel,
                     const float *rw, const float *alpha, const float *ye, const float *ysh);
static int sft_run(ctx_t *c, ds4_engine *e, const char *list_path, const char *out, int il, int ntok_cap, int no_engram);  /* v41_sft_run.inc.c */
static int adv_run(ctx_t *c, ds4_engine *e, const char *list_path, const char *out, int il, int ntok_cap, int no_engram);  /* v41_adv_run.inc.c(第七版) */
static int *rows_spec_parse(const char *spec, int ntok, int *out_n); /* v41_sft_run.inc.c(行子集) */
static int calib_dump(ds4_engine *e, const int *ids, int ntok, const char *dir, int nl, int no_engram, int D, int n_expert,
                      const char *gguf, const char *ids_path);   /* v41_calib_dump.inc.c(量化校准取料, --dump-calib) */


/* 一趟取料前向: 挂目录现状, 钩子只认第 il 层, 收齐 ntok 行即停。 */
static int capture_pass(ctx_t *c, ds4_engine *e, const int *ids, int ntok, const char *lp, int no_engram, int il, ds4_v41_moe_hook_fn fn) {
    c->layer = il; c->got = 0;
    ds4_engine_v41_set_amp_dir(dir_has_plugins(c) ? c->out_dir : NULL);
    ds4_engine_v41_set_moe_hook(fn, c);
    ds4_engine_v41_set_moe_hook_layer(il);   /* 只要这一层, 省掉另外 39 层的白拷贝 */
    const double tf = now_s();
    const int rc = ds4_engine_v41_score_ids(e, ids, ntok, lp, no_engram, 0);
    ds4_engine_v41_set_moe_hook(NULL, NULL);
    c->t_fwd += now_s() - tf;
    if (rc != 0) { fprintf(stderr, "★第 %d 遍前向失败 rc=%d★\n", il, rc); return rc; }
    if (c->got != ntok) { fprintf(stderr, "★第 %d 遍只收到 %d/%d 行★\n", il, c->got, ntok); return 1; }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "%s", USAGE); return 2; }
    const char *gguf = argv[1], *hf = argv[2], *ids_path = argv[3], *out = argv[5];
    const int ntok = atoi(argv[4]);
    ctx_t c; memset(&c, 0, sizeof c);
    c.out_dir = out;
    int mem_mb = 0, cache_mb = 0, no_engram = 0, nlayers = 40, only_layer = -1, dump_calib = 0;
    const char *fit_spec = NULL, *val_spec = NULL, *e2e_ref = NULL;
    c.eta_rel = 0.10f;
    /* 网格默认值: τ=1 nat(p_对/p_错 ≥ e, 给 bf16/rms 的二阶误差留余量), ρ 三档看"别处别动"多贵,
     * λ 四档同反修量纲。12 个候选, 秒级一个 —— 贵的是取料那 21 分钟, 不是解算。 */
    c.tau_s = "1"; c.rho_s = "0.1,1,10"; c.lam_s = "0.3,1,3,10";
    /* κ=1e9 等于"不加频率岭"(退化成第二版), 留着当对照组; ν=0 同理 = 不看词对重复度。 */
    c.kap_s = "1e9"; c.nu_s = "0"; c.gate_s = "1";
    c.eta_s = "0.1,0.3,1"; c.own_pmax = 0.95f;   /* η = 赢家每个 token 想抬多少 nat(|A| 归一到 1); 0.95 以上的位置自 token≈期望, 方向≈0 */
    for (int i = 6; i < argc; i++) {
        if (!strcmp(argv[i], "--layers") && i + 1 < argc) nlayers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--whiten") && i + 1 < argc) c.whiten = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mem-budget-mb") && i + 1 < argc) mem_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--weight-cache-mb") && i + 1 < argc) cache_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-engram")) no_engram = 1;
        else if (!strcmp(argv[i], "--only-layer") && i + 1 < argc) only_layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--capture-ye")) c.want_ye = 1;
        else if (!strcmp(argv[i], "--dump-calib")) dump_calib = 1;
        else if (!strcmp(argv[i], "--rb") && i + 1 < argc) c.rb_dump = argv[++i];
        else if (!strcmp(argv[i], "--eta-rel") && i + 1 < argc) c.eta_rel = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--sft-list") && i + 1 < argc) c.sft_list = argv[++i];
        else if (!strcmp(argv[i], "--adv-list") && i + 1 < argc) c.adv_list = argv[++i];
        else if (!strcmp(argv[i], "--eta-list") && i + 1 < argc) c.eta_s = argv[++i];
        else if (!strcmp(argv[i], "--own-pmax") && i + 1 < argc) c.own_pmax = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--rows-cap") && i + 1 < argc) c.rows_cap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lowrank") && i + 1 < argc) c.lowrank_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--conf-min") && i + 1 < argc) c.conf_min = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--score-chunk") && i + 1 < argc) c.score_chunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reuse-tables")) c.reuse_tab = 1;
        else if (!strcmp(argv[i], "--share")) c.share = 1;
        else if (!strcmp(argv[i], "--verify-tabs") && i + 1 < argc) c.verify_pref = argv[++i];
        else if (!strcmp(argv[i], "--tau-list") && i + 1 < argc) c.tau_s = argv[++i];
        else if (!strcmp(argv[i], "--rho-list") && i + 1 < argc) c.rho_s = argv[++i];
        else if (!strcmp(argv[i], "--lam-list") && i + 1 < argc) c.lam_s = argv[++i];
        else if (!strcmp(argv[i], "--kappa-list") && i + 1 < argc) c.kap_s = argv[++i];
        else if (!strcmp(argv[i], "--nu-list") && i + 1 < argc) c.nu_s = argv[++i];
        else if (!strcmp(argv[i], "--gates") && i + 1 < argc) c.gate_s = argv[++i];
        else if (!strcmp(argv[i], "--base-amp") && i + 1 < argc) c.base_amp = argv[++i];
        else if (!strcmp(argv[i], "--base-pt") && i + 1 < argc) c.base_pt = argv[++i];
        else if (!strcmp(argv[i], "--fit-rows") && i + 1 < argc) fit_spec = argv[++i];
        else if (!strcmp(argv[i], "--val-rows") && i + 1 < argc) val_spec = argv[++i];
        else if (!strcmp(argv[i], "--e2e-ref") && i + 1 < argc) e2e_ref = argv[++i];
        else if (!strcmp(argv[i], "--target") && i + 1 < argc) {   /* --target kl:<教师锚> */
            const char *t = argv[++i];
            if (strncmp(t, "kl:", 3)) { fprintf(stderr, "★--target 只认 kl:<教师锚.bin>★\n"); return 2; }
            c.kl_ref = t + 3;
        } else { fprintf(stderr, "★不认识的参数 %s★\n", argv[i]); return 2; }
    }
    if (c.kl_ref && only_layer < 0) { fprintf(stderr, "★蒸馏靶只对末层成立(梯度链止于出口), 必须配 --only-layer <末层号>★\n"); return 2; }
    if (c.adv_list) {   /* 第七版靶与决策差靶同一条闭式链: 只对末层精确, 要逐专家输出, 与别的靶互斥 */
        if (c.sft_list || c.kl_ref || c.rb_dump) { fprintf(stderr, "★--adv-list 不与 --sft-list/--target/--rb 同给★\n"); return 2; }
        if (only_layer < 0 || !c.want_ye) { fprintf(stderr, "★--adv-list 必须配 --only-layer <末层号> --capture-ye★\n"); return 2; }
        if (!(c.own_pmax > 0.f && c.own_pmax <= 1.f)) { fprintf(stderr, "★--own-pmax %g 要在 (0,1]★\n", c.own_pmax); return 2; }
        c.sft_list = c.adv_list;   /* 下面按清单模式走(ids 占位、行分层、出口头), 分流点在 sft_run 之前 */
    }
    if (c.sft_list) {
        /* 后训练靶与蒸馏靶同一条闭式链, 同样只对末层精确 —— 往前的层要穿过后面各层的雅可比,
         * 那是 back.md §3.3 的级 2/级 3, 不在这条路上偷偷做。 */
        /* 验证趟只跑前向出表, 不取料不解算 ⇒ 不要求 --only-layer/--capture-ye, 也不用出口头 */
        if (!c.verify_pref) {
            if (only_layer < 0) { fprintf(stderr, "★后训练靶只对末层精确, 必须配 --only-layer <末层号>★\n"); return 2; }
            if (!c.want_ye) { fprintf(stderr, "★后训练解的是逐专家增益, 必须配 --capture-ye★\n"); return 2; }
        }
        if (c.kl_ref) { fprintf(stderr, "★--target kl 与 --sft-list 是两个靶, 只能给一个★\n"); return 2; }
    }
    if (c.eta_rel <= 0.f || c.eta_rel > 2.f) { fprintf(stderr, "★--eta-rel %g 不合理★\n", c.eta_rel); return 2; }
    if (c.rb_dump && (c.sft_list || c.kl_ref)) { fprintf(stderr, "★--rb 不与 --sft-list/--target 同给(配 --capture-ye = 路由+增益合并序贯)★\n"); return 2; }
    const int merged = c.rb_dump && c.want_ye;   /* 合并序贯: 每层 Δb → 挂上 → 增益 */
    /* 权重侧增益的序贯路(gr / grrb)赢家由端到端尺定, 锚不在就不解 —— 不退回层内择优(那把尺 10-20 已判不能当裁判) */
    const int gr_seq = c.want_ye && !c.sft_list && !dump_calib;
    if (gr_seq && !e2e_ref) { fprintf(stderr, "★权重侧增益反修的择优尺是端到端尺, 必须给 --e2e-ref <拟合料教师锚.bin>(v41_judge.sh 落的 teacher_<料>_n<N>.bin)★\n"); return 2; }
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (ntok < 2 || nlayers < 1 || nlayers > 64) { fprintf(stderr, "★ntok %d / layers %d★\n", ntok, nlayers); return 2; }
    /* 清单模式(后训练)自己按条读 ids, 这里的 <ids> 只是个占位; 行分层也由清单的 fit/val 列给,
     * 不走 .layout 窗分层 —— 两套分层同时生效只会打架, 而打架的结果在读数里看不出来。 */
    int *ids = NULL, n = 0, t;
    if (!c.sft_list) {
        ids = malloc(sizeof(int) * (size_t)ntok);
        FILE *fi = fopen(ids_path, "r");
        if (!fi) { fprintf(stderr, "★打不开 %s★\n", ids_path); return 1; }
        while (n < ntok && fscanf(fi, "%d", &t) == 1) ids[n++] = t;
        fclose(fi);
        if (n != ntok) { fprintf(stderr, "★%s 只有 %d 个 id < ntok %d★\n", ids_path, n, ntok); return 1; }
    }
    /* ★行子集(后训练用)★: 报告段之外的行(当日材料)不该进解算 —— 它们不是"该写出来的东西",
     * 梯度在那里没有意义。--fit-rows/--val-rows 直接给绝对行号区间(与判决器 --rows 同语法),
     * 不给就退回 layout 窗分层(反修那条路的老规矩)。两者只能二选一, 免得两套分层打架。 */
    if (fit_spec || val_spec) {
        if (!fit_spec || !val_spec) { fprintf(stderr, "★--fit-rows 与 --val-rows 必须成对给★\n"); return 2; }
        int nf = 0, nv = 0;
        int *fit = rows_spec_parse(fit_spec, ntok, &nf), *val = rows_spec_parse(val_spec, ntok, &nv);
        if (!fit || !val) { fprintf(stderr, "★--fit-rows/--val-rows 要 a:b 或 a:b,c:d,...★\n"); return 2; }
        c.perm = malloc(sizeof(int) * (size_t)(nf + nv));
        memcpy(c.perm, fit, sizeof(int) * (size_t)nf);
        memcpy(c.perm + nf, val, sizeof(int) * (size_t)nv);
        c.nfit = nf; c.n = ntok; c.nsel = nf + nv;
        free(fit); free(val);
        printf("[行子集] 取料 %d 行, 解算用 %d 行(拟合 %d / val %d)\n", ntok, c.nsel, nf, nv);
    } else if (c.sft_list) {
        c.n = ntok; c.nfit = ntok;   /* 真正的行数与分层在 sft_run 里按清单定 */
    } else if (layout_split(ids_path, ntok, &c)) return 1;
    if (mkdir(out, 0775) && errno != EEXIST) { fprintf(stderr, "★建不了 %s★\n", out); return 1; }
    /* HF 出厂索引 + 形状(专家数/MID/D 从张量名与形状读, 不写死) */
    if (v41_st_open(&c.S, hf)) return 1;
    for (int i = 0; i < c.S.ne; i++) { int L, e; if (sscanf(c.S.e[i].name, "layers.%d.ffn.experts.%d.w1.weight", &L, &e) == 2 && L == 0 && e + 1 > c.n_expert) c.n_expert = e + 1; }
    const v41_st_ent *w1 = v41_st_find(&c.S, "layers.0.ffn.experts.0.w1.weight");
    if (!c.n_expert || !w1 || w1->nd != 2) { fprintf(stderr, "★%s 不像 V4.1 HF 目录(没有 layers.0.ffn.experts.*)★\n", hf); return 1; }
    c.MID = (int)w1->shape[0]; c.D = (int)w1->shape[1] * 2;
    if (only_layer >= 0) printf("[反修·引擎路] HF %d 专家 × [%d×%d], 拟合料 %s n=%d, ★只解 L%02d★, 白化 %d → %s\n", c.n_expert, c.MID, c.D, ids_path, ntok, only_layer, c.whiten, out);
    else printf("[反修·引擎路] HF %d 专家 × [%d×%d], 拟合料 %s n=%d, 前 %d 层, 白化 %d → %s\n", c.n_expert, c.MID, c.D, ids_path, ntok, nlayers, c.whiten, out);
    c.ex = calloc((size_t)c.n_expert, sizeof(v41_fp_ffn));
    const size_t nD = (size_t)ntok * c.D, nU = (size_t)ntok * MAXU;
    /* ★第七版(--adv-list)只分配它真用的★: 钩子要 rsel/rrw/ralpha(每行几十字节), ye 走取料窗(钩子里按真实 n_used 分配,
     * 只存生成段, bf16), 设备缓冲在 adv_run 里按封顶行数分配。通用路那套 [ntok][MAXU][D] f32 × 3 + [ntok][D] × 8
     * 在 35k 行上是 40 GB, 模型驻留后整机只剩 ~10 GB(core_model_map.c 内存地板), 分了就把权重挤出缓存或直接 OOM。 */
    const int adv = c.adv_list != NULL;
    c.rrw = malloc(nU * 4); c.rsel = malloc(nU * 4); c.ralpha = malloc((size_t)ntok * 4);
    if (!c.rrw || !c.rsel || !c.ralpha) { fprintf(stderr, "★取料元数据缓冲分配失败★\n"); return 1; }
    if (adv && c.lowrank_k && !(c.rx = malloc(nD * 4))) { fprintf(stderr, "★低秩形态的 x 行缓冲(%.2f GB)分配失败★\n", (double)nD * 4 / 1e9); return 1; }
    if (!adv) {
        c.rx = malloc(nD * 4); c.ry = malloc(nD * 4);
        c.hx = malloc(nD * 4); c.hy = malloc(nD * 4); c.hrw = malloc(nU * 4); c.hsel = malloc(nU * 4); c.halpha = malloc((size_t)ntok * 4);
        if (!c.rx || !c.ry || !c.hx || !c.hy || !c.hrw || !c.hsel || !c.halpha) { fprintf(stderr, "★取料缓冲(%.1f GB)分配失败★\n", (double)nD * 16 / 1e9); return 1; }
    }
    c.hA = malloc((size_t)MAXK * c.D * 4); c.hB = malloc((size_t)MAXK * c.D * 4);
    c.pkA = malloc((size_t)MAXK * c.D / 32u * 17u); c.pkB = malloc((size_t)MAXK * c.D / 32u * 17u);
    c.hvf = malloc((size_t)(ntok - c.nfit + 1) * c.D * 4); c.hvt = malloc((size_t)(ntok - c.nfit + 1) * c.D * 4);
    if (!c.pkA || !c.pkB || !c.hvf || !c.hvt) { fprintf(stderr, "★fp4/val 缓冲分配失败★\n"); return 1; }
    if (c.want_ye && adv) {
        c.hgr = malloc((size_t)c.n_expert * c.D * 4);
        if (!c.hgr) { fprintf(stderr, "★增益缓冲分配失败★\n"); return 1; }
    } else if (c.want_ye) {   /* [n][n_used][D] f32: 8192×6×5120×4 = 1.0 GB */
        const size_t nyD = nU * (size_t)c.D, nsD = (size_t)c.n_expert * c.D;
        c.rye = malloc(nyD * 4); c.rysh = malloc(nD * 4);
        c.hye = malloc(nyD * 4); c.hysh = malloc(nD * 4); c.hgr = malloc(nsD * 4);
        c.hgrpk = (uint8_t *)malloc(nsD / 32u * 17u);
        if (!c.rye || !c.rysh || !c.hye || !c.hysh || !c.hgr || !c.hgrpk) {
            fprintf(stderr, "★逐专家输出缓冲(2×%.1f GB)分配失败★\n", (double)nyD * 4 / 1e9); return 1; }
        for (int b = 0; b < 2; b++) for (int l = 0; l < GR_MAXL; l++) {   /* 两个分支 × 每个 λ 的解各留一份(f32 7.9 MB + 码字 1.0 MB), 端到端择优在它们里挑 */
            c.sol[b].hgr[l] = malloc(nsD * 4); c.sol[b].pk[l] = (uint8_t *)malloc(nsD / 32u * 17u);
            if (!c.sol[b].hgr[l] || !c.sol[b].pk[l]) { fprintf(stderr, "★λ 候选缓冲分配失败★\n"); return 1; }
        }
        if (cudaMalloc((void **)&c.dye, nyD * 4) || cudaMalloc((void **)&c.dysh, nD * 4) ||
            cudaMalloc((void **)&c.drw, nU * 4) || cudaMalloc((void **)&c.dsel, nU * 4) ||
            cudaMalloc((void **)&c.dgr, nsD * 4) ||
            cudaMalloc((void **)&c.dgrpk, nsD / 32u * 17u)) { fprintf(stderr, "★权重侧解算设备缓冲分配失败★\n"); return 1; }
    }
    if (!adv && (cudaMalloc((void **)&c.dX, nD * 4) || cudaMalloc((void **)&c.dYq, nD * 4) || cudaMalloc((void **)&c.dYfp, nD * 4) ||
        cudaMalloc((void **)&c.dYtmp, nD * 4) || cudaMalloc((void **)&c.dA, (size_t)MAXK * c.D * 4) || cudaMalloc((void **)&c.dB, (size_t)MAXK * c.D * 4))) {
        fprintf(stderr, "★设备缓冲分配失败★\n"); return 1;
    }
    char mp[4300]; snprintf(mp, sizeof mp, "%s/manifest.txt", out);
    c.mf = fopen(mp, "w");
    if (!c.mf) { fprintf(stderr, "★写不了 %s★\n", mp); return 1; }
    fprintf(c.mf, "# 层 λ K 靶/‖y_q‖ train%% val%% ‖Δ‖/‖y_q‖ [val_w%%]  (val<gate 的层记 skip; 数值来自 C 库; 百分比=1−‖残差‖/‖靶‖ 幅值比)\n"
                  "# 候选 K=[64, 128, 256, 512] λ=[1, 3, 10, 30, 100] gate=%.1f%% keep=%.1f whiten=%d%s\n"
                  "# 解算前向 = ds4 引擎(部署同路, v41_amp_run 多遍序贯) gguf=%s 拟合=%s n=%d\n",
            GATE_PCT, KEEP, c.whiten, c.whiten ? "(择优看白化 val)" : "", gguf, ids_path, ntok);
    if (gr_seq) fprintf(c.mf, "# 增益路择优 = 端到端(e2e 行): 候选挂上跑全前向, 拟合料 val 行对教师锚 %s 算 Same top / KLD / RMS Δp; 基线(不挂增益; 本层 Δb 层内闸过了就是只挂 Δb)也参赛, 三项全优者赢, 没有全优者先否决 Same top 比基线退超 2√不一致行数 的候选, 剩下 KLD 最低者赢(差在趟间噪声内比 RMS Δp 再比 Same top);基线赢本层不挂增益; Δb 只由层内闸定, 端到端不撤\n", e2e_ref);
    fflush(c.mf);
    /* 引擎: 与 CLI 同款打开(CUDA 后端) */
    if (mem_mb > 0) ds4_set_mem_budget_mb(mem_mb);
    /* 拟合工作区 ~17 GiB(09-20 实测反推)叠在整模型设备缓存之上, 105.87 GiB 的模型在 121 GiB 机器上装不下
     * ⇒ 封顶缓存, 装不下的层走逐层流式(见 ds4_gpu_core.h)。必须在引擎打开(注册映射)之前设。 */
    if (cache_mb > 0) ds4_gpu_set_model_cache_limit_mb((uint64_t)cache_mb);
    ds4_engine_options opt; memset(&opt, 0, sizeof opt);
    opt.model_path = gguf; opt.backend = DS4_BACKEND_CUDA;
    ds4_engine *e = NULL;
    const double t0 = now_s();
    if (ds4_engine_open(&e, &opt) != 0 || !e) { fprintf(stderr, "★引擎打不开 %s★\n", gguf); return 1; }
    if (!ds4_engine_is_v41(e)) { fprintf(stderr, "★%s 不是 V4.1 模型★\n", gguf); return 1; }
    printf("[引擎] 打开 %.0fs; 多遍序贯反修开始(每遍取一层)\n", now_s() - t0);
    char lp[4300]; snprintf(lp, sizeof lp, "%s/fit_logits.bin", out);
    int rc = 0;
    if (dump_calib) {   /* 量化校准取料: 40 层同趟落 x/路由, 不解算(v41_calib_dump.inc.c) */
        const int drc = calib_dump(e, ids, ntok, out, nlayers, no_engram, c.D, c.n_expert, gguf, ids_path);
        fclose(c.mf); unlink(mp);   /* 取料目录不该有 manifest(那是反修产物的账) */
        ds4_engine_close(e); v41_st_close(&c.S);
        return drc == 0 ? 0 : 1;
    }
    if (c.rb_dump) {   /* 路由侧车(两种模式都要): 路由张量从 GGUF 取, 与引擎同一份字节 */
        char err[256];
        if (ds4_gguf_open(&c.gq, gguf, err, sizeof err)) { fprintf(stderr, "★路由侧车要读 GGUF 的路由张量, 打不开 %s: %s★\n", gguf, err); return 1; }
    }
    if (c.rb_dump && !merged) {   /* 路由偏置侧车一趟模式: 一趟前向统计全部层的 Δb, 落 rb_Lnn.bin(与 gr 同目录部署) */
        const int prc = rb_fit(&c, e, ids, ntok, lp, no_engram, c.rb_dump, c.base_amp, nlayers);
        if (prc != 0) fprintf(c.mf, "# 失败 路由侧车\n");
        fclose(c.mf);
        ds4_gguf_close(&c.gq); ds4_engine_close(e); v41_st_close(&c.S);
        return prc == 0 ? 0 : 1;
    }
    if ((c.kl_ref || c.sft_list) && !c.verify_pref) {   /* 出口头 + output_norm 从同一个 GGUF 取(学生自己的出口) */
        if (v41_klt_open(&c.klt, gguf, c.D)) { fprintf(stderr, "★出口靶初始化失败★\n"); return 1; }
        c.kl_stu = lp;
        if (c.sft_list) printf("[后训练] 样本清单 %s; 网格 τ=%s ρ=%s λ=%s; 底座 ②=%s ③=%s\n",
                               c.sft_list, c.tau_s, c.rho_s, c.lam_s, c.base_amp ? c.base_amp : "(无)", c.base_pt ? c.base_pt : "(无)");
        else printf("[蒸馏靶] 教师锚 %s; 学生 logits = 本趟前向落盘 %s; --eta-rel %.3f\n", c.kl_ref, lp, c.eta_rel);
    }
    if (c.sft_list) {   /* 后训练: 多条样本各取一趟料, 累积后一次解(v41_sft_run.inc.c / 第七版 v41_adv_run.inc.c) */
        const int src = c.adv_list ? adv_run(&c, e, c.adv_list, out, only_layer, ntok, no_engram)
                                   : sft_run(&c, e, c.sft_list, out, only_layer, ntok, no_engram);
        fprintf(c.mf, "# 后训练 %s\n", src == 0 ? "完成" : "失败");
        fclose(c.mf);
        if (c.klt) v41_klt_close(c.klt);
        ds4_engine_close(e);
        v41_st_close(&c.S);
        return src == 0 ? 0 : 1;
    }
    const int il_lo = only_layer >= 0 ? only_layer : 0, il_hi = only_layer >= 0 ? only_layer + 1 : nlayers;
    c.eng = e; c.ids_all = ids; c.ntok_all = ntok; c.no_engram_all = no_engram;
    if (gr_seq) {   /* 端到端尺: 开锚, 先量一趟"起点态"(目录里已有的插件都挂上; 全新目录 = 裸底座)当第一层的基线 */
        if (e2e_open(&c, e2e_ref, ids, ntok)) return 1;
        if (e2e_run(&c, "起点", &c.e2e_base)) return 1;
        if (e2e_noise_check(&c)) return 1;   /* 起点再跑一趟: 趟间噪声落 manifest, 超过 Same top 的容差就停(尺不可信) */
    }
    if (merged) {
        if (rb_seq_init(&c, ntok, nlayers)) { fprintf(stderr, "★路由侧车统计表分配失败★\n"); return 1; }
        printf("[合并序贯] 每层: 路由统计+增益取料 同一遍 → 解 Δb(层内严格涨才落盘, 落了就留) → 有 Δb 挂着重取再解增益, 没有就在第一趟料上解 → 层内前 2 个增益候选端到端裁(基线 = 只挂 Δb / 上一层赢家); FP 路由 ← %s\n", c.rb_dump);
    }
    for (int il = il_lo; il < il_hi && rc == 0; il++) {   /* 第 il 遍: 挂 L0..L(il−1) 已解的放大器, 到 L(il) 取料即停 */
        /* ★取料必须在"要部署的那个态"上★: 后训练解的是 ①+②(+③上一版)之上的增量, 所以取料遍挂的是
         * --base-amp/--base-pt, 不是本次的输出目录(输出目录里此刻还什么都没有)。反修序贯那条路仍
         * 挂自己的输出目录(第 k 遍挂 L0..L(k−1))。目录里一个插件都没有时引擎会拒开状态, 所以按盘上有没有文件挂。
         * ★合并序贯★(2026-10-21): 第一趟取料同时统计路由 → 解 Δb(层内严格涨才落盘, 落了就留) → 有 Δb 就挂着它重取一趟再解增益 →
         * 增益候选过端到端闸(基线 = 只挂 Δb, 没 Δb 就是上一层赢家)。重取不能省: 挂着 Δb 的路由与第一趟取料时的不是一回事。 */
        if (gr_seq && !c.rx && gr_work_alloc(&c)) { rc = 1; break; }   /* 上一层的端到端前向之前把工作区放了, 取料前重开 */
        if ((rc = capture_pass(&c, e, ids, ntok, lp, no_engram, il, merged ? rbgr_hook : hook)) != 0) break;
        int has_rb = 0;
        if (merged) {
            const int w = rb_seq_solve(&c, il, &c.nrb);
            if (w < 0) { rc = 1; break; }
            has_rb = (w == 1);
        }
        if (!gr_seq) { if (solve_layer(&c, il) < 0) rc = 1; continue; }
        /* ★Δb 层内闸过了就留, 增益只在挂着它重取的料上解★(2026-10-21 第二趟后的诊断): 逐层端到端裁 Δb 去留会错杀 —— 单层看每层只亏
         * 0.02~0.78%, 一串层一起挂却是赚(判决份 KLD −2.4%), 账在 v41_gr_run.inc.c gr_e2e_select 的头注释。闸没过的层只解无 Δb 一支。 */
        if (!has_rb && gr_solve_branch(&c, il, 0) < 0) { rc = 1; break; }
        if (has_rb) {
            if ((rc = capture_pass(&c, e, ids, ntok, lp, no_engram, il, hook)) != 0) break;
            if (gr_solve_branch(&c, il, 1) < 0) { rc = 1; break; }
        }
        if (gr_e2e_select(&c, il, has_rb) < 0) rc = 1;
    }
    if (rc == 0) {   /* 末遍: 全部放大器挂上跑完整前向, 拟合料 PPL(与 Python 解算趟末尾的 PPL 同口径) */
        ds4_engine_v41_set_amp_dir(dir_has_plugins(&c) ? out : NULL);
        const double tf = now_s();
        rc = ds4_engine_v41_score_ids(e, ids, ntok, lp, no_engram, 0);
        c.t_fwd += now_s() - tf;
        unlink(lp);   /* 拟合料 logits(4 GB)只为打 PPL, 文件不留 */
        if (rc != 0) fprintf(stderr, "★末遍前向失败 rc=%d★\n", rc);
    }
    if (rc != 0) { fprintf(c.mf, "# 失败 rc=%d\n", rc); fclose(c.mf); fprintf(stderr, "★反修失败, 放大器目录是半成品★\n"); return 1; }
    /* 总账(= v41_amp_hooks.report_select) */
    double mb = 0.0; char kd[128] = "";
    /* 体积 = 盘上真实字节: A、B 两块各 K×D 元素, fp4x32 是 17 B/32 元素 */
    for (int k = 0; k < NK; k++) { mb += 2.0 * ((double)c.D * SEL_KS[k] / 32.0 * 17.0) / 1e6 * c.kcount[k]; if (c.kcount[k]) { char b[32]; snprintf(b, sizeof b, "%s%d×%d", kd[0] ? " " : "", SEL_KS[k], c.kcount[k]); strncat(kd, b, sizeof kd - strlen(kd) - 1); } }
    if (c.ok_layers)
        printf("[择优] 挂 %d 层 / 闸掉 %d 层; fp4 后 val 平均 %+.2f%%; ‖Δ‖/‖y_q‖ 平均 %.4f; K 分布 %s; fp4x32 落地 %.0f MB = %.2f%% 解码带宽; 用时 前向 %.0fs fp %.0fs 扫 %.0fs 解 %.0fs 端到端 %.0fs 总 %.0fs\n",
               c.ok_layers, c.skip_layers, c.val_sum / c.ok_layers, c.dz_sum / c.ok_layers, kd, mb, mb / 10050 * 100, c.t_fwd, c.t_fp, c.t_scan, c.t_solve, c.t_e2e, now_s() - t0);
    else printf("[择优] 零层挂上(闸掉 %d) —— 检查靶/输入\n", c.skip_layers);
    if (gr_seq) printf("[端到端择优] 终态 val 行: KLD %.5f RMS Δp %.4f%% Same top %.2f%%(每层候选与基线的读数在 manifest 的 e2e 行)\n", c.e2e_base.kld, c.e2e_base.rms, c.e2e_base.same);
    if (merged) { printf("[合并序贯] 路由偏置落盘 %d 层 rb_Lnn.bin(与 gr 同目录)\n", c.nrb); rb_seq_free(); ds4_gguf_close(&c.gq); }
    if (merged) fprintf(c.mf, "# 完成 挂%d 闸%d 路由%d\n", c.ok_layers, c.skip_layers, c.nrb);
    else fprintf(c.mf, "# 完成 挂%d 闸%d\n", c.ok_layers, c.skip_layers);
    fclose(c.mf);
    if (c.klt) v41_klt_close(c.klt);
    ds4_engine_close(e);
    v41_st_close(&c.S);
    return 0;
}

#include "v41_fp_bind.inc.c"  /* HF 张量 → v41_fp_mat 的绑定(同上) */
#include "v41_amp_hook.inc.c" /* 取料钩子 + layout 窗分层 + bf16 舍入(同上) */
#include "v41_e2e_select.inc.c" /* 端到端择优: 教师锚、全前向三指标、入围与名次(同上) */
#include "v41_amp_lowrank.inc.c" /* 低秩放大器那条路的逐层解算(同上) */
#include "v41_gr_run.inc.c"   /* 权重侧逐专家增益的驱动编排(单 TU; 拆文件只为守 500 行) */
#include "v41_rb_run.inc.c"   /* 路由偏置侧车的拟合驱动(同上) */
#include "v41_sft_run.inc.c"  /* 后训练第三件: 取料、清单、行表(同上) */
#include "v41_margin_solve.inc.c" /* 后训练第三件: 决策差解算 + 主动集(用上面的 mg_acc) */
#include "v41_margin_gen.inc.c"   /* 后训练第三件: 留一泛化尺 + 诊断(用上面的 mg_candidate) */
#include "v41_adv_run.inc.c"      /* 后训练第七版(09-29): 优势加权自 token 靶(用上面的 mg_acc / mg_read_gr / sft_write_base) */
#include "v41_calib_dump.inc.c"   /* 量化校准取料(--dump-calib; 单 TU 同上) */
