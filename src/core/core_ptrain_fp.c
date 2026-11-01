/* core_ptrain_fp.c — ③ 的 FP 教师(2026-10-22, docs/think_close.md §3)。总述见 core_ptrain.h。
 *
 * 为什么教师要换: 要修的是量化把 </think> / EOS 这类"停"决策位压掉 4~22 倍(09-30 s1: FP 22%~49% 对量化 1%~6%; 整篇文档结尾 EOS 8 倍)。
 * 老路的教师 = 部署态自己(①+②), 它在这些位上自己就被压着, 学生再怎么逼近也到不了 FP 的 22%~49%。出厂原件 476 GB 装不进 spark 的 128 GB,
 * 只能离线逐层流式打分(v41_teacher.py, 与判决尺同一条参考前向), 所以分两步:
 *   ① dumpids=<文件>: 装料后把每道题的教师序列(tids)去重写出去(一行一串 id), 退出 —— 外面拿 v41_teacher.py --seqs 打 top-K;
 *   ② teacher=fp:<表>: 读回 top-K 表, 每道题按整条教师序列逐 id 对上才填表; 任一道对不上就停车, 不回退到部署态教师(回退 = 两种教师混在一趟里, 查不出来)。
 *   前缀形态的原文段(pt_sample.fpkey ≠ 0, 10-26): 键 = 整篇序列, 导一条、填整篇所有段 —— 表序列只要以该段的 tids 开头(因果前向, 前缀相同则各行相同), 行号同 ta0−1。
 *   表可以不是本地 FP 教师打的: 官方 API 开 logprobs 的 top-20(api_teacher.py ptf, 与 HF 原件 Same top 96.7% / KL 0.0065, fable5 10-26), 格式同一份。
 * 有了 FP 教师, one-hot 行(text 原文段 / 不带材料的问答 / 保持料)全部改成向 FP 蒸馏: 整篇文档的 EOS、思考收口的 </think> 都在 FP 的分布里,
 * 而保持料若仍钉在部署态自己, 等于把被压的 EOS 钉死(保持料的回答都以 EOS 收尾)。
 *
 * 表格式(v41_teacher_seqs.py 写, 可多段首尾相接 —— 大料分几趟打分后 cat 起来):
 *   每段: u32 magic 'PTF1' | u32 K | u32 N; 随后 N 条: u32 L | i32 ids[L] | i32 top_id[(L-1)·K] | f32 top_p[(L-1)·K] | f32 rest[L-1]
 *   第 i 行 = 位置 i 的 logits 预测 ids[i+1]。题的答案第 j 位(序列下标 ta0+j)用第 ta0-1+j 行 —— 与 pt_teacher_tail 同一口径。 */
#include "core_ptrain.h"
#ifndef DS4_NO_GPU

#define PT_FP_MAGIC 0x31465450u   /* 'PTF1' 小端 */

uint64_t pt_ids_fnv(const int32_t *ids, uint32_t n) {
    uint64_t h = 1469598103934665603ull ^ n;
    for (uint32_t i = 0; i < n; i++) { h ^= (uint64_t)(uint32_t)ids[i]; h *= 1099511628211ull; }
    return h;
}

/* 开放寻址表: 键 → 值(u32), 空槽 UINT32_MAX */
typedef struct { uint64_t *k; uint32_t *v, mask; } pt_fpmap;
static void pt_fpmap_init(pt_fpmap *m, uint32_t n) {
    uint32_t cap = 16; while (cap < 2u * n) cap <<= 1;
    m->k = xmalloc((size_t)cap * 8); m->v = xmalloc((size_t)cap * 4); memset(m->v, 0xff, (size_t)cap * 4); m->mask = cap - 1u;
}
static uint32_t *pt_fpmap_slot(pt_fpmap *m, uint64_t key) {
    uint32_t i = (uint32_t)(key ^ (key >> 29)) & m->mask;
    while (m->v[i] != UINT32_MAX && m->k[i] != key) i = (i + 1u) & m->mask;
    m->k[i] = key;
    return &m->v[i];
}

bool pt_dump_ids(const pt_cfg *c, const pt_data *d) {
    FILE *f = fopen(c->dumpids, "w");
    if (!f) { fprintf(stderr, "ds4: [ptrain] dumpids 写不了 %s\n", c->dumpids); return false; }
    pt_fpmap m; pt_fpmap_init(&m, d->ns);
    uint32_t n = 0, mx = 0; uint64_t tot = 0;
    for (uint32_t i = 0; i < d->ns; i++) {
        const pt_sample *s = &d->s[i];
        const int32_t *ids = s->fpkey ? d->docids[s->doc] : s->tids; const uint32_t n_ids = s->fpkey ? d->docn[s->doc] : s->tn;   /* 整篇 / 自己的窗 */
        uint32_t *v = pt_fpmap_slot(&m, s->fpkey ? s->fpkey : pt_ids_fnv(s->tids, s->tn));
        if (*v != UINT32_MAX) continue;   /* 同一串教师序列(重复题 / 同一篇的各段)只写一次 */
        *v = i;
        for (uint32_t t = 0; t < n_ids; t++) fprintf(f, t ? " %d" : "%d", ids[t]);
        fputc('\n', f);
        n++; tot += n_ids; if (n_ids > mx) mx = n_ids;
    }
    fclose(f); free(m.k); free(m.v);
    fprintf(stderr, "ds4: [ptrain] dumpids: %u 条教师序列(料 %u 题去重)共 %llu 位, 最长 %u → %s; 下一步 v41_teacher.py <HF> --seqs 这个文件 --topk %u\n",
            n, d->ns, (unsigned long long)tot, mx, c->dumpids, c->topk);
    return true;
}

/* 读整个表进内存(500k 位 × K=64 ≈ 260 MB), 记每条序列的 (ids 起点, 行表起点); 再逐题对上填表 */
bool pt_teacher_fp(const pt_cfg *c, pt_data *d) {
    const uint32_t K = c->topk;
    FILE *f = fopen(c->teacher_fp, "rb");
    if (!f) { fprintf(stderr, "ds4: [ptrain] 教师表 %s 打不开\n", c->teacher_fp); return false; }
    fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = xmalloc((size_t)sz + 1);
    if (sz <= 0 || fread(b, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(b); fprintf(stderr, "ds4: [ptrain] 教师表 %s 读不全\n", c->teacher_fp); return false; }
    fclose(f);
    /* 第一遍: 数条数、核格式 */
    uint32_t nseq = 0; size_t off = 0;
    while (off + 12 <= (size_t)sz) {
        uint32_t hd[3]; memcpy(hd, b + off, 12); off += 12;
        if (hd[0] != PT_FP_MAGIC) { fprintf(stderr, "ds4: [ptrain] 教师表 %s 偏移 %zu 不是 PTF1 段\n", c->teacher_fp, off - 12); free(b); return false; }
        if (hd[1] != K) { fprintf(stderr, "ds4: [ptrain] 教师表 K=%u 与 topk=%u 不同(表要按训练的 topk 重打)\n", hd[1], K); free(b); return false; }
        for (uint32_t i = 0; i < hd[2]; i++) {
            if (off + 4 > (size_t)sz) { fprintf(stderr, "ds4: [ptrain] 教师表 %s 尾部残缺\n", c->teacher_fp); free(b); return false; }
            uint32_t L; memcpy(&L, b + off, 4);
            const size_t need = 4 + (size_t)L * 4 + (size_t)(L - 1) * K * 8 + (size_t)(L - 1) * 4;
            if (L < 2 || off + need > (size_t)sz) { fprintf(stderr, "ds4: [ptrain] 教师表 %s 第 %u 条残缺(L=%u)\n", c->teacher_fp, nseq, L); free(b); return false; }
            off += need; nseq++;
        }
    }
    /* 第二遍: 建表 键(整条 ids) → 偏移 */
    pt_fpmap m; pt_fpmap_init(&m, nseq);
    size_t *at = xmalloc((size_t)(nseq + 1) * sizeof *at);
    off = 0; uint32_t q = 0;
    while (off + 12 <= (size_t)sz) {
        uint32_t hd[3]; memcpy(hd, b + off, 12); off += 12;
        for (uint32_t i = 0; i < hd[2]; i++) {
            uint32_t L; memcpy(&L, b + off, 4);
            int32_t *ids = (int32_t *)(b + off + 4);
            *pt_fpmap_slot(&m, pt_ids_fnv(ids, L)) = q; at[q++] = off;
            off += 4 + (size_t)L * 4 + (size_t)(L - 1) * K * 8 + (size_t)(L - 1) * 4;
        }
    }
    uint32_t filled = 0, miss = 0; uint64_t rows = 0;
    for (uint32_t i = 0; i < d->ns; i++) {
        pt_sample *s = &d->s[i];
        const uint32_t *v = pt_fpmap_slot(&m, s->fpkey ? s->fpkey : pt_ids_fnv(s->tids, s->tn));
        const uint8_t *rec = *v != UINT32_MAX ? b + at[*v] : NULL;
        uint32_t L = 0; if (rec) memcpy(&L, rec, 4);
        /* 自己的窗做键: 整条逐 id 同; 整篇做键: 表序列以 tids 开头(L ≥ tn)即可, 多出的尾巴是同篇后面段的行 */
        if (!rec || (s->fpkey ? L < s->tn : L != s->tn) || memcmp(rec + 4, s->tids, (size_t)s->tn * 4)) {
            if (miss < 5) fprintf(stderr, "ds4: [ptrain] ★教师表里没有第 %u 题的序列(%u 位, %s)★\n", i, s->tn, d->ch[s->chunk].name);
            miss++; continue;
        }
        const int32_t *tid = (const int32_t *)(rec + 4 + (size_t)L * 4);
        const float *tp = (const float *)(tid + (size_t)(L - 1) * K), *rest = tp + (size_t)(L - 1) * K;
        const uint32_t r0 = s->ta0 - 1u, mm = s->m;
        if (r0 + mm > L - 1u) { fprintf(stderr, "ds4: [ptrain] ★第 %u 题答案 %u 位超出表的 %u 行★\n", i, mm, L - 1u); miss++; continue; }
        free(s->top_id); free(s->top_p); free(s->top_rest);
        s->top_id = xmalloc((size_t)mm * K * 4); s->top_p = xmalloc((size_t)mm * K * 4); s->top_rest = xmalloc((size_t)mm * 4);
        memcpy(s->top_id, tid + (size_t)r0 * K, (size_t)mm * K * 4); memcpy(s->top_p, tp + (size_t)r0 * K, (size_t)mm * K * 4); memcpy(s->top_rest, rest + r0, (size_t)mm * 4);
        s->sft = 0;   /* 有 FP 教师就没有 one-hot: 全部样本向原件蒸馏 */
        filled++; rows += mm;
    }
    free(m.k); free(m.v); free(at); free(b);
    fprintf(stderr, "ds4: [ptrain] FP 教师表 %s: %u 条序列, 填 %u/%u 题 %llu 行%s\n", c->teacher_fp, nseq, filled, d->ns, (unsigned long long)rows, miss ? " ★有题没对上, 停车★" : "");
    return miss == 0 && filled > 0;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_ptrain_fp_nonempty_tu;
