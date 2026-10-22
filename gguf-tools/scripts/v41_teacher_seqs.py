"""v41_teacher_seqs.py — FP 教师的多序列 top-K 档(2026-10-22, docs/think_close.md §3; 被 v41_teacher.py --seqs 调用)。

干什么: 后训练 ③ 要向出厂原件蒸馏(core_ptrain_fp.c), 料是几百条各自独立的短序列(一篇文档 / 一段思考尾段, ≤1024 位)。
这里把它们放进官方 model.py 的 batch 维一次前向, 每条每位只落 top-K + 余量(512 B/位), 不落全 logits(517 KB/位: 500k 位 = 258 GB, 盘装不下)。

为什么不拼成一条长序列: 教师看到的上下文必须与学生训练时一样 —— 学生一题一条、从 BOS 起; 拼接后后面的题带着前面题的上下文, 教师分布就不是学生那道题的分布。
为什么不一条一趟: 流式装一遍 476 GB 原件 ≈ 25 分钟(10-21: 12k~15k 位 1,622 s, 大头是读盘不是算), 500 条就是 200 小时。
为什么 batch 维可以: 官方 model.py 本来就按 [bsz, seqlen] 写(缓存按 max_batch_size 分, 掩码只有因果), 右侧补齐的位置在因果注意力下影响不到前面任何一位; 补的位置不落表。
★不重写 forward★: 只换输出头(逐条取 top-K, 免得 [B][S][V] 全 logits 落显存)和输入形状, 层逻辑一个字不碰。

表格式(与 core_ptrain_fp.c 的读法一一对应; 文件可多段首尾相接, cat 即合并):
  u32 magic 'PTF1' | u32 K | u32 N; 随后 N 条: u32 L | i32 ids[L] | i32 top_id[(L-1)·K] | f32 top_p[(L-1)·K] | f32 rest[L-1]
  第 i 行 = 位置 i 的 logits 预测 ids[i+1]。
"""
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F

MAGIC = 0x31465450  # 'PTF1' 小端


def load_seqs(path):
    seqs = []
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            seqs.append([int(x) for x in ln.split()])
    if not seqs:
        raise SystemExit(f"★--seqs {path} 里没有序列★")
    for i, s in enumerate(seqs):
        if len(s) < 2:
            raise SystemExit(f"★--seqs 第 {i} 条只有 {len(s)} 位, 没有可预测的位置★")
    return seqs


def shape_of(path):
    """建模前要知道 (条数, 最长): max_batch_size / max_seq_len 决定官方缓存的尺寸, meta 建模后改不了。"""
    seqs = load_seqs(path)
    return len(seqs), max(len(s) for s in seqs)


def run(a, net, dev, pad_id, t0):
    seqs = load_seqs(a.seqs)
    B, L, K = len(seqs), max(len(s) for s in seqs), a.topk
    x = torch.full((B, L), pad_id, dtype=torch.long, device=dev)
    for b, s in enumerate(seqs):
        x[b, : len(s)] = torch.tensor(s, dtype=torch.long, device=dev)
    print(f"[seqs] {B} 条, 最长 {L}, 合计 {sum(len(s) for s in seqs)} 位, top-{K}; 补齐 id {pad_id}", flush=True)

    # 输出头换成"逐条算 logits → 立刻取 top-K": 官方 ParallelHead 一次出 [B][S][V] f32(64 条 × 1024 位 = 34 GB), 这里一条 0.5 GB 用完即弃。
    # 返回每条末位的 logits([B][V]), 官方 forward 末尾的 sample(argmax) 照常跑, 我们不用它。
    out = [None] * B
    head = net.head
    W = None

    def head_fwd(h, full_logits=True):
        nonlocal W
        W = head.weight          # 流式 hook 在模块调用时已把它装上 GPU
        last = []
        for b in range(B):
            n = len(seqs[b]) - 1
            lg = F.linear(h[b, : n + 1].float(), W)           # [n+1][V]: 前 n 行预测 ids[1..n], 末行只给 sample 用
            p = torch.softmax(lg[:n], dim=-1)
            tp, ti = p.topk(K, dim=-1)
            rest = (1.0 - tp.sum(dim=-1)).clamp_(min=0.0)
            out[b] = (ti.to(torch.int32).cpu().numpy(), tp.float().cpu().numpy(), rest.float().cpu().numpy(), lg[:n].argmax(dim=-1).cpu())
            last.append(lg[n])
            del lg, p
        return torch.stack(last)
    head.forward = head_fwd

    t1 = time.time()
    net(x)
    torch.cuda.synchronize()
    print(f"[完成] {time.time()-t1:.1f}s(建模起 {time.time()-t0:.1f}s)", flush=True)

    hit = tot = 0
    with open(a.out, "wb") as f:
        f.write(struct.pack("<III", MAGIC, K, B))
        for b, s in enumerate(seqs):
            ti, tp, rest, am = out[b]
            n = len(s) - 1
            tgt = torch.tensor(s[1:], device=am.device)   # 主文件已 set_default_device(cuda), 不指明就与搬到 CPU 的 argmax 不同设备(10-22 第一趟实撞: 36 分钟前向跑完在这一行崩)
            hit += int((am == tgt).sum()); tot += n
            f.write(struct.pack("<I", len(s)))
            np.asarray(s, dtype=np.int32).tofile(f)
            ti.reshape(-1).tofile(f); tp.reshape(-1).tofile(f); rest.tofile(f)
    print(f"  → {a.out}: {B} 条 {tot} 行 × top-{K}; teacher-forcing argmax 命中 {hit}/{tot} = {100.0*hit/max(tot,1):.2f}%"
          f"(与判决尺的 FP 口径同一个前向; 命中率 50%~85% 是正常文本, 远低于此先怀疑补齐/形状)", flush=True)
