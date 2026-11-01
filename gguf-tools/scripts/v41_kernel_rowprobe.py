#!/usr/bin/env python3
"""v41_kernel_rowprobe.py — 官方 kernel.py 里教师仍在跑的三个 tilelang kernel 在 GB10 上的行无关性探针(2026-10-25)。

为什么要有它: FP 教师(v41_teacher.py)在 500 位之后回看失灵(抄 1500 位的第二份错块、工具 schema 段 NLL 5 对引擎 1.3),
sparse_attn/索引器/压缩器/RoPE/YaRN/预填方式逐行对过官方都没差; 教师里唯一没验过数值的是官方 act_quant / fp4_act_quant /
hc_split_sinkhorn 三个 tilelang kernel —— 它们按 32 行一个 tile 切序列维, 照数据中心 Blackwell 写的, 在 GB10 上"能跑"
不等于"算对", 行 tile 调度错的症状正是"按位置坏"。
怎么测: 同一行数据单独量化(放第 0 行)与放在第 r 行一起量化, 结果必须逐位相同; 再和纯 torch 参考实现比, 抓"每行都错"的那种。
用法(spark, 不需要模型, GPU 上只放几十 MB):
    ~/v41env/bin/python gguf-tools/scripts/v41_kernel_rowprobe.py hf/DeepSeek-V4.1-Flash-1017 [S=3001]
出错会怎样: 某个 kernel 的某些行 maxdiff > 0 ⇒ 教师在那些位置用的是错的量化值, 病根就在 kernel; 全 0 ⇒ 这三个 kernel 无罪, 嫌疑回到别处。
"""
import json
import sys
from pathlib import Path

import torch

hf = Path(sys.argv[1])
S = int(sys.argv[2]) if len(sys.argv) > 2 else 3001
sys.path.insert(0, str(hf / "inference"))
import kernel as K  # noqa: E402

cfg = json.loads((hf / "inference" / "config.json").read_text())
dev = "cuda"
torch.manual_seed(0)
ROWS = [0, 1, 31, 32, 100, 400, 511, 512, 513, 600, 1000, 1500, 2000, 2990, S - 1]
ROWS = [r for r in ROWS if r < S]


def pow2_ceil(a):
    return torch.exp2(torch.ceil(torch.log2(a)))


def e2m1_rne(v):
    """e2m1 的 RNE: 网格 0 .5 1 1.5 2 3 4 6, 平手取尾数位为 0 的那个(0 1 2 4 6)。"""
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=v.device)
    even = torch.tensor([1, 0, 1, 0, 1, 0, 1, 1], device=v.device, dtype=torch.bool)
    a = v.abs().clamp(max=6).unsqueeze(-1)
    d = (a - grid).abs()
    dmin = d.amin(-1, keepdim=True)
    cand = (d == dmin)
    tie = cand.sum(-1, keepdim=True) > 1
    pick = torch.where(tie, cand & even, cand).float().argmax(-1)
    return grid[pick] * v.sign()


def ref_fp8_inplace(x, blk=32):
    xf = x.float().unflatten(-1, (-1, blk))
    amax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    s = pow2_ceil(amax / 448.0)                       # ue8m0: 2 的幂, 向上取
    q = (xf / s).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (q * s).flatten(-2).to(x.dtype)


def ref_fp4_inplace(x, blk, e4m3_scale):
    xf = x.float().unflatten(-1, (-1, blk))
    amax = xf.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        amax = amax.clamp(min=6 * 2 ** -9)
        s = (amax / 6).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp(min=6 * 2 ** -126)
        s = pow2_ceil(amax / 6)
    return (e2m1_rne(xf / s) * s).flatten(-2).to(x.dtype)


def check(name, x, fn, ref=None):
    full = fn(x.clone())
    worst = 0.0
    bad = []
    for r in ROWS:
        one = fn(x[:, r:r + 1].clone())
        d = (full[:, r].float() - one[:, 0].float()).abs().max().item()
        worst = max(worst, d)
        if d > 0:
            bad.append((r, d))
    # 行数补到 32 的倍数再算一遍: 看尾 tile(S % 32 = 25 行)有没有处理错
    pad = (-S) % 32
    xp = torch.cat([x, x[:, :pad]], dim=1) if pad else x
    fp = fn(xp.clone())[:, :S]
    dpad = (full.float() - fp.float()).abs().max().item()
    line = f"[{name}] 单行 vs 整批 maxdiff {worst:.3g}(坏行 {bad[:6]}) | 补齐 32 行 vs 原长 maxdiff {dpad:.3g}"
    if ref is not None:
        rf = ref(x.clone())
        dr = (full.float() - rf.float()).abs()
        nbad = (dr > 0).sum().item()
        line += f" | vs torch 参考 maxdiff {dr.max().item():.3g}, 不同元素 {nbad}/{dr.numel()}"
        if nbad:
            per_row = (dr.flatten(2).amax(-1) > 0).squeeze(0) if dr.ndim > 2 else (dr > 0).any(-1).squeeze(0)
            idx = per_row.nonzero().squeeze(-1)
            line += f", 首个不同行 {idx[:5].tolist()}, 不同行数 {idx.numel()}/{S}"
    print(line, flush=True)


hd = cfg["head_dim"]
# ① 窗口 K / 注意力输出 o: act_quant fp8 blk 32 ue8m0 原地(model.py 708 / 787 行)
x = (torch.randn(1, S, hd, device=dev) * 0.7).to(torch.bfloat16)
check("act_quant fp8 blk32 ue8m0 [1,S,512]", x,
      lambda t: K.act_quant(t, 32, "ue8m0", torch.float8_e8m0fnu, True), lambda t: ref_fp8_inplace(t, 32))
# ② 索引器 q [1,S,32,128] 与 k [1,S,128]: fp4 blk 32 e8m0 原地(546 / 553 行)
q = (torch.randn(1, S, cfg["index_n_heads"], cfg["index_head_dim"], device=dev) * 2).to(torch.bfloat16)
check("fp4_act_quant blk32 e8m0 索引 q [1,S,32,128]", q,
      lambda t: K.fp4_act_quant(t, 32, True), lambda t: ref_fp4_inplace(t, 32, False))
k = (torch.randn(1, S, cfg["index_head_dim"], device=dev) * 2).to(torch.bfloat16)
check("fp4_act_quant blk32 e8m0 索引 k [1,S,128]", k,
      lambda t: K.fp4_act_quant(t, 32, True), lambda t: ref_fp4_inplace(t, 32, False))
# ③ 压缩 KV latent: fp4 blk 16, e4m3 scale 原地(761 行)
lat = (torch.randn(1, S, hd, device=dev) * 1.5).to(torch.bfloat16)
check("fp4_act_quant blk16 e4m3 压缩 KV [1,S,512]", lat,
      lambda t: K.fp4_act_quant(t, 16, True, scale_dtype=torch.float8_e4m3fn), lambda t: ref_fp4_inplace(t, 16, True))
# ④ 超连接 sinkhorn(957 行): mixes [1,S,(2+hc)*hc] f32 → pre/post/comb 拼成一行比
hc = cfg["hc_mult"]
mixes = torch.randn(1, S, (2 + hc) * hc, device=dev)
sc = torch.randn(3, device=dev) * 0.5
base = torch.randn((2 + hc) * hc, device=dev) * 0.5


def hc_fn(t):
    pre, post, comb = K.hc_split_sinkhorn(t, sc, base, hc, cfg["hc_sinkhorn_iters"], cfg["hc_eps"])
    return torch.cat([pre, post, comb.flatten(2)], dim=-1)


check("hc_split_sinkhorn [1,S,24]", mixes, hc_fn)
print("done", flush=True)
