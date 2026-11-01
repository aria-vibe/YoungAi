#!/usr/bin/env python3
# leak_audit.py — 泄题审计(2026-10-24 用户令"再看看有没有泄题的"): 任何要进侧车/蒸馏的料, 对全部判决份做 16-gram 指纹核对, 只报不改。
#
# 用法(spark ~/ds4-main):
#   ~/v41env/bin/python gguf-tools/scripts/leak_audit.py --hf hf/DeepSeek-V4.1-Flash-1017 \
#       --judge gguf/go-onebit/code/judge.ids --judge gguf/go-onebit/g7/wt2.ids --judge gguf/go-onebit/law/judge.ids ... \
#       --ids gguf/go-onebit/restore/fit.ids --jsonl gguf-tools/data/posttrain/formal2_1022.jsonl --jsonl gguf-tools/data/posttrain/hold.jsonl
#   --ids   一份 ids(一行一个 token): 按 128 位窗报(窗号 / 命中数 / 片段样例)
#   --jsonl 一份料: {"text": …} 原样分词; {"messages": […]} 把各段 content 拼起来分词(模板 token 不进料, 只看正文)
# 输出: 每份料一行汇总(篇/窗数, 命中篇数, 命中 16-gram 总数)+ 命中最多的前 8 篇各一条解码样例 + 命中了哪份判决份。
# 判读: 命中是事实, 是不是"样板"(HTML 头 / 许可证头 / 题干格式句)要人看样例定 —— 这把尺不替人判, 10-24 下午两次按"任一命中"自动剔都剔错了。
# 不做数值: 只分词、查集合。
import argparse, json, os, sys
from tokenizers import Tokenizer

NG = 16


def grams(ids):
    return [tuple(ids[i:i + NG]) for i in range(len(ids) - NG + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True); ap.add_argument("--judge", action="append", required=True)
    ap.add_argument("--ids", action="append", default=[]); ap.add_argument("--jsonl", action="append", default=[])
    ap.add_argument("--top", type=int, default=8)
    # --clean-out DIR(10-24 用户令"看看有没有泄题的"后的落地): 每份 --jsonl 另写一份 <名>.clean.jsonl 到 DIR, 命中任一判决份 16-gram 的整条剔掉; 原文件不动
    ap.add_argument("--clean-out")
    a = ap.parse_args()
    tok = Tokenizer.from_file(os.path.join(a.hf, "tokenizer.json"))
    J = {}   # gram → 判决份名
    for p in a.judge:
        ids = [int(x) for x in open(p).read().split()]
        for g in grams(ids): J.setdefault(g, os.path.relpath(p))
    print(f"判决份 {len(a.judge)} 份, {len(J)} 个 {NG}-gram 指纹")

    def report(name, units):   # units = [(标签, ids)]; 返回命中单元的标签集合
        hit = []
        total = 0
        for lab, ids in units:
            gs = [g for g in grams(ids) if g in J]
            total += len(gs)
            if gs: hit.append((len(gs), lab, gs))
        hit.sort(reverse=True)
        print(f"== {name}: {len(units)} 单元, 命中 {len(hit)} 单元 / {total} 个 {NG}-gram")
        for n, lab, gs in hit[: a.top]:
            srcs = sorted({J[g] for g in gs})
            print(f"   {lab}: {n} 命中 ← {', '.join(srcs)}  例 {tok.decode(list(gs[0]))[:70]!r}")
        return {lab for _, lab, _ in hit}

    for p in a.ids:
        ids = [int(x) for x in open(p).read().split()]
        units = [(f"窗 {k // 128}", ids[k:k + 128]) for k in range(0, len(ids) - 127, 128)]
        report(os.path.relpath(p), units)
    for p in a.jsonl:
        units, lines = [], []
        for i, ln in enumerate(open(p, encoding="utf-8")):
            if not ln.strip(): continue
            d = json.loads(ln)
            text = d["text"] if "text" in d else "\n".join(m.get("content", "") for m in d.get("messages", []))
            units.append((f"第 {i} 条", tok.encode(text, add_special_tokens=False).ids)); lines.append((f"第 {i} 条", ln))
        bad = report(os.path.relpath(p), units)
        if a.clean_out:
            os.makedirs(a.clean_out, exist_ok=True)
            q = os.path.join(a.clean_out, os.path.basename(p).replace(".jsonl", ".clean.jsonl"))
            with open(q, "w", encoding="utf-8") as f:
                for lab, ln in lines:
                    if lab not in bad: f.write(ln if ln.endswith("\n") else ln + "\n")
            print(f"   → {q}: 留 {len(lines) - len(bad)} / 剔 {len(bad)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
