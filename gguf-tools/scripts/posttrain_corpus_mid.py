#!/usr/bin/env python3
# posttrain_corpus_mid.py — ③ 加料: 从本地语料取"长文档里连续的 1000-token 段", 剔掉跟判决份沾边的整篇文档(2026-10-23)。
#
# 用法(spark, ~/ds4-main):
#   ~/v41env/bin/python gguf-tools/scripts/posttrain_corpus_mid.py --hf hf/DeepSeek-V4.1-Flash-1017 \
#       --judge gguf/go-onebit/code/judge.ids --judge gguf/go-onebit/g7/wt2.ids \
#       --src gguf-tools/data/corpus/code:750 --src gguf-tools/data/corpus/wiki.train.raw:250 ... --out <料.jsonl>
#   --src <目录或文件>:<段数>   目录 = 每个 .jsonl 一个子域(取 "text"), 段数在子域间平分; wiki.train.raw = 按 " = 标题 = " 切文章
# 产物: <out>(一行 {"text": 段}, 段 = 某篇文档里连续 1000 个 token 解码回的原文) + <out>.manifest.txt(每个子域取了几篇几段、剔了几篇)
#       + <out>.docmap.txt(第 k 行 = 第 k 段出自哪篇: 子域:篇号)
# ★留出会漏题★: 一篇文档最多出 4 段相邻文本, 训练器按段做哈希分留出, 同篇的兄弟段会一段留出、一段训练 ——
#   留出读数因此虚高(见过同一篇的变量名/写法)。用这份料时留出必须按 docmap 整篇分。
#
# 为什么取长文档的连续段: 判决份的窗是长文件中段的 128 位, 而之前 ③ 的料全是 ≤1000 token 的短整文件(从文件头开始),
#   10-23 逐位配对: 第二刀在判决份上是 50/50 的随机搅动 —— 料的形态与判决份不同分布是候选原因之一。
# 为什么 1000 不是 1024: 训练器 maxlen 1024 含 BOS; 解码再分词偶尔多一两个 token, 留余量免得被切出一个几 token 的尾巴样本。
# ★不泄题★: 判决份(编程 judge.ids 15360 位 + wt2.ids 全部)每 16 个连续 token 做一个指纹; 候选文档只要有一个 16-token 片段命中,
#   整篇不要(不只是那一段) —— 判决窗来自哪篇文档, 那篇文档的任何部分都不进训练。16 token 在代码里偶尔撞许可证头这类样板, 宁可多剔。
# 不做数值: 只分词、解码、查重、挑文档; 选哪几篇是确定性的(按文件内顺序等距取), 同参数重跑逐字节同。
import argparse, json, os, sys
from tokenizers import Tokenizer

SEG = 1000      # 每段 token 数(见上: 留 24 个给 BOS 与重分词的漂移)
NG = 16         # 泄题指纹的片段长度
MAX_SEG_PER_DOC = 4   # 一篇文档最多取几段: 不让少数超长文档占满配额(多篇 > 一篇的多段)


def judge_prints(paths):
    s = set()
    for p in paths:
        ids = [int(x) for x in open(p).read().split()]
        for i in range(len(ids) - NG + 1):
            s.add(tuple(ids[i:i + NG]))
    return s


def docs_of(path):
    """一个子域的文档列表(原文)"""
    if path.endswith(".jsonl"):
        return [json.loads(ln)["text"] for ln in open(path, encoding="utf-8") if ln.strip()]
    # wiki.train.raw: 顶层标题行 " = X = " 开一篇新文章(二级标题 " = = X = = " 不算)
    out, cur = [], []
    for ln in open(path, encoding="utf-8"):
        t = ln.strip()
        if t.startswith("= ") and t.endswith(" =") and not t.startswith("= ="):
            if cur: out.append("".join(cur))
            cur = []
        cur.append(ln)
    if cur: out.append("".join(cur))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--judge", action="append", required=True)
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = Tokenizer.from_file(os.path.join(a.hf, "tokenizer.json"))
    jp = judge_prints(a.judge)
    out, man, docmap = [], [], []   # docmap[k] = 第 k 段出自哪篇文档(子域:篇号) —— 同篇相邻段被训练器的留出拆开 = 留出里有兄弟段进了训练, 读数会虚高
    for spec in a.src:
        src, n = spec.rsplit(":", 1)
        n = int(n)
        subs = [os.path.join(src, f) for f in sorted(os.listdir(src)) if f.endswith(".jsonl")] if os.path.isdir(src) else [src]
        per = [n // len(subs) + (1 if i < n % len(subs) else 0) for i in range(len(subs))]
        for sub, want in zip(subs, per):
            docs = docs_of(sub)
            # 等距取: 从全部文档里均匀挑, 不是只吃文件开头那几篇
            order = sorted(range(len(docs)), key=lambda i: ((i * 2654435761) % 4294967296, i))
            got, used, leak = 0, 0, 0
            for i in order:
                if got >= want: break
                ids = tok.encode(docs[i], add_special_tokens=False).ids
                if any(tuple(ids[k:k + NG]) in jp for k in range(len(ids) - NG + 1)):
                    leak += 1
                    continue
                if len(ids) < SEG // 4: continue   # 太短的文档不要: 要的是"长文档里的一段"
                used += 1
                for s in range(0, min(len(ids), SEG * MAX_SEG_PER_DOC), SEG):
                    if got >= want: break
                    piece = ids[s:s + SEG]
                    if len(piece) < SEG // 4: break
                    out.append(tok.decode(piece))
                    docmap.append(f"{os.path.basename(sub)}:{i}")
                    got += 1
            man.append(f"{os.path.basename(sub)}\t段 {got}/{want}\t用文档 {used}\t剔(撞判决) {leak}\t共 {len(docs)} 篇")
            if got < want: man[-1] += "\t★配额没取满★"
    with open(a.out, "w", encoding="utf-8") as f:
        for t in out: f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    with open(a.out + ".docmap.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(docmap) + "\n")
    with open(a.out + ".manifest.txt", "w", encoding="utf-8") as f:
        f.write(f"# posttrain_corpus_mid.py: {len(out)} 段, 每段 {SEG} token, 泄题指纹 {NG}-gram × {len(jp)} 个({' '.join(a.judge)})\n")
        f.write("\n".join(man) + "\n")
    print("\n".join(man)); print(f"共 {len(out)} 段 → {a.out}")


if __name__ == "__main__":
    sys.exit(main())
