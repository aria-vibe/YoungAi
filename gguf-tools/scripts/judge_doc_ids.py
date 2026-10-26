#!/usr/bin/env python3
# judge_doc_ids.py — 部署态格式的编程判决份(2026-10-24): BOS + 整篇文件从头, 一种语言一篇, 每篇 512 位(BOS + 511), 30 语言 = 15360 位。
#
# 用法(spark, ~/ds4-main):
#   ~/v41env/bin/python gguf-tools/scripts/judge_doc_ids.py --hf hf/DeepSeek-V4.1-Flash-1017 --judge gguf/go-onebit/code/judge.ids \
#       --corpus gguf-tools/data/corpus/code --train gguf-tools/data/posttrain/night_1023.jsonl --train gguf-tools/data/posttrain/code_only_1024.jsonl \
#       --out gguf/go-onebit/code/judge_doc.ids
# 产物: <out>(一行一个 token, v41_judge.sh 直接吃) + <out>.layout(与 judge.ids.layout 同格式, 每语言 512 位) + <out>.docs.txt(每篇出处)
#
# 为什么要第二把尺: judge.ids 是 120 个 128 位窗首尾相接、无 BOS、相邻窗来自不相干文件(08-29 为反修定的格式);
#   10-24 实验: 同 6 篇没见过的文档, ③ 第二刀在整篇格式上 KLD −4.1% / 拼窗格式 +0.6% —— 收益是格式相关的, 拼窗尺看不见部署态(BOS + 连贯文本)的收益。
#   judge.ids 不动(反修的尺照旧), 这把并排报。
# 选哪些文档: 含 judge.ids 任一 16-gram 的文档 = 判决窗的出处(判决半边, 从没进过任何训练料: 训练料造的时候就是按这个指纹剔的), 每语言按哈希序取第一篇 ≥511 token 的;
#   再对 --train 给的全部训练料做一遍 16-gram 核对, 命中就停 —— 这把尺要是泄题, 读数就是假的。
# 为什么是文件头 511 而不是整篇: ③ 的训练序列也是 BOS + 文件头 ≤1000, 这把尺量的就是"训练那种格式"; 整篇的话一篇几千 token, 30 篇装不进 15360。
# 不做数值: 只分词、查重、切 ids; 同参数重跑逐字节同。
import argparse, json, os, sys
from tokenizers import Tokenizer

NG, PER, BOS = 16, 511, 0


def grams(ids):
    return {tuple(ids[i:i + NG]) for i in range(len(ids) - NG + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True); ap.add_argument("--judge", required=True); ap.add_argument("--corpus", required=True)
    ap.add_argument("--train", action="append", default=[]); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = Tokenizer.from_file(os.path.join(a.hf, "tokenizer.json"))
    jp = grams([int(x) for x in open(a.judge).read().split()])
    tp = set()
    for t in a.train:
        for ln in open(t, encoding="utf-8"):
            if ln.strip(): tp |= grams(tok.encode(json.loads(ln)["text"], add_special_tokens=False).ids)
    out, lay, docs = [], [], []
    for fn in sorted(os.listdir(a.corpus)):
        if not fn.endswith(".jsonl"): continue
        rows = [json.loads(ln)["text"] for ln in open(os.path.join(a.corpus, fn), encoding="utf-8") if ln.strip()]
        order = sorted(range(len(rows)), key=lambda i: ((i * 2654435761) % 4294967296, i))
        pick, skipped = None, 0
        for i in order:
            ids = tok.encode(rows[i], add_special_tokens=False).ids
            if len(ids) < PER or not (grams(ids) & jp): continue
            hit = grams(ids) & tp
            # 与训练料共享片段的不要(实撞 code_csharp 第 18 篇 3 个 16-gram: code5 当年对判决份只做了 24-gram 剔重, 样板行漏过去了), 换下一篇
            if hit: skipped += 1; print(f"  {fn} 第 {i} 篇与训练料共享 {len(hit)} 个 {NG}-gram, 跳过"); continue
            pick = (i, ids); break
        if not pick: print(f"★{fn}: 没有 ≥{PER} token、含判决窗、且不与训练料共享 {NG}-gram 的文档(跳过 {skipped} 篇)★"); return 2
        i, ids = pick
        lay.append(f"{fn[:-6]} {len(out)} {PER + 1}")
        out += [BOS] + ids[:PER]
        docs.append(f"{fn}\t篇 {i}\t全文 {len(ids)} token, 取 BOS + 前 {PER}")
    open(a.out, "w").write("\n".join(map(str, out)) + "\n")
    with open(a.out + ".layout", "w") as f:
        f.write(f"# {os.path.basename(a.out)} 的行布局 — judge_doc_ids.py 产出, 勿手改\n# 每语言一篇: BOS + 文件头 {PER} token(部署态格式); 每行: <域名> <起始行> <行数>\nwin {PER + 1}\n")
        f.write("\n".join(lay) + "\n")
    open(a.out + ".docs.txt", "w", encoding="utf-8").write("\n".join(docs) + "\n")
    print("\n".join(docs)); print(f"共 {len(out)} 位 → {a.out}; 训练料 {NG}-gram 指纹 {len(tp)} 个, 0 命中")


if __name__ == "__main__":
    sys.exit(main())
