#!/usr/bin/env python3
"""judge_gate_rows.py — 把两把尺的"门半边"窗变成段 B 训练器能读的料行, 专给参数锚攒 F 用(2026-10-31)。

为什么: 参数锚(core_restoreb_init.c)按 F 决定哪些坐标不许动; t005 用 o010 裸续写料攒的 F 没按住编程尺(85.03 → 84.95, 门半边 KLD +1.3 SE),
  说明那份料在乎的坐标和尺在乎的不重合。F 要保尺, 就在尺自己的文字上量梯度 —— 这不是拿尺训练(lr=0 只攒 g², 参数一位不动), 只是量方向。
  只用门半边(code_gate.lst: 每种语言第 1/3 窗)+ wt2 前半; 报数半边(第 2/4 窗)和 wt2 后半不碰, 留着验证锚有没有"玩尺"。
怎么用(spark 仓库根):
  python3 gguf-tools/scripts/judge_gate_rows.py --tokenizer <HF 目录或 tokenizer.json> --out <目录>/rows.jsonl
    读 gguf/go-onebit/code/judge.ids + judge.ids.layout(win 128; 每种语言 起点 长度, 4 窗取第 0/2 窗) 与 gguf/go-onebit/g7/wt2.ids 前 256 位,
    每窗解码成文字写一行 {"text": …, "eos": false}。训练器自己再分词(BOS + text), 教师表用训练器导出的序列打(posttrain_fp_spark.sh ①②), 配表按训练器的序列, 不按这里的 id。
出错会怎样: 窗起点/长度与 ids 文件对不上就停; 解码后再编码与原 id 不同只打印计数(不影响: 教师按训练器的序列打)。"""
import argparse, json, os
from tokenizers import Tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--code-ids", default="gguf/go-onebit/code/judge.ids"); ap.add_argument("--layout", default="gguf/go-onebit/code/judge.ids.layout")
    ap.add_argument("--wt2-ids", default="gguf/go-onebit/g7/wt2.ids"); ap.add_argument("--wt2-n", type=int, default=256)
    a = ap.parse_args()
    tp = a.tokenizer if a.tokenizer.endswith(".json") else os.path.join(a.tokenizer, "tokenizer.json")
    tok = Tokenizer.from_file(tp)
    ids = [int(l) for l in open(a.code_ids) if l.strip()]
    win = 128; rows = []; mism = 0
    for l in open(a.layout):
        f = l.split()
        if not f or f[0].startswith("#"): continue
        if f[0] == "win": win = int(f[1]); continue
        if len(f) != 3: continue
        name, s, n = f[0], int(f[1]), int(f[2])
        if s + n > len(ids): raise SystemExit(f"★{name} 起点 {s} 长 {n} 超出 ids {len(ids)}★")
        for k in range(n // win):
            if k % 2: continue   # 第 1/3 窗进门(与 restore_B_api_round_spark.sh 的 code_gate.lst 同一条规则), 第 2/4 窗只报数, 不进 F
            seg = ids[s + k * win: s + (k + 1) * win]
            txt = tok.decode(seg, skip_special_tokens=False)
            if tok.encode(txt, add_special_tokens=False).ids != seg: mism += 1
            rows.append({"text": txt, "eos": False, "src": f"{name}#w{k}"})
    w = [int(l) for l in open(a.wt2_ids) if l.strip()][: a.wt2_n]
    rows.append({"text": tok.decode(w, skip_special_tokens=False), "eos": False, "src": f"wt2#0-{len(w)}"})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps({"text": r["text"], "eos": r["eos"]}, ensure_ascii=False) + "\n")
    print(f"rows {len(rows)} 行(编程门半边 {len(rows) - 1} 窗 × {win} 位 + wt2 前 {len(w)} 位) → {a.out}; 解码再编码与原 id 不同的窗 {mism}(不影响: 教师按训练器序列打)")


if __name__ == "__main__":
    main()
