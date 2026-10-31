#!/usr/bin/env python3
"""ifeval_grade.py — IFEval 判分, 用 Google 官方 instruction_following_eval(vendored 在 gguf-tools/bench/ifeval_official/),
prompt 级 strict 口径(unsloth 同). 跑在 Mac 上(要 nltk punkt_tab / langdetect / immutabledict, spark 上没装)。

用法: python3 ifeval_grade.py --out <从 spark 拉回来的 ifeval 目录> --data gguf-tools/bench/data/ifeval.jsonl
产物: <out>/graded.jsonl + 更新 <out>/summary.json 的 pass_at_1(= prompt-level strict accuracy)。
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bench", "ifeval_official"))
from instruction_following_eval import evaluation_lib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(args.data) if l.strip()}
    recs = [json.loads(l) for l in open(os.path.join(args.out, "answers.jsonl")) if l.strip()]
    graded, npass = [], 0
    for r in recs:
        raw = json.load(open(os.path.join(args.out, "raw", f"{r['id']}_{r['sample']}.json")))
        content = raw["response"]["choices"][0]["message"].get("content") or ""
        m = rows[r["id"]]["meta"]
        inp = evaluation_lib.InputExample(key=m["key"], instruction_id_list=m["instruction_id_list"], prompt=m["prompt"],
                                          kwargs=[{k: v for k, v in kw.items() if v is not None} for kw in m["kwargs"]])
        out = evaluation_lib.test_instruction_following_strict(inp, {m["prompt"]: content})
        ok = bool(out.follow_all_instructions); npass += ok
        graded.append({"id": r["id"], "sample": r["sample"], "passed": ok, "per_instruction": out.follow_instruction_list,
                       "truncated": r["truncated"]})
    with open(os.path.join(args.out, "graded.jsonl"), "w") as f:
        for g in graded:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    sp = os.path.join(args.out, "summary.json")
    s = json.load(open(sp)) if os.path.exists(sp) else {}
    n = len(graded); acc = npass / n if n else None
    s.update({"graded": n, "pass_at_1": None if acc is None else round(100 * acc, 2),
              "se_pct": None if acc is None else round(100 * (acc * (1 - acc) / n) ** 0.5, 2), "grader": "ifeval_grade.py (official strict, prompt level)"})
    json.dump(s, open(sp, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: s.get(k) for k in ("answers", "graded", "pass_at_1", "se_pct", "truncated_pct")}), flush=True)


if __name__ == "__main__":
    main()
