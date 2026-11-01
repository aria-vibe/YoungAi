#!/usr/bin/env python3
"""bench_prep.py — 把各基准的原始数据集整理成 bench_suite.py 吃的 jsonl(每行 id / prompt / gold / grade / system? / meta?)。
跑在 Mac 上(要 pyarrow), 产物进 gguf-tools/bench/data/, 再 scp 到 spark。题面模板照 unsloth 页 Test Methodology:
  AIME25 / HMMT: 题面 + "Please reason step by step, and put your final answer within \\boxed{}."
  GPQA-D / MMLU-Pro: OpenAI simple-evals 多选模板("Answer: $LETTER" 那段)
  IFEval: 原 prompt
  LCB v6: LiveCodeBench 官方 system + user 模板; 测试用例另存 lcb_tests.jsonl 给 lcb_grade.py
用法: bench_prep.py --src <下载目录> --dst gguf-tools/bench/data
  <src>/MathArena_aime_2025/data/*.parquet  <src>/MathArena_hmmt_feb_2026/data/*.parquet
  <src>/gpqa_mc/data/*.parquet(hendrydong/gpqa_diamond_mc: 题面已含选项, 答案 \\boxed{X})
  <src>/TIGER-Lab_MMLU-Pro/data/test-*.parquet  <src>/google_IFEval/ifeval_input_data.jsonl  <src>/lcb/test6.jsonl
"""
import argparse, base64, glob, json, os, pickle, random, re, zlib

BOXED_TAIL = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
MC_HEAD = ("Answer the following multiple choice question. The last line of your response should be of the following format: "
           "'Answer: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.\n\n")
LCB_SYSTEM = ("You are an expert Python programmer. You will be given a question (problem specification) and will generate a "
              "correct Python program that matches the specification and passes all tests.")
LCB_FMT_STARTER = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
LCB_FMT_STDIN = ("Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). "
                 "Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, "
                 "runs the algorithm and writes output to STDOUT.")
SEED = 20261029
MMLU_SUBSET = 1203   # unsloth 的 1,203 题分层子集(他们没公开抽法, 这里按类别比例抽, 种子固定)


def parquet_rows(pattern):
    import pyarrow.parquet as pq
    rows = []
    for p in sorted(glob.glob(pattern)):
        rows += pq.read_table(p).to_pylist()
    return rows


def write(dst, name, rows):
    p = os.path.join(dst, name)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} 行")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args(); os.makedirs(a.dst, exist_ok=True)

    for tag, sub in (("aime25", "MathArena_aime_2025"), ("hmmt_feb26", "MathArena_hmmt_feb_2026")):
        rows = parquet_rows(f"{a.src}/{sub}/data/*.parquet")
        if rows:
            write(a.dst, f"{tag}.jsonl", [{"id": str(r["problem_idx"]), "prompt": r["problem"] + BOXED_TAIL,
                                           "gold": str(r["answer"]), "grade": "math"} for r in rows])

    rows = parquet_rows(f"{a.src}/gpqa_mc/data/*.parquet")
    if rows:
        out = []
        for i, r in enumerate(rows):
            m = re.search(r"\\boxed\{([A-D])\}", r["solution"])
            # 镜像题面末尾自带一句 "Please write your final answer in the form of \boxed{A}…", 与 simple-evals 的 "Answer: X" 冲突, 去掉
            body = re.sub(r"\s*Please write your final answer in the form of[^\n]*$", "", r["problem"].strip())
            out.append({"id": f"g{i:03d}", "prompt": MC_HEAD.format(letters="ABCD") + body, "gold": m.group(1),
                        "grade": "mc", "meta": {"domain": r["domain"]}})
        write(a.dst, "gpqa_diamond.jsonl", out)

    rows = parquet_rows(f"{a.src}/TIGER-Lab_MMLU-Pro/data/test-*.parquet")
    if rows:
        rnd = random.Random(SEED); by = {}
        for r in rows:
            by.setdefault(r["category"], []).append(r)
        total = len(rows); picked = []
        for cat, lst in sorted(by.items()):
            k = round(len(lst) / total * MMLU_SUBSET); rnd.shuffle(lst); picked += lst[:k]
        out = []
        for r in picked:
            opts = [o for o in r["options"] if o != "N/A"]
            letters = "ABCDEFGHIJ"[:len(opts)]
            body = r["question"].strip() + "\n\n" + "\n".join(f"{letters[i]}) {o}" for i, o in enumerate(opts))
            out.append({"id": str(r["question_id"]), "prompt": MC_HEAD.format(letters=letters) + body, "gold": r["answer"],
                        "grade": "mc", "meta": {"category": r["category"]}})
        write(a.dst, "mmlu_pro.jsonl", out)

    p = f"{a.src}/google_IFEval/ifeval_input_data.jsonl"
    if os.path.exists(p):
        rows = [json.loads(l) for l in open(p) if l.strip()]
        write(a.dst, "ifeval.jsonl", [{"id": str(r["key"]), "prompt": r["prompt"], "gold": None, "grade": "raw", "meta": r} for r in rows])

    p = f"{a.src}/lcb/test6.jsonl"
    if os.path.exists(p):
        out, tests = [], []
        for l in open(p):
            r = json.loads(l)
            if r["starter_code"]:
                fmt = f"### Format: {LCB_FMT_STARTER}\n```python\n{r['starter_code']}\n```\n\n"
            else:
                fmt = f"### Format: {LCB_FMT_STDIN}\n```python\n# YOUR CODE HERE\n```\n\n"
            prompt = f"### Question:\n{r['question_content']}\n\n{fmt}### Answer: (use the provided format with backticks)\n\n"
            meta = json.loads(r["metadata"] or "{}")
            pub = json.loads(r["public_test_cases"])
            priv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(r["private_test_cases"]))))
            out.append({"id": r["question_id"], "prompt": prompt, "system": LCB_SYSTEM, "gold": None, "grade": "raw",
                        "meta": {"platform": r["platform"], "difficulty": r["difficulty"], "date": r["contest_date"]}})
            tests.append({"id": r["question_id"], "func_name": meta.get("func_name"), "tests": pub + priv})
        write(a.dst, "lcb_v6.jsonl", out); write(a.dst, "lcb_tests.jsonl", tests)


if __name__ == "__main__":
    main()
