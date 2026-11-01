#!/usr/bin/env python3
"""lcb_grade.py — LiveCodeBench 判分: 把 bench_suite.py 跑出的回答里最后一个 ```python 块拿出来, 对公开 + 私有用例逐个跑。

口径照 LiveCodeBench 官方 testing_util: stdin 题 = 喂 input 比 stdout(按空白切成 token 逐个比, 数字容差 1e-6);
leetcode 函数题 = 调 Solution().<func>(*args), 参数是 input 按行 json.loads, 结果 json 比(元组当列表)。
每个用例 6 s 超时(unsloth 口径), 任一用例失败即该答案判错。模型代码跑在子进程里, 当前目录是空的临时目录。

用法: python3 lcb_grade.py --out gguf/v41/bench/<日期>/lcb --tests gguf-tools/bench/data/lcb_tests.jsonl
产物: <out>/graded.jsonl(每答案一行: id/sample/passed/fail_reason/n_tests) + 更新 <out>/summary.json 的 pass_at_1。
"""
import argparse, json, os, re, subprocess, sys, tempfile

RUNNER = r'''
import json, sys, importlib.util
spec = importlib.util.spec_from_file_location("sol", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
args = [json.loads(l) for l in sys.stdin.read().split("\n") if l.strip()]
print(json.dumps(getattr(m.Solution(), sys.argv[2])(*args)))
'''


def extract_code(content):
    blocks = re.findall(r"```(?:python|py|Python)?\s*\n(.*?)```", content or "", re.S)
    return blocks[-1] if blocks else None


def tokens_equal(a, b):
    ta, tb = a.split(), b.split()
    if ta == tb:
        return True
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        try:
            if abs(float(x) - float(y)) <= 1e-6 * max(1.0, abs(float(y))):
                continue
        except ValueError:
            pass
        return False
    return True


def json_equal(a, b):
    def norm(v):
        if isinstance(v, (list, tuple)):
            return [norm(x) for x in v]
        if isinstance(v, float) and v == int(v):
            return int(v)
        return v
    return norm(a) == norm(b)


def run_tests(code, tests, func_name, timeout):
    with tempfile.TemporaryDirectory() as d:
        sol = os.path.join(d, "sol.py"); open(sol, "w").write(code)
        run = os.path.join(d, "run.py"); open(run, "w").write(RUNNER)
        for i, t in enumerate(tests):
            try:
                if t.get("testtype") == "functional":
                    p = subprocess.run([sys.executable, "-I", run, sol, func_name], input=t["input"], capture_output=True,
                                       text=True, timeout=timeout, cwd=d)
                    if p.returncode != 0:
                        return False, f"test {i}: runtime error: {p.stderr.strip()[-200:]}"
                    try:
                        ok = json_equal(json.loads(p.stdout.strip()), json.loads(t["output"]))
                    except json.JSONDecodeError:
                        ok = tokens_equal(p.stdout, t["output"])
                else:
                    p = subprocess.run([sys.executable, "-I", sol], input=t["input"], capture_output=True, text=True,
                                       timeout=timeout, cwd=d)
                    if p.returncode != 0:
                        return False, f"test {i}: runtime error: {p.stderr.strip()[-200:]}"
                    ok = tokens_equal(p.stdout, t["output"])
                if not ok:
                    return False, f"test {i}: wrong answer"
            except subprocess.TimeoutExpired:
                return False, f"test {i}: timeout"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tests", required=True)
    ap.add_argument("--timeout", type=float, default=6.0)
    args = ap.parse_args()
    tests = {json.loads(l)["id"]: json.loads(l) for l in open(args.tests) if l.strip()}
    recs = [json.loads(l) for l in open(os.path.join(args.out, "answers.jsonl")) if l.strip()]
    graded, npass = [], 0
    for r in recs:
        raw = json.load(open(os.path.join(args.out, "raw", f"{r['id']}_{r['sample']}.json")))
        content = raw["response"]["choices"][0]["message"].get("content") or ""
        code = extract_code(content)
        t = tests[r["id"]]
        if code is None:
            ok, why = False, "no code block"
        else:
            ok, why = run_tests(code, t["tests"], t.get("func_name"), args.timeout)
        npass += ok
        graded.append({"id": r["id"], "sample": r["sample"], "passed": ok, "fail_reason": why, "n_tests": len(t["tests"]),
                       "truncated": r["truncated"]})
        print(f"{r['id']} s{r['sample']}: {'过' if ok else '挂'} {why}", flush=True)
    with open(os.path.join(args.out, "graded.jsonl"), "w") as f:
        for g in graded:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    sp = os.path.join(args.out, "summary.json")
    s = json.load(open(sp)) if os.path.exists(sp) else {}
    n = len(graded); acc = npass / n if n else None
    s.update({"graded": n, "pass_at_1": None if acc is None else round(100 * acc, 2),
              "se_pct": None if acc is None else round(100 * (acc * (1 - acc) / n) ** 0.5, 2), "grader": "lcb_grade.py"})
    json.dump(s, open(sp, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: s.get(k) for k in ("answers", "graded", "pass_at_1", "se_pct", "truncated_pct")}), flush=True)


if __name__ == "__main__":
    main()
