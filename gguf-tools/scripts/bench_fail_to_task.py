#!/usr/bin/env python3
"""bench_fail_to_task.py — 跑分里答错的题 → ① 排除清单(以后跑分跳过, 进了训练料的题不能再当尺) ② 段 B --task 的料(真实请求 + 我们当时的失败轨迹当学生)。

为什么(2026-10-31 用户): "把今天跑的 9 指标都不好的地方总结出来跑后训练 b 段, 也标记记录, 后面再跑 9 指标的时候要排除"。
料的形状与 t001(aider go/wordy)同: 请求逐字节 = 跑分时发给服务端的那条(bench_suite.py 的 raw/<id>_<s>.json 存了 prompt / system;
aider 的用 api_record_proxy.py 录的 messages), 学生 = 当时那条失败的回答(reasoning_content + content), 官方从学生思考的切点接写。

用法(Mac, 仓库根):
  exclude --bench gguf/v41/bench/1031 --aider <aider 结果目录> --round t002 [--out gguf-tools/bench/data/trained_exclude.jsonl]
      六项的 answers.jsonl / graded.jsonl 里 correct/passed 为假的 + aider 目录里 tests_outcomes 末位为假的 → 追加到排除清单(task id why round date; 已有的不重复)
  task --bench gguf/v41/bench/1031 --records <录的 jsonl 目录> --exclude <清单> --round t002 --out <工作目录>
      按清单里 round == 本轮的题出 <out>/task_q.jsonl(id q messages src) 与 <out>/student.jsonl(think_freegen_probe 同格式, 轮脚本见已有就不再生成)
      aider 的题: <records>/<题名>.jsonl 里消息数最多的那条请求(主请求; 其余是 aider 的摘要调用)
排除清单只收★留下来的轮★(10-31 晚纠正): 练过但那轮回退了的题, 现役侧车根本没学过, 排除它们只会让下次跑分白少题。
  所以 exclude/task 写的是 trained_pending.jsonl(候选); 轮脚本 ⑤ 判"留"之后, 把该轮的行从 pending 挪进 trained_exclude.jsonl(bench_suite.py / aider_bench_mac.sh 读的是后者)。
  t002/t003 的 18 道 10-31 全部回退过, 已从 exclude 挪回 pending。
出错会怎样: 清单里的题在结果目录找不到原始返回就报出来跳过, 不猜。"""
import argparse, datetime, glob, json, os

TASKS = ("aime25", "hmmt_feb26", "gpqa_diamond", "mmlu_pro", "lcb_v6", "ifeval")


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def cmd_exclude(a):
    have = {(r["task"], r["id"]) for r in read_jsonl(a.out)}; rows = []; today = datetime.date.today().isoformat()
    for t in TASKS:
        d = os.path.join(a.bench, t)
        if t in ("lcb_v6", "ifeval"):
            for r in read_jsonl(os.path.join(d, "graded.jsonl")):
                if not r["passed"]: rows.append({"task": t, "id": r["id"], "why": r.get("fail_reason") or ("撞上限" if r["truncated"] else "违指令")})
        else:
            for r in read_jsonl(os.path.join(d, "answers.jsonl")):
                if not r["correct"]: rows.append({"task": t, "id": r["id"], "why": f"答 {r['pred']} 金 {r['gold']}" + (" 撞上限" if r["truncated"] else "")})
    if a.aider:
        for p in sorted(glob.glob(os.path.join(a.aider, "*/exercises/practice/*/.aider.results.json"))):
            r = json.load(open(p))
            if not r["tests_outcomes"][-1]: rows.append({"task": "aider", "id": "/".join(p.split("/")[-5:-1]), "why": "单测挂"})
    n = 0
    with open(a.out, "a", encoding="utf-8") as f:
        for r in rows:
            if (r["task"], r["id"]) in have: continue
            f.write(json.dumps({**r, "round": a.round, "date": today}, ensure_ascii=False) + "\n"); have.add((r["task"], r["id"])); n += 1
    print(f"exclude: 错题 {len(rows)} 道, 新记 {n} 道 → {a.out}(共 {len(have)} 道)")


def cmd_task(a):
    picks = [r for r in read_jsonl(a.exclude) if r["round"] == a.round]
    os.makedirs(a.out, exist_ok=True); nq = ns = 0
    with open(os.path.join(a.out, "task_q.jsonl"), "w", encoding="utf-8") as fq, open(os.path.join(a.out, "student.jsonl"), "w", encoding="utf-8") as fs:
        for r in picks:
            t, i = r["task"], r["id"]; tid = f"{t}_{i.replace('/', '_')}"
            if t == "aider":
                recs = read_jsonl(os.path.join(a.records, os.path.basename(i) + ".jsonl"))
                recs = [x for x in recs if isinstance(x.get("request"), dict)]
                if not recs: print(f"  ★{t} {i}: 没有录到的请求, 跳过"); continue
                rec = max(recs, key=lambda x: len(x["request"]["messages"]))   # 主请求 = 消息最多的那条(摘要调用只有 2 条)
                msgs = rec["request"]["messages"]; resp = rec["response"]
            else:
                p = os.path.join(a.bench, t, "raw", f"{i}_0.json")
                if not os.path.exists(p): print(f"  ★{t} {i}: 没有 {p}, 跳过"); continue
                raw = json.load(open(p, encoding="utf-8")); req = raw["request"]; resp = raw["response"]
                msgs = ([{"role": "system", "content": req["system"]}] if req.get("system") else []) + [{"role": "user", "content": req["prompt"]}]
            q = [m for m in msgs if m["role"] == "user"][-1]["content"]
            fq.write(json.dumps({"id": tid, "q": q, "messages": msgs, "src": t, "why": r["why"]}, ensure_ascii=False) + "\n"); nq += 1
            ch = resp["choices"][0]; m = ch["message"]; rs = m.get("reasoning_content") or ""; cs = m.get("content") or ""; fin = ch.get("finish_reason")
            if not rs: print(f"  ★{t} {i}: 回答里没有思考段, 学生条不出(轮脚本会让现役重做这题)"); continue
            fs.write(json.dumps({"id": tid, "q": q, "messages": msgs, "reasoning": rs, "content": cs, "finish": fin, "closed": bool(cs) and fin == "stop",
                                 "usage": resp.get("usage"), "src": t}, ensure_ascii=False) + "\n"); ns += 1
    print(f"task: 本轮 {a.round} 题 {len(picks)} 道 → task_q {nq} 行, student {ns} 行 → {a.out}")


def cmd_grade_official(a):
    """官方作答(api_raw.jsonl)按各基准的判分器判一遍: 段 B 的靶是官方分布, 对错不影响训练, 但要记下来——官方也错的题修不回"对"。
    math / mc 用 bench_suite.grade, LCB 用 lcb_grade 跑用例(要 --tests), IFEval 用官方 strict 判分(要 hfenv: nltk / langdetect / absl)。"""
    import sys
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
    import bench_suite, lcb_grade
    data = {}
    for t in TASKS:
        for r in read_jsonl(os.path.join(a.data_dir, t + ".jsonl")): data[f"{t}_{r['id']}"] = (t, r)
    tests = {r["id"]: r for r in read_jsonl(a.tests)} if a.tests else {}
    for rec in read_jsonl(a.raw):
        tid = rec["id"]
        if tid not in data: print(f"  {tid}: 不是六项的题(aider 的靠单测另判)"); continue
        t, row = data[tid]; content = rec.get("content") or ""
        if t == "lcb_v6":
            code = lcb_grade.extract_code(content); tt = tests.get(row["id"])
            ok, why = (False, "no code block") if code is None else (False, "没有 --tests") if not tt else lcb_grade.run_tests(code, tt["tests"], tt.get("func_name"), 6.0)
            print(f"  {tid}: 官方 {'过' if ok else '挂 ' + why}")
        elif t == "ifeval":
            sys.path.insert(0, os.path.join(here, "..", "bench", "ifeval_official"))
            from instruction_following_eval import evaluation_lib
            m = row["meta"]; inp = evaluation_lib.InputExample(key=m["key"], instruction_id_list=m["instruction_id_list"], prompt=m["prompt"],
                                                              kwargs=[{k: v for k, v in kw.items() if v is not None} for kw in m["kwargs"]])
            out = evaluation_lib.test_instruction_following_strict(inp, {m["prompt"]: content})
            print(f"  {tid}: 官方 {'过' if out.follow_all_instructions else '挂 ' + str(out.follow_instruction_list)}")
        else:
            pred, how, ok = bench_suite.grade(row, content)
            print(f"  {tid}: 官方 {'对' if ok else '错'} 答 {str(pred)[:30]} 金 {str(row['gold'])[:30]} ({how}, finish {rec.get('finish')}, {(rec.get('usage') or {}).get('completion_tokens')} tok)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("grade-official"); g.add_argument("--raw", required=True); g.add_argument("--data-dir", default="gguf-tools/bench/data"); g.add_argument("--tests", default="gguf-tools/bench/data/lcb_tests.jsonl")
    g.set_defaults(f=cmd_grade_official)
    e = sp.add_parser("exclude"); e.add_argument("--bench", required=True); e.add_argument("--aider", default=""); e.add_argument("--round", required=True)
    e.add_argument("--out", default="gguf-tools/bench/data/trained_pending.jsonl"); e.set_defaults(f=cmd_exclude)
    t = sp.add_parser("task"); t.add_argument("--bench", required=True); t.add_argument("--records", default=""); t.add_argument("--exclude", default="gguf-tools/bench/data/trained_pending.jsonl")
    t.add_argument("--round", required=True); t.add_argument("--out", required=True); t.set_defaults(f=cmd_task)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
