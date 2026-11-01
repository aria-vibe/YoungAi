#!/usr/bin/env python3
"""dsh_scorecard.py — 鹈鹕类任务的行为记分卡(2026-10-24, 用户定): 把 tests/asst/dsh_compare.sh 跑出来的每个 run 目录数成一行, 和官方原件并排。

用法: python3 tests/asst/dsh_scorecard.py gguf/dsh/runs/official-pel1 gguf/dsh/runs/local-pel-s1 ...   (不给参数 = gguf/dsh/runs/ 下全部, 按名字排)
每行的栏(全是数出来的, 不打分):
  步数   = events.jsonl 里 type=tool_call 的行数                思考字符 = type=thinking 的 text 总长
  用时   = timing.txt(dsh_compare.sh 10-24 起落盘; 没有就写 —)    收尾    = 最后一条 type=final 的 text 非空(模型自己说"做完了")
  看图   = tool=read_image 的次数                               文件    = ws/ 下最大的 .html(或 .svg)字节数
  元素 / path / 数字 = 那份文件里 '<标签' 个数 / '<path' 个数 / 数字串个数(坐标多少)
为什么只数不评: 之前判鹈鹕靠截图说"简陋", 改了 ② ③ 之后变没变说不清; 这张表前后各 ≥3 趟(温 1 下单趟是抽样), 数动了才算动了。
它不管像不像、美不美 —— 那是最后看截图的事, 不是尺。
出错会怎样: 目录里没 events.jsonl 就跳过并打 ★; ws/ 没有 html 时文件栏写 0(模型一笔没写也是一种结果, 不能漏行)。
★思考字符栏只是下限★: dsh 的 events.jsonl 把每段 thinking / write 截在 ~8-9k 字(10-26 实撞, 据此判"设计推理太短"是错的)。
本地臂的真思考在 spark 服务 trace 里, 用 --trace 模式数:
  python3 tests/asst/dsh_scorecard.py --trace gguf/v41/night/trace/serve_<时间>.txt [--from 请求号]
  每个请求一行: 开始时刻 / finish / 生成 token / 思考字数 / 非空行与不重复行占比 / 200 字窗重复次数(滑步 50, 逐字复读才会 >0) /
  出现 ≥2 次的长行(>30 字)条数 / Hmm 与 wait 次数(犹豫的粗信号) / 工具名。还在生成的请求 finish 写"进行中"。
"""
import json, os, re, sys

TAG = re.compile(r"<([a-zA-Z][a-zA-Z0-9]*)")
NUM = re.compile(r"-?\d+\.?\d*")


def one(run):
    ev = os.path.join(run, "events.jsonl")
    if not os.path.isfile(ev):
        print(f"★{run}: 没有 events.jsonl, 跳过★", file=sys.stderr); return None
    steps = think = img = 0; final = ""
    for ln in open(ev, encoding="utf-8"):
        try: e = json.loads(ln)
        except json.JSONDecodeError: continue
        t = e.get("type")
        if t == "tool_call":
            steps += 1
            if e.get("tool") == "read_image": img += 1
        elif t == "thinking": think += len(e.get("text") or "")
        elif t == "final": final = (e.get("text") or "").strip()
    tp = os.path.join(run, "timing.txt")
    secs = open(tp).read().strip() if os.path.isfile(tp) else "—"
    best, size = None, 0
    ws = os.path.join(run, "ws")
    for dp, dn, fn in os.walk(ws):
        dn[:] = [d for d in dn if not d.startswith(".")]
        for f in fn:
            if f.endswith((".html", ".svg")):
                p = os.path.join(dp, f); s = os.path.getsize(p)
                if s > size: best, size = p, s
    el = paths = nums = 0
    if best:
        txt = open(best, encoding="utf-8", errors="replace").read()
        tags = [m.lower() for m in TAG.findall(txt)]
        el = len(tags); paths = tags.count("path"); nums = len(NUM.findall(txt))
    return dict(run=os.path.basename(run.rstrip("/")), steps=steps, think=think, secs=secs, done="有" if final else "无",
                img=img, bytes=size, el=el, paths=paths, nums=nums, file=os.path.basename(best) if best else "-")


REQ = re.compile(r"^===== request (\d+) \S+ (\S+) =====$", re.M)


def trace(path, first):
    t = open(path, encoding="utf-8", errors="replace").read()
    heads = list(REQ.finditer(t))
    print("| 请求 | 开始 | finish | 生成 tok | 思考字 | 行 | 不重复 | 200字窗重复 | 重复长行 | Hmm | wait | 工具 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, h in enumerate(heads):
        n = int(h.group(1))
        if n < first: continue
        body = t[h.end():heads[i + 1].start() if i + 1 < len(heads) else len(t)]
        end = body.find(f"===== end request {n} =====")
        if end >= 0: body = body[:end]
        g = re.search(r"^generated_tokens: (\d+)", body, re.M)
        fi = re.search(r"^finish: (\S+)", body, re.M)
        r = re.search(r"^reasoning:\n(.*?)(?=^tool_call\[0\]:|^content:|\Z)", body, re.S | re.M)
        rs = r.group(1) if r else ""
        lines = [l.strip() for l in rs.split("\n") if l.strip()]
        uniq = 100 * len(set(lines)) // max(1, len(lines))
        seen, rep = set(), 0
        for k in range(0, max(0, len(rs) - 200), 50):
            s = rs[k:k + 200]; rep += s in seen; seen.add(s)
        cnt = {}
        for l in lines:
            if len(l) > 30: cnt[l] = cnt.get(l, 0) + 1
        dup = sum(1 for v in cnt.values() if v > 1)
        tools = ",".join(re.findall(r"^name: (\S+)", body, re.M)[:3])
        print(f"| {n} | {h.group(2)} | {fi.group(1) if fi else '进行中'} | {g.group(1) if g else '-'} | {len(rs)} | {len(lines)} | {uniq}% | {rep} | {dup} | "
              f"{len(re.findall('Hmm', rs))} | {len(re.findall('wait', rs, re.I))} | {tools} |")
    return 0


def main():
    if sys.argv[1:2] == ["--trace"]:
        a = sys.argv[2:]
        return trace(a[0], int(a[a.index("--from") + 1]) if "--from" in a else 0)
    runs = sys.argv[1:]
    if not runs:
        base = "gguf/dsh/runs"
        runs = sorted(os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    rows = [r for r in (one(x) for x in runs) if r]
    print("| run | 步数 | 思考字符 | 用时 s | 收尾 | 看图 | 文件 B | 元素 | path | 数字 | 文件名 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['run']} | {r['steps']} | {r['think']} | {r['secs']} | {r['done']} | {r['img']} | {r['bytes']} | {r['el']} | {r['paths']} | {r['nums']} | {r['file']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
