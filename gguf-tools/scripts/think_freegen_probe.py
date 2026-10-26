#!/usr/bin/env python3
"""think_freegen_probe.py — 自由生成尺(2026-10-26): 模型自己从头想, 能不能走到收口、犹豫有多密。

为什么要它: 训练器的"决策位 KL"和收口探针都是把官方轨迹喂进去(teacher-forced), 量"到了官方收口的那一位给 </think> 多大概率";
模型自己生成时走的是自己的路, 走不到那一位就量不到(10-26 full2: 决策位 KL 减半, 鹈鹕 49k token 没收口)。这把尺直接让部署中的服务
对着【训练没见过】的官方图形题生成, 和同题官方轨迹并排:
  收口了没 / 到收口位多少 token(官方多少) / Hmm 与 wait 每千字(官方多少) / 正文字节(官方多少)。
带上限的趟只能说"到上限没停", 不能说"不停"(fable5 09-22 铁律), 表里原样标。

用法(spark, 模型服务开着): ~/v41env/bin/python gguf-tools/scripts/think_freegen_probe.py --raw gguf-tools/data/posttrain/graphics_raw.jsonl \
      --exclude gguf-tools/data/posttrain/restore_clean/api_ok27.jsonl --n 3 --out gguf/go-onebit/restore/freegen_<标签>.jsonl [--max-tokens 32768] [--server http://127.0.0.1:8001]
  --exclude 里的题(text 行 <｜User｜>…<｜Assistant｜> 之间)不选; 其余按 id 排序取前 n 道收口了的官方题, 每次同一批, 前后趟可比。
  服务走 OpenAI /v1/chat/completions 流式, reasoning_effort=high 开思考(server_config.c), 采样参数不传 = 服务默认(模型卡配方)。
出错会怎样: 服务没起 → 连接错直接退; 流里既没有 reasoning_content 也没有 <think> → 这条记 "没有思考段", 不猜。"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from think_trace_collect import USER, ASST  # noqa: E402


def density(text, pat):
    return 1000.0 * len(re.findall(pat, text)) / max(1, len(text))


def gen(server, q, max_tokens, timeout):
    body = {"model": "deepseek-flash", "messages": [{"role": "user", "content": q}], "max_tokens": max_tokens, "stream": True,
            "reasoning_effort": "high", "stream_options": {"include_usage": True}}
    req = urllib.request.Request(server.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json"})
    reasoning, content, finish, usage, t0, t_first = [], [], None, None, time.time(), None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data:") or ln[5:].strip() == "[DONE]":
                continue
            ev = json.loads(ln[5:]); usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if t_first is None and (d.get("reasoning_content") or d.get("reasoning") or d.get("content")): t_first = time.time()
                if d.get("reasoning_content") or d.get("reasoning"): reasoning.append(d.get("reasoning_content") or d.get("reasoning"))
                if d.get("content"): content.append(d["content"])
                finish = ch.get("finish_reason") or finish
    rs, cs = "".join(reasoning), "".join(content)
    if not rs and "<think>" in cs:   # 服务把思考混在 content 里的兜不住就按标签拆; 两样都没有就记没有思考段
        a, b = cs.find("<think>"), cs.find("</think>")
        rs, cs = (cs[a + 7:b] if b > 0 else cs[a + 7:]), (cs[b + 8:] if b > 0 else "")
    return rs, cs, finish, usage, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True); ap.add_argument("--exclude"); ap.add_argument("--n", type=int, default=3); ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=32768); ap.add_argument("--server", default="http://127.0.0.1:8001"); ap.add_argument("--timeout", type=int, default=7200)
    a = ap.parse_args()
    ex = set()
    if a.exclude:
        for ln in open(a.exclude, encoding="utf-8"):
            t = json.loads(ln).get("text", ""); u, s = t.find(USER), t.find(ASST)
            if 0 <= u < s: ex.add(t[u + len(USER):s])
    offi = sorted((json.loads(ln) for ln in open(a.raw, encoding="utf-8") if ln.strip()), key=lambda r: r["id"])
    picks = [r for r in offi if r.get("ok") and r["q"] not in ex][: a.n]
    print(f"freegen: 官方图形题 {len(offi)} 条, 排除训练用 {len(ex)} 题, 取 {len(picks)} 道; 服务 {a.server}; 上限 {a.max_tokens}", flush=True)
    print("| 题 | 收口 | 思考 tok(官方) | Hmm/千字(官方) | wait/千字(官方) | 正文字节(官方) | 用时 s |")
    print("|---|---|---|---|---|---|---|")
    with open(a.out, "a", encoding="utf-8") as f:
        for r in picks:
            rs, cs, finish, usage, secs = gen(a.server, r["q"], a.max_tokens, a.timeout)
            u = usage or {}; rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or u.get("completion_tokens")
            ou = r.get("usage") or {}; ort = (ou.get("completion_tokens_details") or {}).get("reasoning_tokens")
            closed = bool(cs) and finish == "stop"
            state = "收口" if closed else ("有正文但 finish=" + str(finish) if cs else ("到上限没停" if finish == "length" else "没收口 finish=" + str(finish)))
            if not rs: state += "(没有思考段)"
            rec = dict(id=r["id"], q=r["q"], reasoning=rs, content=cs, finish=finish, usage=usage, secs=round(secs, 1), closed=closed, max_tokens=a.max_tokens,
                       official=dict(reasoning_tokens=ort, hmm=round(density(r["reasoning"], "Hmm"), 2), wait=round(density(r["reasoning"], "(?i)wait"), 2), body=len(r.get("content") or "")))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(f"| {r['id']} | {state} | {rt}({ort}) | {density(rs, 'Hmm'):.2f}({rec['official']['hmm']}) | {density(rs, '(?i)wait'):.2f}({rec['official']['wait']}) | "
                  f"{len(cs)}({rec['official']['body']}) | {secs:.0f} |", flush=True)


if __name__ == "__main__":
    main()
