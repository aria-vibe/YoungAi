#!/usr/bin/env python3
"""think_trace_collect.py — ③ 的 B 份料: 官方 API 采 DeepSeek-V4.1-Flash 的思考轨迹, 只留尾段(2026-10-22, docs/think_close.md §2)。

为什么要它: 量化把"该停就停"压掉(r316 的 </think> 全段 Σp FP 0.82 对量化 0.28); 要修它, 料里得有 </think> 这个决策位,
而且教师分布得是原件自己的 —— 官方 API 跑的就是出厂原件, reasoning_content 是它真实的思考。本地量化模型自吐的轨迹不许当料(09-30 铁律)。
为什么只留尾段: 一条思考几千到几万 token, 训练器 maxlen 1024; 决策位在末尾, 取"思考末段 + </think> + 正文"≤1000 token 把它包住。
Python 在这里是外部 API 采集(全仓零 Python 裁决的例外); 数值一个都不碰, 分词只用来量长度切尾, 训练器自己再分一遍。

三步(都在 spark 仓库根跑, 分词器走 ~/v41env):
  questions  --humaneval gguf-tools/bench/data/HumanEval.jsonl --n-code 100 --exclude HumanEval/129 \
             --general gguf-tools/data/posttrain/think_q_general.txt --out gguf-tools/data/posttrain/think_q.jsonl
  collect    --key gguf-tools/data/posttrain/deepseek_api.key --questions …/think_q.jsonl --out …/think_raw.jsonl [--jobs 4] [--limit N]
             续跑安全: 已在 out 里的 id 跳过; 一行一条 {"id","src","q","reasoning","content","finish","usage","ok"}
  tail       --hf hf/DeepSeek-V4.1-Flash-1017 --raw …/think_raw.jsonl --out …/think_b.jsonl [--max-tok 1000] [--min-tail 96]
             渲染照官方 encoding.py(思考档, effort high=75): <｜System｜>Reasoning Effort: 75 (…)\n\n<｜User｜>题<｜Assistant｜><think>尾段</think>正文
             ★不写 EOS 字面★: 训练器 pt_add_text 自己补 EOS(core_ptrain_data.c), 写了就是两个 EOS。
  mid        --hf … --raw …/think_raw.jsonl --out …/think_mid.jsonl [--max-tok 1000] [--min-mid 96] [--min-rest 192] [--frac 0.7]
             思考【中段】负样本(10-22 晚): 同样的头 + <think> + 思考开头一段, 不带 </think> 不带正文 —— 正式趟只喂尾段, ③ 学成"思考一段就该收",
             r316 收口 Σp 冲到 FP 的 2 倍; 教师在中段给 </think> 只有 ~8e-4, 这就是"这里别收"的靶。前缀离真结尾 ≥ min-rest 且 ≤ frac×整条, 否则又是"快收口"样本。
             训练器补的那个 EOS 只进统计(决策位/命中), 不当任何位置的输入, 蒸馏靶不受影响; ptrain 的"决策位 KL"均值会混进这些位(教师 p≈0), 读数按 ptfstat 的"中段"类拆。
出错会怎样: collect 里 finish≠stop(撞 max_tokens 没收口)的条 ok=false, tail 跳过; tail 里 题+正文 太长(尾段不足 min-tail)的条丢掉并计数;
API 429/5xx 退避重试 4 次再放弃(该条不写, 下次续跑再来)。key 文件在 .gitignore(*.key), 日志里永不打印 key。
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "https://api.deepseek.com/chat/completions"
EFFORT_PREFIX = "Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
EFFORT_BUDGET = {"low": 50, "high": 75, "max": 100}   # 官方 encoding.py REASONING_EFFORT_MAPPINGS; 引擎 core_globals.c 三个常量逐字同
SYS, USER, ASST, THINK, THINK_END, EOS = "<｜System｜>", "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>", "<｜end▁of▁sentence｜>"


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


# ---------- questions ----------
def cmd_questions(a):
    qs = []
    if a.humaneval:
        he = read_jsonl(a.humaneval)
        ex = set(a.exclude or [])
        he = [h for h in he if h["task_id"] not in ex]
        step = max(1, len(he) // a.n_code)
        for h in he[::step][: a.n_code]:   # 等距取, 不挑内容
            q = ("Complete the following Python function. Return the full implementation in one code block.\n\n"
                 "```python\n" + h["prompt"].rstrip() + "\n```")
            qs.append({"id": h["task_id"].replace("/", "_"), "src": "humaneval", "q": q})
    if a.general:
        n = 0
        for ln in open(a.general, encoding="utf-8"):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            qs.append({"id": f"gen_{n:03d}", "src": "general", "q": ln}); n += 1
    with open(a.out, "w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"questions: {len(qs)} 题 → {a.out} (" + ", ".join(f"{s} {sum(1 for q in qs if q['src']==s)}" for s in sorted({q['src'] for q in qs})) + ")")


# ---------- collect ----------
def api_stream(key, model, q, max_tokens, timeout):
    """一条题流式拉完; 返回 (reasoning, content, finish, usage)。流式是为了长思考不被网关超时掐断。"""
    body = {"model": model, "messages": [{"role": "user", "content": q}], "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(API_URL, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    reasoning, content, finish, usage = [], [], None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data:"):
                continue
            ln = ln[5:].strip()
            if ln == "[DONE]":
                break
            ev = json.loads(ln)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
                if d.get("content"):
                    content.append(d["content"])
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return "".join(reasoning), "".join(content), finish, usage


def collect_one(key, model, item, max_tokens, timeout):
    delay = 5
    for attempt in range(5):
        try:
            t0 = time.time()
            reasoning, content, finish, usage = api_stream(key, model, item["q"], max_tokens, timeout)
            ok = finish == "stop" and bool(reasoning) and bool(content)
            return dict(item, reasoning=reasoning, content=content, finish=finish, usage=usage, ok=ok,
                        model=model, effort="high", secs=round(time.time() - t0, 1))
        except urllib.error.HTTPError as e:
            msg = f"HTTP {e.code}"
            if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                return dict(item, ok=False, finish=f"error:{msg}", reasoning="", content="", usage=None)
        except Exception as e:   # 网络抖动 / 超时
            msg = type(e).__name__
            if attempt == 4:
                return dict(item, ok=False, finish=f"error:{msg}", reasoning="", content="", usage=None)
        print(f"  [{item['id']}] {msg}, {delay}s 后重试", flush=True)
        time.sleep(delay); delay *= 3


def cmd_collect(a):
    key = open(a.key, encoding="utf-8").read().strip()
    done = {r["id"] for r in read_jsonl(a.out) if r.get("ok")}   # 失败的下次重来
    todo = [q for q in read_jsonl(a.questions) if q["id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"collect: 已有 {len(done)} 条 ok, 本次 {len(todo)} 题, {a.jobs} 路并发, max_tokens {a.max_tokens}", flush=True)
    lock = threading.Lock()
    nok = nbad = 0
    t0 = time.time()
    with open(a.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(collect_one, key, a.model, q, a.max_tokens, a.timeout): q for q in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            u = r.get("usage") or {}
            rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", "?")
            if r["ok"]:
                nok += 1
            else:
                nbad += 1
            print(f"  {i}/{len(todo)} [{r['id']}] {'ok' if r['ok'] else '★'+str(r['finish'])} 思考 {rt} tok, 正文 {len(r.get('content',''))} 字, "
                  f"{r.get('secs','?')} s, 累计 {time.time()-t0:.0f} s", flush=True)
    print(f"collect 完成: ok {nok} / 失败 {nbad} → {a.out}", flush=True)


# ---------- tail ----------
def split_units(text):
    """先按空行切段, 段太长再按句切; 保留分隔符, 拼回去逐字同原文。"""
    parts, cur = [], ""
    for ch in text:
        cur += ch
        if cur.endswith("\n\n"):
            parts.append(cur); cur = ""
    if cur:
        parts.append(cur)
    return parts


def split_sentences(par):
    out, cur = [], ""
    for ch in par:
        cur += ch
        if ch in "。！？!?\n" or (ch == "." and len(cur) > 1):
            out.append(cur); cur = ""
    if cur:
        out.append(cur)
    return out


def cmd_tail(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.hf, "tokenizer.json"))
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    think_end_id = enc(THINK_END)
    assert len(think_end_id) == 1, f"</think> 没有单独的 id: {think_end_id}"
    think_end_id = think_end_id[0]
    prefix = SYS + EFFORT_PREFIX.format(budget=EFFORT_BUDGET[a.effort])
    rows = [r for r in read_jsonl(a.raw) if r.get("ok")]
    kept, dropped, man = 0, {}, []
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            head = prefix + USER + r["q"] + ASST + THINK
            suffix = THINK_END + r["content"]
            n_head, n_suf = len(enc(head)), len(enc(suffix))
            budget = a.max_tok - 2 - n_head - n_suf   # BOS + 训练器补的 EOS 各 1
            if budget < a.min_tail:
                dropped["题+正文太长"] = dropped.get("题+正文太长", 0) + 1
                man.append((r["id"], r["src"], "drop", n_head, n_suf, 0, 0)); continue
            units = split_units(r["reasoning"])
            # 从末尾往前攒整段; 最后一段单独放不下就按句子攒
            tail = ""
            for u in reversed(units):
                cand = u + tail
                if len(enc(cand)) <= budget:
                    tail = cand
                else:
                    if not tail:
                        for s in reversed(split_sentences(u)):
                            c2 = s + tail
                            if len(enc(c2)) <= budget:
                                tail = c2
                            else:
                                break
                    break
            text = head + tail + suffix
            ids = enc(text)
            while len(ids) + 2 > a.max_tok and tail:   # BPE 拼接边界可能多出一两个 token: 从头掐一句
                ss = split_sentences(tail)
                tail = "".join(ss[1:]) if len(ss) > 1 else ""
                text = head + tail + suffix; ids = enc(text)
            if tail != r["reasoning"] and len(enc(tail)) < a.min_tail:   # 整条思考本来就短(几十 token 就收口)是完整样本, 只丢"截过还剩不到 min_tail"的
                dropped["尾段不足"] = dropped.get("尾段不足", 0) + 1
                man.append((r["id"], r["src"], "drop", n_head, n_suf, len(enc(tail)), 0)); continue
            assert ids.count(think_end_id) == 1 and EOS not in text, r["id"]
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n"); kept += 1
            man.append((r["id"], r["src"], "keep", n_head, n_suf, len(enc(tail)), len(ids) + 2))
    with open(a.out + ".manifest.txt", "w", encoding="utf-8") as f:
        f.write("# id src keep/drop 头(含题)tok 尾(</think>+正文)tok 思考尾段tok 整条tok(含BOS/EOS)\n")
        for m in man:
            f.write(" ".join(str(x) for x in m) + "\n")
    print(f"tail: 收 {kept} 条 → {a.out}; 丢 {dropped or 0}; 清单 {a.out}.manifest.txt")
    if kept:
        tot = [m[6] for m in man if m[2] == "keep"]
        print(f"  整条 token: 均值 {sum(tot)/len(tot):.0f}, 最长 {max(tot)}, 最短 {min(tot)}; 思考尾段均值 {sum(m[5] for m in man if m[2]=='keep')/kept:.0f}")


def cmd_mid(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.hf, "tokenizer.json"))
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    prefix = SYS + EFFORT_PREFIX.format(budget=EFFORT_BUDGET[a.effort])
    rows = [r for r in read_jsonl(a.raw) if r.get("ok")]
    kept, dropped, man = 0, {}, []
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            head = prefix + USER + r["q"] + ASST + THINK
            n_head, n_reas = len(enc(head)), len(enc(r["reasoning"]))
            # 前缀上限: 放得进 maxlen(BOS + 训练器补的 EOS 各 1) / 离真结尾够远 / 不超过整条的 frac —— 三者取小
            allowed = min(a.max_tok - 2 - n_head, n_reas - a.min_rest, int(a.frac * n_reas))
            if allowed < a.min_mid:
                dropped["思考太短"] = dropped.get("思考太短", 0) + 1
                man.append((r["id"], r["src"], "drop", n_head, n_reas, 0, 0)); continue
            units = split_units(r["reasoning"])
            mid = ""
            for u in units:            # 从开头往后攒整段; 第一段单独放不下就按句子攒
                cand = mid + u
                if len(enc(cand)) <= allowed:
                    mid = cand
                else:
                    if not mid:
                        for s in split_sentences(u):
                            c2 = mid + s
                            if len(enc(c2)) <= allowed:
                                mid = c2
                            else:
                                break
                    break
            text = head + mid
            ids = enc(text)
            while len(ids) + 2 > a.max_tok and mid:   # BPE 拼接边界可能多出一两个 token: 从尾掐一句
                ss = split_sentences(mid)
                mid = "".join(ss[:-1]) if len(ss) > 1 else ""
                text = head + mid; ids = enc(text)
            n_mid = len(enc(mid))
            if n_mid < a.min_mid:
                dropped["前缀不足"] = dropped.get("前缀不足", 0) + 1
                man.append((r["id"], r["src"], "drop", n_head, n_reas, n_mid, 0)); continue
            assert THINK_END not in text and EOS not in text, r["id"]
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n"); kept += 1
            man.append((r["id"], r["src"], "keep", n_head, n_reas, n_mid, len(ids) + 2))
    with open(a.out + ".manifest.txt", "w", encoding="utf-8") as f:
        f.write("# id src keep/drop 头(含题)tok 整条思考tok 取的前缀tok 整条tok(含BOS/EOS)\n")
        for m in man:
            f.write(" ".join(str(x) for x in m) + "\n")
    print(f"mid: 收 {kept} 条 → {a.out}; 丢 {dropped or 0}; 清单 {a.out}.manifest.txt")
    if kept:
        ks = [m for m in man if m[2] == "keep"]
        print(f"  整条 token: 均值 {sum(m[6] for m in ks)/kept:.0f}, 最长 {max(m[6] for m in ks)}, 最短 {min(m[6] for m in ks)}; "
              f"前缀均值 {sum(m[5] for m in ks)/kept:.0f}, 前缀占整条思考的均值 {sum(m[5]/m[4] for m in ks)/kept:.2f}, 余下思考均值 {sum(m[4]-m[5] for m in ks)/kept:.0f} tok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("questions"); q.add_argument("--humaneval"); q.add_argument("--n-code", type=int, default=100)
    q.add_argument("--exclude", nargs="*"); q.add_argument("--general"); q.add_argument("--out", required=True)
    c = sp.add_parser("collect"); c.add_argument("--key", required=True); c.add_argument("--questions", required=True); c.add_argument("--out", required=True)
    c.add_argument("--model", default="deepseek-flash"); c.add_argument("--max-tokens", type=int, default=32768)
    c.add_argument("--jobs", type=int, default=4); c.add_argument("--limit", type=int, default=0); c.add_argument("--timeout", type=int, default=1800)
    t = sp.add_parser("tail"); t.add_argument("--hf", required=True); t.add_argument("--raw", required=True); t.add_argument("--out", required=True)
    t.add_argument("--max-tok", type=int, default=1000); t.add_argument("--min-tail", type=int, default=96); t.add_argument("--effort", default="high", choices=list(EFFORT_BUDGET))
    m = sp.add_parser("mid"); m.add_argument("--hf", required=True); m.add_argument("--raw", required=True); m.add_argument("--out", required=True)
    m.add_argument("--max-tok", type=int, default=1000); m.add_argument("--min-mid", type=int, default=96); m.add_argument("--min-rest", type=int, default=192)
    m.add_argument("--frac", type=float, default=0.7); m.add_argument("--effort", default="high", choices=list(EFFORT_BUDGET))
    a = ap.parse_args()
    {"questions": cmd_questions, "collect": cmd_collect, "tail": cmd_tail, "mid": cmd_mid}[a.cmd](a)


if __name__ == "__main__":
    main()
