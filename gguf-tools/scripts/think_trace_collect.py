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
  chunks  --hf … --raw …/graphics_raw.jsonl --out …/api_chunks4096.jsonl [--max-tok 4096] [--max-chunks 6]
             整条轨迹(思考 + </think> + 正文)切成连续 ≤max_tok 的窗, 每窗带同一个头, 当 one-hot 硬靶(10-25 晚: FP 教师失灵后段 B 不再向它蒸, 见 cmd_chunks)。
  ids     --hf … --raw …/heldout_raw.jsonl --out-dir gguf/go-onebit/code/<留出集>   官方轨迹 → 判决尺 ids + list.txt(泛化留出尺, 见 cmd_ids)
出错会怎样: collect 里 finish≠stop(撞 max_tokens 没收口)的条 ok=false, tail 跳过; tail 里 题+正文 太长(尾段不足 min-tail)的条丢掉并计数;
API 429/5xx 退避重试 4 次再放弃(该条不写, 下次续跑再来)。key 文件在 .gitignore(*.key), 日志里永不打印 key。
"""
import argparse
import json
import os
import re
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
    for spec in a.set or []:   # --set 族名=一行一题的文件(10-26 晚: 6 族等份配比的题集 mix_q_*.txt), id = 族名_NNN
        name, path = spec.split("=", 1); n = 0
        for ln in open(path, encoding="utf-8"):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            qs.append({"id": f"{name}_{n:03d}", "src": name, "q": ln.replace("\\n", "\n")}); n += 1
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


# ---------- chunks(10-25 晚: 段 B 改硬靶, 整条轨迹切窗) ----------
def cmd_chunks(a):
    """整条 API 轨迹(思考 + </think> + 正文)按 ≤max_tok 切成连续的窗, 每窗都带同一个头(系统 effort + 题 + <｜Assistant｜><think>), 落 {"text"}。
    为什么: FP 教师长程失灵(fable5 10-25), 段 B 不能再向它蒸; 官方 API 的轨迹本身就是原件吐的, 一字不改当 one-hot 靶(训练器 text 行不带
    teacher= 就是交叉熵)。鹈鹕"path 少一半"是一两万位设计推理的累积, 所以整条都要, 不止前 4096 位(mid)或末段(tail)。
    窗 k>0 的文本在 <think> 后直接从思考中段开始, 与推理时的前缀不一样 —— 和 mid 一样的妥协, 训练器 maxlen 放不下整条。
    finish≠stop 的条(撞 max_tokens 没收口)思考仍是真轨迹, 只收思考窗、不接 </think>+正文; ok 的条最后一窗带 </think>+正文(正文超窗再切)。
    每条最多 max_chunks 窗: 超了取前 max_chunks-1 窗 + 最后一窗(正文在那)。
    只有收口条的最后一窗是真结尾; 其余窗与没收口的条写 "eos": false(训练器默认给 text 行补 EOS, 补在思考中间 = 教它想到一半就停)。
    整条一行(--max-tok 1000000 --max-chunks 1)交给训练器按 ctxlen 自己切, 是 ③ 长上下文的用法(restore_B2ctx_*.cfg)。"""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.hf, "tokenizer.json"))
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    prefix = SYS + EFFORT_PREFIX.format(budget=EFFORT_BUDGET[a.effort])
    rows = [r for r in read_jsonl(a.raw) if r.get("reasoning")]
    kept, man, ntrace = 0, [], 0
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            head = prefix + USER + r["q"] + ASST + THINK
            budget = a.max_tok - 2 - len(enc(head))      # BOS + 训练器补的 EOS
            fin = r.get("ok") and not r.get("head")      # head(api_branch.py head 截的思考开头)标了 ok 只为 ptf 能配, 它没有结尾: 不接 </think>、不补 EOS
            body = r["reasoning"] + (THINK_END + (r.get("content") or "") if fin else "")
            units = []
            for u in split_units(body):                  # 段太长按句再切, 保证单个单元放得进一窗
                units += split_sentences(u) if len(enc(u)) > budget else [u]
            chunks, cur = [], ""
            for u in units:
                if len(enc(cur + u)) <= budget:
                    cur += u
                else:
                    if cur:
                        chunks.append(cur)
                    cur = u if len(enc(u)) <= budget else u[: len(u) // 2]   # 单句仍超窗(极长代码行)只能硬截一半
            if cur:
                chunks.append(cur)
            if len(chunks) > a.max_chunks:
                chunks = chunks[: a.max_chunks - 1] + chunks[-1:]
            ntrace += 1
            for k, c in enumerate(chunks):
                text = head + c
                ids = enc(text)
                while len(ids) + 2 > a.max_tok:          # BPE 拼接边界多出的 token: 从尾掐一句
                    ss = split_sentences(c)
                    c = "".join(ss[:-1]) if len(ss) > 1 else c[: int(len(c) * 0.9)]
                    text = head + c; ids = enc(text)
                assert EOS not in text and ids.count(enc(THINK_END)[0]) <= 1, r["id"]
                row = {"text": text}
                if not (fin and k == len(chunks) - 1):   # 训练器默认给 text 行末尾补 EOS 当靶: 只有收口轨迹的最后一窗是真结尾, 别的窗/没收口的条标 eos false
                    row["eos"] = False
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); kept += 1
                man.append((r["id"], r["src"], "ok" if fin else "partial", k, len(chunks), len(ids) + 2, int(THINK_END in c)))
    with open(a.out + ".manifest.txt", "w", encoding="utf-8") as f:
        f.write("# id src ok/partial 窗号 本条窗数 整窗tok(含BOS/EOS) 含</think>+正文\n")
        for m in man:
            f.write(" ".join(str(x) for x in m) + "\n")
    tot = [m[5] for m in man]
    print(f"chunks: {ntrace} 条轨迹 → {kept} 窗 → {a.out}; 整窗 token 均值 {sum(tot)/len(tot):.0f} 最长 {max(tot)} 最短 {min(tot)}; "
          f"带正文的窗 {sum(m[6] for m in man)}; 清单 {a.out}.manifest.txt")


# ---------- code(10-24: 图形代码域的料, docs/graphics_sidecar.md §2) ----------
FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)


def cmd_code(a):
    """collect 的产物 → 领域语料目录: 每条取正文里最长的一个代码围栏(整份程序), 按 src 分文件写 <out>/<src>.jsonl({"text": 代码});
    stage_idshalf 目录模式一文件=一子域、一行=一篇, 之后 amp_campaign.sh --profile <域> sidecar 直接吃。
    只收 ok 的条; 围栏缺或太短(< min-chars)的丢掉并计数 —— 半份程序进语料只会把坐标区的靶做脏。"""
    os.makedirs(a.out, exist_ok=True)
    # ★不泄题★: --exclude-ids 给判决用的 ids(鹈鹕三份页面), 命中【内容区】16-gram 的整篇不要 —— 尺上的页面哪怕一段进了拟合份, 读数就是假的。
    # 15:38 实撞: 按"任一 16-gram"剔, 60 篇剔掉 37 —— 命中的全是 <meta name="viewport" content="width=device-width…> 这类 HTML 头样板,
    # 任何网页都有, 不是鹈鹕页面的内容。改成只看 ids 旁的 .regions(judge_rows_tokclass.py --html 产出): 片段在页面那边的标签落在
    # 文字/注释任一个, 或坐标区且片段里有真数字(两位以上或带小数点: 鹈鹕的 412.5 这种; x1="0" y1="1" 这类渐变样板不算)才算泄题;
    # 标签名/属性值(content="width=device-width…" 也是属性值)/CSS/JS 收尾样板/空白的片段不算。没有 .regions 的 ids 退回"任一命中"的严格口径。
    # 15:4x 二撞: 第一版把 attr/js 也算内容, 还是剔掉 37 篇 —— viewport 的 content=… 是属性值, requestAnimationFrame 收尾是 JS 样板。
    NG, jp, tok = 16, {}, None
    REALNUM = re.compile(r"\d{2,}|\d\.\d")
    if a.exclude_ids:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(a.hf, "tokenizer.json"))
        for p in a.exclude_ids:
            ids = [int(x) for x in open(p).read().split()]
            regs = open(p + ".regions").read().split() if os.path.exists(p + ".regions") else None
            for i in range(len(ids) - NG + 1):
                g = tuple(ids[i:i + NG])
                if regs is None: leak = True
                else:
                    rs = regs[i:i + NG]
                    leak = any(r in ("text", "comment") for r in rs) or ("coord" in rs and bool(REALNUM.search(tok.decode(list(g)))))
                jp[g] = jp.get(g, False) or leak
    by, drop = {}, []
    for r in read_jsonl(a.raw):
        if not r.get("ok"):
            drop.append((r["id"], "not ok")); continue
        blocks = FENCE.findall(r.get("content") or "")
        code = max(blocks, key=len) if blocks else ""
        if len(code) < a.min_chars:
            drop.append((r["id"], f"围栏 {len(code)} 字 < {a.min_chars}")); continue
        if tok:
            ids = tok.encode(code, add_special_tokens=False).ids
            grams = [tuple(ids[i:i + NG]) for i in range(len(ids) - NG + 1)]
            boiler = sum(1 for g in grams if g in jp and not jp[g])
            leaks = [g for g in grams if jp.get(g)]
            if leaks:
                drop.append((r["id"], f"与判决页面共享 {len(leaks)} 个内容区 {NG}-gram(另 {boiler} 个样板), 如 {tok.decode(list(leaks[0]))[:60]!r}")); continue
            if boiler: print(f"  留 {r['id']}: 只共享 {boiler} 个样板 {NG}-gram(标签/CSS/空白), 不算泄题")
        by.setdefault(r["src"], []).append(code)
    for src, codes in sorted(by.items()):
        with open(os.path.join(a.out, f"{src}.jsonl"), "w", encoding="utf-8") as f:
            for c in codes:
                f.write(json.dumps({"text": c}, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "manifest.txt"), "w", encoding="utf-8") as f:
        f.write(f"# think_trace_collect.py code: {a.raw} → 子域 " + " ".join(f"{s} {len(c)} 篇" for s, c in sorted(by.items())) + f"; 丢 {len(drop)}\n")
        for i, why in drop: f.write(f"丢 {i}: {why}\n")
    print(f"code: " + ", ".join(f"{s} {len(c)} 篇({sum(map(len, c)) // 1000} K 字)" for s, c in sorted(by.items())) + f"; 丢 {len(drop)} → {a.out}")
    for i, why in drop: print(f"  丢 {i}: {why}")


def cmd_ids(a):
    """collect 的产物 → 判决尺的 ids(10-27, 泛化留出尺): 每条 ok 的官方轨迹渲染成部署同款字节 BOS + 头 + 题 + <think>思考</think>正文,
    写 <out-dir>/<id>.ids(一行一个 id, 与 v41_judge.sh / 鹈鹕设计推理段同格式), 清单 <out-dir>/list.txt 一行 "id 全长 生成段起点"。
    为什么: 鹈鹕一道题判不了泛化 —— 训练里一道没见过的题上, 学生对官方设计推理的逐位贴合度涨没涨(段 C 读这份清单), 才是泛化的数。
    生成段起点 = 头 + 题 + <think> 的 token 数, 段 C 只统计它之后(题面是给定的, 不算模型写的)。不写 EOS: 尺只看到正文末位。
    出错会怎样: 前缀单独分词与整条分词在分界处不一致(BPE 跨界合并)就停 —— 起点错一位, 分桶全是假数。"""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.hf, "tokenizer.json"))
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    bos = "<｜begin▁of▁sentence｜>"
    os.makedirs(a.out_dir, exist_ok=True)
    rows = [r for r in read_jsonl(a.raw) if r.get("ok")]
    with open(os.path.join(a.out_dir, "list.txt"), "w", encoding="utf-8") as lst:
        for r in rows:
            pre = bos + SYS + EFFORT_PREFIX.format(budget=EFFORT_BUDGET[a.effort]) + USER + r["q"] + ASST + THINK
            ids, pid = enc(pre + r["reasoning"] + THINK_END + r["content"]), enc(pre)
            if ids[:len(pid)] != pid:
                sys.exit(f"★{r['id']}: 题面前缀与整条分词在分界处不一致, 停★")
            with open(os.path.join(a.out_dir, f"{r['id']}.ids"), "w") as f:
                f.write("\n".join(map(str, ids)) + "\n")
            lst.write(f"{r['id']} {len(ids)} {len(pid)}\n")
            print(f"  {r['id']}: {len(ids)} 位, 生成段从 {len(pid)} 起")
    print(f"ids: {len(rows)} 条 → {a.out_dir}/list.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("questions"); q.add_argument("--humaneval"); q.add_argument("--n-code", type=int, default=100)
    q.add_argument("--exclude", nargs="*"); q.add_argument("--general"); q.add_argument("--set", nargs="*"); q.add_argument("--out", required=True)
    c = sp.add_parser("collect"); c.add_argument("--key", required=True); c.add_argument("--questions", required=True); c.add_argument("--out", required=True)
    c.add_argument("--model", default="deepseek-flash"); c.add_argument("--max-tokens", type=int, default=32768)
    c.add_argument("--jobs", type=int, default=4); c.add_argument("--limit", type=int, default=0); c.add_argument("--timeout", type=int, default=1800)
    t = sp.add_parser("tail"); t.add_argument("--hf", required=True); t.add_argument("--raw", required=True); t.add_argument("--out", required=True)
    t.add_argument("--max-tok", type=int, default=1000); t.add_argument("--min-tail", type=int, default=96); t.add_argument("--effort", default="high", choices=list(EFFORT_BUDGET))
    m = sp.add_parser("mid"); m.add_argument("--hf", required=True); m.add_argument("--raw", required=True); m.add_argument("--out", required=True)
    m.add_argument("--max-tok", type=int, default=1000); m.add_argument("--min-mid", type=int, default=96); m.add_argument("--min-rest", type=int, default=192)
    m.add_argument("--frac", type=float, default=0.7); m.add_argument("--effort", default="high", choices=list(EFFORT_BUDGET))
    k = sp.add_parser("code"); k.add_argument("--raw", required=True); k.add_argument("--out", required=True); k.add_argument("--min-chars", type=int, default=1500)
    k.add_argument("--exclude-ids", nargs="*"); k.add_argument("--hf")
    h = sp.add_parser("chunks"); h.add_argument("--hf", required=True); h.add_argument("--raw", required=True); h.add_argument("--out", required=True)
    h.add_argument("--max-tok", type=int, default=4096); h.add_argument("--max-chunks", type=int, default=6); h.add_argument("--effort", default="high", choices=list(EFFORT_BUDGET))
    d = sp.add_parser("ids"); d.add_argument("--hf", required=True); d.add_argument("--raw", required=True); d.add_argument("--out-dir", required=True)
    d.add_argument("--effort", default="high", choices=list(EFFORT_BUDGET))
    a = ap.parse_args()
    {"questions": cmd_questions, "collect": cmd_collect, "tail": cmd_tail, "mid": cmd_mid, "code": cmd_code, "chunks": cmd_chunks, "ids": cmd_ids}[a.cmd](a)


if __name__ == "__main__":
    main()
