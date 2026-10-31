#!/usr/bin/env python3
"""api_teacher.py — 官方 API 当 ③ 的软靶教师(2026-10-26, 用户问"教师成本能不能用官方 API")。

为什么: 本地 FP 教师(v41_teacher.py, 逐层流式装 476 GB 原件)实测 33~35 位/s, 27 条图形轨迹 ≈ 6.8 小时; 官方 API 开 logprobs 生成是 150~170 tok/s,
而且思考段(reasoning_content)也逐 token 给 top-20 分布(10-26 探针实测)。代价: K 只有 20(训练器 topk=20, 不够的质量进 rest 桶);
API 给的是 token 字串/bytes 不是 id, 要用 HF 分词器的字节级词表反查(probe-align 验逐位同)。
已知坑(10-26 探针): chat/completions 不给 </think> 与 EOS 这两行的分布(usage 比分布条数多 1, 决策位正好缺) —— probe-echo 探 beta/completions
的 echo 能不能把整条序列的逐位分布一次吐回来(能 = 现有轨迹直接拿去打表, 决策位也在)。

子命令(正式两条, 在 spark 上跑 —— spark 出得去 api.deepseek.com, 而且 bytes→id 要 hf/ 下的 tokenizer.json):
  collect --key K --questions Q --out raw.jsonl --hf HF [--jobs 4 --max-tokens 65536]
      Q 可以是题单(有 q 字段)或训练行(text 里 <｜User｜>…<｜Assistant｜>, 取中间当题)。每题流式生成 + top-20, 当场 bytes→id, 并把思考/正文重切一遍核逐位同(align_ok)。
      输出行与 think_trace_collect.py collect 同字段(id q reasoning content finish usage ok …) ⇒ 直接喂 `think_trace_collect.py chunks --max-chunks 1` 出训练行;
      多一个 lp 字段: {"reasoning_content": [[id, logprob, [候选 id…], [候选 logprob…]]…], "content": […]}。ok 还要求 align_ok。
  ptf --ids teacher.ids --raw raw.jsonl --out part.ptf --covered part.ids --topk 20
      训练器 dumpids 导出的每条序列, 在 raw 里找"思考 id 块 + 正文 id 块"都整段出现的那条轨迹: 这两块的行写 API top-20(查不到 id 的候选并进 rest),
      其余行(模板 / </think> / EOS —— API 不给这两个决策位的分布, 10-26 探针)写 one-hot。写 PTF1(core_ptrain_fp.c 的格式); 对上的序列原行抄进 --covered,
      posttrain_fp_spark.sh 把它当已打好的一份, 只给剩下的(保持料 / 没对上的)跑本地 FP 教师。
探针(10-26 用过, 留着复核): probe-logprobs / probe-align / probe-echo / probe-hfcompare。
共用的常量(API_URL / 模板片段 / EFFORT)从 think_trace_collect.py 引, 两边不许各写一份。
出错会怎样: key 文件读不到 → 直接报错退出; 探针对 API 的 error 原样打出; collect 一题失败重试 5 次后记 ok:false(下次续跑只补 ok:false 的); ptf 没对上的序列不写, 打印条数, 由 FP 教师兜。"""
import base64
import struct
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import math
import os
import sys
import time
import urllib.request
import http.client

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from think_trace_collect import API_URL, EFFORT_PREFIX, EFFORT_BUDGET, SYS, USER, ASST, THINK, THINK_END, EOS  # noqa: E402

COMPLETIONS_URL = "https://api.deepseek.com/beta/completions"


def post(url, key, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    return urllib.request.urlopen(req, timeout=timeout)


def cmd_probe_logprobs(a):
    key = open(a.key, encoding="utf-8").read().strip()
    body = {"model": a.model, "messages": [{"role": "user", "content": a.q}], "max_tokens": a.max_tokens, "logprobs": True, "top_logprobs": 20,
            "stream": True, "stream_options": {"include_usage": True}}
    ent = {"reasoning_content": [], "content": []}; txt = {"reasoning_content": [], "content": []}; usage = finish = None; t0 = time.time()
    with post(API_URL, key, body, a.timeout) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data:") or ln[5:].strip() == "[DONE]":
                continue
            ev = json.loads(ln[5:]); usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}; lp = ch.get("logprobs") or {}
                for k in ent:
                    if d.get(k): txt[k].append(d[k])
                    ent[k].extend(lp.get(k) or [])
                finish = ch.get("finish_reason") or finish
    secs = time.time() - t0
    json.dump({"q": a.q, "model": a.model, "usage": usage, "finish": finish, "secs": secs, "text": {k: "".join(v) for k, v in txt.items()}, "logprobs": ent},
              open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    u = usage or {}; ct = u.get("completion_tokens", 0); rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    nr, nc = len(ent["reasoning_content"]), len(ent["content"])
    print(f"finish {finish}; usage completion {ct} (reasoning {rt}); 分布条数 reasoning {nr} / content {nc}; 没给分布的 token {ct - nr - nc} 个; {ct / secs:.1f} tok/s → {a.out}")
    for k, v in ent.items():
        if not v: continue
        mass = sorted(sum(math.exp(t["logprob"]) for t in e.get("top_logprobs") or []) for e in v)
        hit = sum(1 for e in v if any(t["token"] == e["token"] for t in e.get("top_logprobs") or []))
        print(f"  {k}: top-20 质量 中位 {mass[len(mass)//2]:.3f} / p10 {mass[len(mass)//10]:.3f} / 最低 {mass[0]:.3f}; 采样 token 在自己 top-20 里 {hit}/{len(v)}; "
              f"bytes 为空 {sum(1 for e in v if e.get('bytes') is None)}; 首 2 个 {[e['token'] for e in v[:2]]} 末 2 个 {[e['token'] for e in v[-2:]]}")
    special = [e["token"] for k in ent for e in ent[k] if e["token"] in (THINK, THINK_END, EOS)]
    print(f"  决策位 token 出现在分布列表里: {special or '没有'}")


def hf_byte_table(hf):
    """词表是字节级 BPE(GPT-2 式 256 字节↔可见字符表), 反查出 bytes→id。特殊 token 不是字节串, 查不到就跳过。"""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(hf, "tokenizer.json"))
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    u2b = {chr(c): b for b, c in zip(bs, cs)}; b2id = {}
    for t, i in tok.get_vocab().items():
        if all(ch in u2b for ch in t): b2id[bytes(u2b[ch] for ch in t)] = i
    return tok, b2id


def cmd_probe_align(a):
    d = json.load(open(a.probe, encoding="utf-8")); tok, b2id = hf_byte_table(a.hf)
    for k in ("reasoning_content", "content"):
        ent, text = d["logprobs"][k], d["text"][k]
        if not ent: continue
        ab = [bytes(e["bytes"] or []) for e in ent]; ids_api = [b2id.get(b) for b in ab]; ids_hf = tok.encode(text, add_special_tokens=False).ids
        first = next((i for i, (x, y) in enumerate(zip(ids_api, ids_hf)) if x != y), None)
        alt = [t for e in ent for t in e.get("top_logprobs") or []]; alt_miss = sum(1 for t in alt if b2id.get(bytes(t["bytes"] or [])) is None)
        print(f"{k}: API {len(ent)} tok / HF 重切 {len(ids_hf)} tok; 字节拼回原文 {b''.join(ab) == text.encode()}; bytes→id 查不到 {ids_api.count(None)}; "
              f"与 HF 逐位同到 {'全程' if first is None and len(ids_api) == len(ids_hf) else first}; top-20 候选 {len(alt)} 个里查不到 id {alt_miss}")


def cmd_probe_echo(a):
    """prompt = 部署同款模板(System 里 Reasoning Effort 75 + User + Assistant + <think>) + 这题官方自己的思考原文; 让它接 N 个 token。
    看三件事: ① echo 回来的 prompt 位有没有 logprobs(有 = 整条现成轨迹一次打表) ② 接出来的第一个 token 是不是 </think>, 它那一行有没有 top-20 ③ 模板里的特殊 token 被切成一个还是几个。"""
    key = open(a.key, encoding="utf-8").read().strip(); d = json.load(open(a.probe, encoding="utf-8"))
    prompt = SYS + EFFORT_PREFIX.format(budget=EFFORT_BUDGET["high"]) + USER + d["q"] + ASST + THINK + d["text"]["reasoning_content"]
    if a.with_content: prompt += THINK_END + d["text"]["content"]   # 接在正文末尾: 下一个该是 EOS, 看 EOS 那一行
    # 文档写 echo 不能与 logprobs 同开(10-26 查), 默认不开; --echo 只为核对文档说的对不对
    body = {"model": a.model, "prompt": prompt, "max_tokens": a.max_tokens, "logprobs": 20, "temperature": 1.0, **({"echo": True} if a.echo else {})}
    t0 = time.time()
    try:
        with post(COMPLETIONS_URL, key, body, a.timeout) as r: res = json.load(r)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:600]}"); return
    print(f"{time.time()-t0:.1f}s; 顶层键 {list(res.keys())}; usage {res.get('usage')}")
    ch = (res.get("choices") or [{}])[0]; lp = ch.get("logprobs") or {}
    print(f"finish {ch.get('finish_reason')}; text 开头 {ch.get('text', '')[:80]!r}")
    print(f"logprobs 键 {list(lp.keys()) if isinstance(lp, dict) else lp}")
    toks, tlp, top = lp.get("tokens") or [], lp.get("token_logprobs") or [], lp.get("top_logprobs") or []
    print(f"tokens {len(toks)} / token_logprobs {len(tlp)}(None {sum(1 for x in tlp if x is None)}) / top_logprobs {len(top)}; prompt 字 {len(prompt)}")
    print(f"首 6 个 token {toks[:6]}"); print(f"末 8 个 token {toks[-8:]}")
    for i, t in enumerate(toks):
        if t in (THINK, THINK_END, EOS, USER, ASST, SYS):
            tp = top[i] if i < len(top) else None
            print(f"  特殊 token 位 {i} {t!r}: logprob {tlp[i] if i < len(tlp) else None}; top-20 {len(tp) if isinstance(tp, dict) else tp}" + (f" 前 3 {sorted(tp.items(), key=lambda kv: -kv[1])[:3]}" if isinstance(tp, dict) and tp else ""))
    json.dump(res, open(a.probe + ".echo.json", "w", encoding="utf-8"), ensure_ascii=False)


def cmd_probe_raw(a):
    """裸续写探针(10-28 段 B 第一层"原始语料"): beta/completions 给一段裸前缀(源文件开头, 不套对话模板), 看它接出来像不像源文件、
    逐位 top-20 是什么格式(旧式接口的 top_logprobs 是 字串→logprob, 可能没有 bytes)、字串能不能逐位对回 HF 词表 id。"""
    key = open(a.key, encoding="utf-8").read().strip(); tok, b2id = hf_byte_table(a.hf)
    body = {"model": a.model, "prompt": a.prompt.replace("\\n", "\n"), "max_tokens": a.max_tokens, "logprobs": 20, "temperature": 1.0}
    t0 = time.time()
    try:
        with post(COMPLETIONS_URL, key, body, a.timeout) as r: res = json.load(r)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:600]}"); return
    ch = (res.get("choices") or [{}])[0]; lp = ch.get("logprobs") or {}
    print(f"{time.time()-t0:.1f}s; usage {res.get('usage')}; finish {ch.get('finish_reason')}; logprobs 键 {list(lp.keys()) if isinstance(lp, dict) else type(lp)}")
    toks, top = lp.get("tokens") or [], lp.get("top_logprobs") or []
    print(f"tokens {len(toks)}; top_logprobs {len(top)} 条, 第一条类型 {type(top[0]).__name__ if top else None}: {str(top[0])[:300] if top else ''}")
    miss = sum(1 for t in toks if t.encode("utf-8") not in b2id)
    print(f"按 utf-8 字节查 HF 词表: tokens 里查不到 {miss}/{len(toks)}; 前 12 个 {toks[:12]}")
    print("—— 续写原文 ——\n" + ch.get("text", "")[:1500])
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)


def cmd_probe_hfcompare(a):
    """API 的分布和本仓判决尺的 FP 教师(v41_teacher.py, 出厂原件)是不是一回事: 五指标全锚在 HF 原件上, API 若是另一套精度/另一版权重, 拿它当教师会把 ③ 往错的锚拉。
    第一步(没有 --teacher-bin 或文件不在): 把 探针 json 还原成教师要吃的 id 串 = BOS + 部署同款模板 + API 的思考 token(bytes→id), 写 --ids, 外面拿 v41_teacher.py --ids --ntok --out 打 logits。
    第二步(有 .bin): 逐位比 API top-20 与 HF 同位 softmax: Same top / 采样 token 的 |Δp| / Σmin(top-20 ∪ rest) / KL(HF‖API)。.bin 格式 = anchor_metrics 的 <i32 S><i32 V><f32 S×V>。"""
    import numpy as np
    d = json.load(open(a.probe, encoding="utf-8")); tok, b2id = hf_byte_table(a.hf)
    pre = [0] + tok.encode(SYS + EFFORT_PREFIX.format(budget=EFFORT_BUDGET["high"]) + USER + d["q"] + ASST + THINK, add_special_tokens=False).ids
    ent = d["logprobs"]["reasoning_content"]; ids = pre + [b2id[bytes(e["bytes"])] for e in ent]
    if not (a.teacher_bin and os.path.exists(a.teacher_bin)):
        open(a.ids, "w").write(" ".join(map(str, ids)) + "\n")
        print(f"ids → {a.ids}: {len(ids)} 位(模板 {len(pre)} + 思考 {len(ent)}); 下一步 v41_teacher.py <HF> --ids {a.ids} --ntok {len(ids)} --out <bin>"); return
    with open(a.teacher_bin, "rb") as f: S, V = __import__("struct").unpack("<ii", f.read(8))
    lg = np.memmap(a.teacher_bin, dtype=np.float32, mode="r", offset=8, shape=(S, V))
    P = len(pre); same = 0; dp = []; smin = []; kl = []
    for j, e in enumerate(ent):
        row = lg[P + j - 1].astype(np.float64); row -= row.max(); ph = np.exp(row); ph /= ph.sum()
        top = {b2id.get(bytes(t["bytes"] or [])): math.exp(t["logprob"]) for t in e["top_logprobs"]}; top.pop(None, None)
        pa_rest = max(0.0, 1.0 - sum(top.values())); ph_rest = max(0.0, 1.0 - sum(ph[i] for i in top))
        same += int(ph.argmax()) == max(top, key=top.get)
        sid = b2id[bytes(e["bytes"])]; dp.append(abs(math.exp(e["logprob"]) - ph[sid]) if e["logprob"] > -9000 else float("nan"))
        smin.append(sum(min(p, ph[i]) for i, p in top.items()) + min(pa_rest, ph_rest))
        kl.append(sum(ph[i] * math.log(ph[i] / p) for i, p in top.items() if ph[i] > 0 and p > 0) + (ph_rest * math.log(ph_rest / pa_rest) if ph_rest > 1e-9 and pa_rest > 1e-9 else 0.0))
    dp = np.array(dp); smin = np.array(smin); kl = np.array(kl)
    print(f"API vs HF 原件, {len(ent)} 位: Same top {100*same/len(ent):.2f}%; 采样 token |Δp| 中位 {np.nanmedian(dp):.4f} p95 {np.nanpercentile(dp, 95):.4f}; "
          f"Σmin(top-20∪rest) 均 {smin.mean():.4f} 中位 {np.median(smin):.4f} p5 {np.percentile(smin, 5):.4f}; KL(HF‖API) 均 {kl.mean():.5f} 中位 {np.median(kl):.5f} p95 {np.percentile(kl, 95):.5f}")


def stream_lp(key, model, q, max_tokens, timeout, think=True, messages=None):
    """一题流式拉完, 带 top-20; 返回 (reasoning, content, finish, usage, {列: [原始分布条…]})。
    think=False: 关思考(thinking disabled, v41_api_parity.py 同写法), 官方直接从正文写起, 只有 content 一列 —— 10-28 段 B 自学习:
    思考模式的料正文只占 11~13%, 编程尺量的却是原始代码
    messages 给了就原样发(真实 agent 请求: system + 示例轮 + user, 10-31), q 只当标签"""
    body = {"model": model, "messages": messages or [{"role": "user", "content": q}], "max_tokens": max_tokens, "logprobs": True, "top_logprobs": 20,
            "stream": True, "stream_options": {"include_usage": True}, **({} if think else {"thinking": {"type": "disabled"}})}
    ent = {"reasoning_content": [], "content": []}; txt = {"reasoning_content": [], "content": []}; usage = finish = None
    with post(API_URL, key, body, timeout) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data:") or ln[5:].strip() == "[DONE]":
                continue
            ev = json.loads(ln[5:]); usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}; lp = ch.get("logprobs") or {}
                for k in ent:
                    if d.get(k): txt[k].append(d[k])
                    ent[k].extend(lp.get(k) or [])
                finish = ch.get("finish_reason") or finish
    return "".join(txt["reasoning_content"]), "".join(txt["content"]), finish, usage, ent


def text_from_lp(rows, id2b):
    """正文/思考的文字按 token 的 bytes 拼回来, 不用 delta 文字: API 把 reasoning_content 末尾的换行吃了(实撞 10-26: 末 token 是 b' \\n', 文字只剩 b' '),
    训练行若用 delta 文字, 训练器重切出来的 id 就和 API 的对不上(think198 有 31+ 条)。全部 id 都认识才拼, 否则 None。"""
    if any(r[0] is None or r[0] not in id2b for r in rows): return None
    return b"".join(id2b[r[0]] for r in rows).decode("utf-8", "replace")


def realign(rec, tok, id2b):
    """按 lp 重建 reasoning/content 文字并核 align_ok / ok; 返回改好的 rec"""
    if not rec.get("lp"): return rec
    align = True
    for k, f in (("reasoning_content", "reasoning"), ("content", "content")):
        rows = rec["lp"].get(k) or []
        t = text_from_lp(rows, id2b)
        if t is not None: rec[f] = t
        align = align and t is not None and [r[0] for r in rows] == tok.encode(rec[f], add_special_tokens=False).ids
    rec["align_ok"] = align   # 只是记账: 偶有非规范切分(中文词边界, 198 条里 5 条各 1~2 处), ptf 按字节偏移对位, 那一两位写 one-hot, 整条不作废
    # 关思考的条没有思考段; 裸续写(raw)是文档不是回答, 写到 max_tokens 停(finish=length)也算一份完整料
    rec["ok"] = (rec.get("finish") == "stop" or bool(rec.get("raw"))) and (bool(rec.get("reasoning")) or rec.get("think") is False) and bool(rec.get("content"))
    rec.setdefault("src", "api")   # think_trace_collect.py chunks 的清单要这个字段
    return rec


def raw_lp(key, model, prompt, max_tokens, timeout):
    """裸续写(10-28 段 B 第一层"原始语料"): beta/completions 拿 prompt 当文档开头接着写, 不套对话模板 —— 两把尺(编程尺 = 裸源文件,
    wt2 = 裸百科)量的都是这种"接续文档"的能力, 对话料里没有。返回与 stream_lp 同形(content 一列); 旧式接口的分布是 字串→logprob,
    这里转成对话接口那种 {token, bytes, logprob, top_logprobs:[{token, bytes, logprob}]}, 后面 bytes→id 的路共用。"""
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "logprobs": 20, "temperature": 1.0}
    with post(COMPLETIONS_URL, key, body, timeout) as r: res = json.load(r)
    ch = (res.get("choices") or [{}])[0]; lp = ch.get("logprobs") or {}
    bt = lambda s: list(s.encode("utf-8"))
    ent = [{"token": t, "bytes": bt(t), "logprob": l, "top_logprobs": [{"token": c, "bytes": bt(c), "logprob": v} for c, v in (top or {}).items()]}
           for t, l, top in zip(lp.get("tokens") or [], lp.get("token_logprobs") or [], lp.get("top_logprobs") or [])]
    return "", ch.get("text", ""), ch.get("finish_reason"), res.get("usage"), {"reasoning_content": [], "content": ent}


def collect_one(key, model, item, max_tokens, timeout, tok, b2id, id2b, think=True, raw=False):
    delay = 5
    for attempt in range(5):
        try:
            t0 = time.time()
            reasoning, content, finish, usage, ent = (raw_lp(key, model, item["q"], max_tokens, timeout) if raw
                                                      else stream_lp(key, model, item["q"], max_tokens, timeout, think, item.get("messages")))
            # 裸续写第一位就采到 EOS(0 位): 多数是温 1 采样本身(10-28 同一个开头三采一次), 同一个开头再采;
            #   第三次起开头补一个换行 —— Go 的 "package x\n" 后面几乎总是空行, 词表里那是一个 "\n\n" token, 停在单个换行上它从没见过这种切法,
            #   o004 四采全是 EOS; 补成 "\n\n" 结尾就是规范切分。实际发出去的开头写回 q(行的 prefix 用它, 与官方看到的逐字同)
            for k in range(3 if raw and not ent["content"] else 0):
                if k == 1: item = dict(item, q=item["q"] + "\n")
                reasoning, content, finish, usage, ent = raw_lp(key, model, item["q"], max_tokens, timeout)
                if ent["content"]: break
            lp = {}
            for k in ("reasoning_content", "content"):
                rows = []
                for e in ent[k]:
                    cid = [b2id.get(bytes(t["bytes"] or [])) for t in e.get("top_logprobs") or []]
                    rows.append([b2id.get(bytes(e["bytes"] or [])), e["logprob"], [c for c in cid if c is not None], [t["logprob"] for c, t in zip(cid, e["top_logprobs"]) if c is not None]])
                lp[k] = rows
            return realign(dict(item, reasoning=reasoning, content=content, finish=finish, usage=usage, model=model, effort="high" if think and not raw else "none",
                                think=think and not raw, raw=raw, secs=round(time.time() - t0, 1), lp=lp), tok, id2b)
        # http.client.HTTPException(10-27 实撞): 流式读到一半上游断开抛 IncompleteRead, 它不是 OSError, 漏接就把整个 collect 进程打崩(52 道停在 46)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError, http.client.HTTPException) as e:
            msg = f"HTTP {e.code}" if isinstance(e, urllib.error.HTTPError) else type(e).__name__
            if attempt == 4:
                return dict(item, ok=False, finish=f"error:{msg}", reasoning="", content="", usage=None, lp={})
        print(f"  [{item['id']}] {msg}, {delay}s 后重试", flush=True); time.sleep(delay); delay *= 3


def cmd_collect(a):
    key = open(a.key, encoding="utf-8").read().strip(); tok, b2id = hf_byte_table(a.hf); id2b = {i: b for b, i in b2id.items()}
    qs = []
    for i, ln in enumerate(open(a.questions, encoding="utf-8")):
        if not ln.strip(): continue
        r = json.loads(ln); t = r.get("text", ""); u, s = t.find(USER), t.find(ASST)
        q = r.get("q") or (t[u + len(USER):s] if 0 <= u < s else None)
        if r.get("messages") and not q: q = [m for m in r["messages"] if m["role"] == "user"][-1]["content"]   # 多条消息的行: q = 最后一条 user, 只当标签
        if q: qs.append({"id": str(r.get("id") or f"{os.path.basename(a.questions)}#{i}"), "q": q, **({"messages": r["messages"], "src": r.get("src", "task")} if r.get("messages") else {})})
    done = set(); rows = []
    if os.path.exists(a.out):
        for ln in open(a.out, encoding="utf-8"):
            r = json.loads(ln); rows.append(r)
            if r.get("ok"): done.add(r["id"])
    todo = [q for q in qs if q["id"] not in done]
    print(f"collect: {len(qs)} 题, 已有 ok {len(done)}, 本次 {len(todo)}, {a.jobs} 路, max_tokens {a.max_tokens}", flush=True)
    lock = threading.Lock(); t0 = time.time(); nok = 0
    with open(a.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(collect_one, key, a.model, q, a.max_tokens, a.timeout, tok, b2id, id2b, not a.no_think, a.raw) for q in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result(); nok += bool(r["ok"])
            with lock: f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            u = r.get("usage") or {}; ct = u.get("completion_tokens", 0)
            print(f"  {i}/{len(todo)} [{r['id']}] {'ok' if r['ok'] else '★' + str(r['finish']) + ('' if r.get('align_ok', True) else ' 重切不同')} "
                  f"{ct} tok {r.get('secs', '?')} s{(' = %.0f tok/s' % (ct / r['secs'])) if r.get('secs') else ''}, 累计 {time.time() - t0:.0f} s", flush=True)
    print(f"collect 完成: 本次 ok {nok}/{len(todo)} → {a.out}", flush=True)


def cmd_fixtext(a):
    """已采好的 raw 按 realign 重写(文字从 token bytes 拼回, 重算 ok/align_ok); 原文件留 .orig"""
    tok, b2id = hf_byte_table(a.hf); id2b = {i: b for b, i in b2id.items()}
    recs = [realign(json.loads(ln), tok, id2b) for ln in open(a.raw, encoding="utf-8") if ln.strip()]
    if not os.path.exists(a.raw + ".orig"): os.rename(a.raw, a.raw + ".orig")
    with open(a.raw, "w", encoding="utf-8") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"fixtext {a.raw}: {len(recs)} 条, ok {sum(1 for r in recs if r.get('ok'))}, align_ok {sum(1 for r in recs if r.get('align_ok'))}, finish=stop {sum(1 for r in recs if r.get('finish') == 'stop')}")


def cmd_diffalign(a):
    """collect 标了 align_ok:false 的轨迹, API 自己吐的切分和 HF 重切差在哪: 首个不同位前后各 3 个 token、不同的位置数、多少位后又对齐。"""
    tok, b2id = hf_byte_table(a.hf); id2b = {i: b for b, i in b2id.items()}
    shown = 0
    for ln in open(a.raw, encoding="utf-8"):
        t = json.loads(ln)
        if t.get("align_ok", True) or not t.get("lp"): continue
        for k, text in (("reasoning_content", t["reasoning"]), ("content", t["content"])):
            api = [r[0] for r in t["lp"][k]]; hf = tok.encode(text, add_special_tokens=False).ids
            if api == hf: continue
            # 按字节偏移对齐: 两边各自的 token 起点集合, 共同起点 = 可直接对上的位置
            def offs(ids):
                o, p = [], 0
                for i in ids: o.append(p); p += len(id2b.get(i, b""))
                return o
            oa, oh = offs(api), offs(hf); common = set(oa) & set(oh)
            same = sum(1 for p, i in zip(oh, hf) if p in common and api[oa.index(p)] == i)
            i0 = next((i for i, (x, y) in enumerate(zip(api, hf)) if x != y), min(len(api), len(hf)))
            print(f"[{t['id']}] {k}: API {len(api)} / HF {len(hf)} tok; 起点+id 都同的位 {same}({100*same/len(hf):.1f}%); 首个不同在 {i0}: "
                  f"API {[id2b.get(i, b'?') for i in api[max(0, i0-1):i0+3]]} | HF {[id2b.get(i, b'?') for i in hf[max(0, i0-1):i0+3]]}")
        shown += 1
        if shown >= a.n: break


def find_block(seq, blk):
    """blk 在 seq 里整段出现的位置(-1 = 没有); 先按前 8 个 id 找候选再整段核"""
    n = len(blk); h = min(8, n); head = blk[:h]   # 块可能不到 8 个 token(正文只有两个词的短题, 10-26 实撞 2 条)
    for p in range(len(seq) - n + 1):
        if seq[p:p + h] == head and seq[p:p + n] == blk: return p
    return -1


def cmd_ptf(a):
    import numpy as np
    tok, b2id = hf_byte_table(a.hf); id2b = {i: b for b, i in b2id.items()}
    K = a.topk; traces = [json.loads(l) for l in open(a.raw, encoding="utf-8")]; traces = [t for t in traces if t.get("ok") and t.get("lp")]
    # 训练器导出的序列里, 思考/正文两块是它自己重切的(规范 BPE); API 的切分偶尔非规范(中文词边界) ⇒ 用规范切分找块, 再按字节偏移把 API 的行对到规范位上:
    #   偏移与 id 都同的位拿 API 分布, 对不上的那一两位留 one-hot(整条不作废)。
    def offs(ids):
        o, p = [], 0
        for i in ids: o.append(p); p += len(id2b.get(i, b""))
        return o
    # 每块备两个版本: 原文 / 去掉行尾空白(训练器读 text 行先 pt_rstrip 去尾部 ' '/'\n'/'\r', core_ptrain_data.c —— 块在序列末尾时导出的序列里就没有那个空白 token;
    #   10-27 实撞: 思考头截在 2048 位、末位恰是 " \n" 的 28 条全对不上, 被迫走本地 FP 教师补表 18 分钟)。匹配先试原文再试去尾版。
    for t in traces:
        t["_blk"] = {}
        for k, text in (("reasoning_content", t["reasoning"]), ("content", t["content"])):
            rows = t["lp"][k]; byoff = {o: r for o, r in zip(offs([r[0] for r in rows]), rows)}; vs = []
            for txt in dict.fromkeys([text, text.rstrip(" \r\n")]):
                canon = tok.encode(txt, add_special_tokens=False).ids
                vs.append((canon, [byoff.get(o) if (byoff.get(o) and byoff[o][0] == i) else None for o, i in zip(offs(canon), canon)]))
            t["_blk"][k] = vs
    lines = [ln for ln in open(a.ids, encoding="utf-8") if ln.strip()]
    n = 0; used = set(); tot_rows = api_rows = onehot_mid = 0
    with open(a.out, "wb") as f, open(a.covered, "w", encoding="utf-8") as cf:
        f.write(struct.pack("<III", 0x31465450, K, 0))
        for ln in lines:
            seq = [int(x) for x in ln.split()]; hit = None
            for ti, t in enumerate(traces):
                if ti in used: continue
                for rv in t["_blk"]["reasoning_content"]:
                    for cv in t["_blk"]["content"]:
                        rid, cid = rv[0], cv[0]
                        pr = find_block(seq, rid)
                        if pr < 0: continue
                        pc = find_block(seq[pr + len(rid):], cid) if cid else -1
                        if cid and pc < 0: continue
                        hit = (ti, pr, pr + len(rid) + pc if cid else None, {"reasoning_content": rv, "content": cv}); break
                    if hit: break
                if hit: break
            if not hit: continue
            ti, pr, pc, blk = hit; used.add(ti); t = traces[ti]; L = len(seq)
            top_id = np.full((L - 1, K), -1, np.int32); top_p = np.zeros((L - 1, K), np.float32); rest = np.zeros(L - 1, np.float32)
            top_id[:, 0] = seq[1:]; top_p[:, 0] = 1.0   # 默认 one-hot(模板 / </think> / EOS 这些 API 不给分布的位, 以及对不上的位)
            for p0, k in ((pr, "reasoning_content"), (pc, "content")):
                if p0 is None: continue
                for j, r in enumerate(blk[k][1]):
                    if r is None: onehot_mid += 1; continue
                    i = p0 + j - 1; ids, lps = r[2][:K], r[3][:K]; ps = np.exp(np.array(lps, np.float64))
                    top_id[i] = -1; top_p[i] = 0; top_id[i, :len(ids)] = ids; top_p[i, :len(ids)] = ps; rest[i] = max(0.0, 1.0 - ps.sum()); api_rows += 1
            f.write(struct.pack("<I", L)); np.array(seq, np.int32).tofile(f); top_id.tofile(f); top_p.tofile(f); rest.tofile(f)
            cf.write(ln if ln.endswith("\n") else ln + "\n"); n += 1; tot_rows += L - 1
        f.seek(0); f.write(struct.pack("<III", 0x31465450, K, n))
    print(f"ptf: 导出序列 {len(lines)} 条, 对上 API 轨迹 {n} 条(轨迹 ok {len(traces)} 条) → {a.out}; 行 {tot_rows}, 其中 API 分布 {api_rows} / one-hot {tot_rows - api_rows}"
          f"(块内切分对不上的 {onehot_mid} 位); 覆盖清单 {a.covered}")


def cmd_simids(a):
    """训练行 → 模拟训练器 dumpids 导出的序列(BOS + text 去尾部空白 + EOS(eos 不是 false 时)), 一行一条: 发车前拿它跑 ptf, 先证明教师表配得上。
    为什么(10-28): 配不上的序列会落到本地 FP 教师补表(流式装 476 GB 原件, 一份 ~18 分钟), 一轮的训练超时扛不住; 等到训练链里才发现就晚了。
    prefix 行(10-28): BOS + prefix 自己分词 + text 自己分词再拼(训练器 pt_add_text 同一分界: 两段各自分词, 不跨界合并)。
    HF 分词与 ds4 分词器在 DS4 特殊 token 上一致(10-20 金标)。段 B 自学习的判决序列也由它出(与训练器看到的序列同一种渲染)。"""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.hf, "tokenizer.json")); bos, eos = tok.token_to_id("<｜begin▁of▁sentence｜>"), tok.token_to_id(EOS)
    n = 0
    with open(a.out, "w") as f:
        for r in (json.loads(l) for l in open(a.rows, encoding="utf-8") if l.strip()):
            if "text" not in r: continue
            pre = tok.encode(r["prefix"], add_special_tokens=False).ids if "prefix" in r else []
            ids = [bos] + pre + tok.encode(r["text"].rstrip(" \r\n"), add_special_tokens=False).ids + ([eos] if r.get("eos", True) else [])
            f.write(" ".join(map(str, ids)) + "\n"); n += 1
    print(f"simids: {n} 条 → {a.out}")


def cmd_miss(a):
    """ptf 没对上的序列(--ids 是 posttrain_fp 切出来的剩余份, 如 teacher.part.002)逐条查: 在 raw 里找思考块最长公共前缀的那条轨迹, 打出在第几位分叉、两边各是什么字节。
    为什么(10-27): v6 有 28 条没对上, 被迫走本地 FP 教师补表 18 分钟 —— 一小时一轮扛不住, 先找对不上的根子。"""
    tok, b2id = hf_byte_table(a.hf); id2b = {i: b for b, i in b2id.items()}
    traces = [t for t in (json.loads(l) for l in open(a.raw, encoding="utf-8")) if t.get("ok") and t.get("lp")]
    blks = [(t["id"], tok.encode(t["reasoning"], add_special_tokens=False).ids) for t in traces]
    for n, ln in enumerate(l for l in open(a.ids, encoding="utf-8") if l.strip()):
        if n >= a.n: break
        seq = [int(x) for x in ln.split()]; best = (0, None, 0, 0)
        for tid, blk in blks:   # 用思考块前 8 位定锚, 再量从锚起能一路对上多少位
            for p in range(len(seq) - 8):
                if seq[p:p + 8] == blk[:8]:
                    k = 0
                    while k < len(blk) and p + k < len(seq) and seq[p + k] == blk[k]: k += 1
                    if k > best[0]: best = (k, tid, p, len(blk))
                    break
        k, tid, p, L = best
        if tid is None: print(f"  序列 {n}: 长 {len(seq)}, 没有任何轨迹的思考块开头出现在里面; 头 {b''.join(id2b.get(i, b'?') for i in seq[:40])[:120]!r}"); continue
        blk = dict(blks)[tid]
        print(f"  序列 {n}: 长 {len(seq)}, 最像 {tid}: 从 {p} 起对上 {k}/{L} 位" + ("" if k >= L else
              f"; 分叉处 序列 {[id2b.get(i, b'?') for i in seq[p + k - 1:p + k + 3]]} | 轨迹 {[id2b.get(i, b'?') for i in blk[k - 1:k + 3]]}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("probe-logprobs"); p.add_argument("--key", required=True); p.add_argument("--q", required=True); p.add_argument("--out", required=True)
    p.add_argument("--model", default="deepseek-flash"); p.add_argument("--max-tokens", type=int, default=8000); p.add_argument("--timeout", type=int, default=1800)
    al = sp.add_parser("probe-align"); al.add_argument("--hf", required=True); al.add_argument("--probe", required=True)
    e = sp.add_parser("probe-echo"); e.add_argument("--key", required=True); e.add_argument("--probe", required=True)
    e.add_argument("--model", default="deepseek-flash"); e.add_argument("--max-tokens", type=int, default=4); e.add_argument("--timeout", type=int, default=600)
    e.add_argument("--echo", action="store_true"); e.add_argument("--with-content", action="store_true")
    pr = sp.add_parser("probe-raw"); pr.add_argument("--key", required=True); pr.add_argument("--hf", required=True); pr.add_argument("--prompt", required=True); pr.add_argument("--out", required=True)
    pr.add_argument("--model", default="deepseek-flash"); pr.add_argument("--max-tokens", type=int, default=300); pr.add_argument("--timeout", type=int, default=600)
    h = sp.add_parser("probe-hfcompare"); h.add_argument("--hf", required=True); h.add_argument("--probe", required=True); h.add_argument("--ids", required=True); h.add_argument("--teacher-bin")
    c = sp.add_parser("collect"); c.add_argument("--key", required=True); c.add_argument("--questions", required=True); c.add_argument("--out", required=True); c.add_argument("--hf", required=True)
    c.add_argument("--model", default="deepseek-flash"); c.add_argument("--max-tokens", type=int, default=65536); c.add_argument("--jobs", type=int, default=4); c.add_argument("--timeout", type=int, default=3600)
    c.add_argument("--no-think", action="store_true", help="关思考: 官方直接写正文(料几乎全是代码), 行里记 think:false, chunks 按非思考模板渲染")
    c.add_argument("--raw", action="store_true", help="裸续写: 题单的 q 是文档开头(不套对话模板), 走 beta/completions; 行里记 raw:true, 写到上限停也算 ok")
    t = sp.add_parser("ptf"); t.add_argument("--ids", required=True); t.add_argument("--raw", required=True); t.add_argument("--out", required=True); t.add_argument("--covered", required=True)
    t.add_argument("--topk", type=int, default=20); t.add_argument("--hf", required=True)
    si = sp.add_parser("simids"); si.add_argument("--rows", required=True); si.add_argument("--hf", required=True); si.add_argument("--out", required=True)
    ms = sp.add_parser("miss"); ms.add_argument("--ids", required=True); ms.add_argument("--raw", required=True); ms.add_argument("--hf", required=True); ms.add_argument("--n", type=int, default=30)
    da = sp.add_parser("diffalign"); da.add_argument("--raw", required=True); da.add_argument("--hf", required=True); da.add_argument("--n", type=int, default=4)
    fx = sp.add_parser("fixtext"); fx.add_argument("--raw", required=True); fx.add_argument("--hf", required=True)
    a = ap.parse_args()
    {"probe-logprobs": cmd_probe_logprobs, "probe-raw": cmd_probe_raw, "probe-align": cmd_probe_align, "probe-echo": cmd_probe_echo, "probe-hfcompare": cmd_probe_hfcompare,
     "collect": cmd_collect, "ptf": cmd_ptf, "diffalign": cmd_diffalign, "fixtext": cmd_fixtext, "miss": cmd_miss, "simids": cmd_simids}[a.cmd](a)


if __name__ == "__main__":
    main()
