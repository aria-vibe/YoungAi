#!/usr/bin/env python3
"""api_branch.py — 分支续写料(2026-10-26 晚): 学生自己想到一半的前缀 + 官方 API 从那里往下续写(带 top-20)= ③ 的靶。

为什么: 段 B 三趟(full2 / v3 / v3g)的料全是官方轨迹 teacher-forced, 学生自己生成到两万位后走进官方从没去过的区域(过度核算 / 绕圈 / 收不了口),
官方轨迹上没有一位告诉它"从这里怎么回来"; 决策位 KL 减半、收口探针 2/2 都量不到这个(fable5 10-26 09:1x)。分支续写 = 曝光偏差的教科书解法:
前缀是学生的(训练器 text 行 "prefix" 字段, 只当上下文、零梯度, core_ptrain_data.c pt_add_text), 靶 100% 是原件自己续写的 token + 它的 top-20。
09-30 铁律禁的是"自吐当靶"; 这里自吐只是提问方式, 靶一位都不是学生的。

API 走 beta/chat/completions 的前缀续写: 最后一条 {"role":"assistant","reasoning_content":学生前缀,"content":"","prefix":true}(content 必须给, 空串就行;
不给 400 "content or tool_calls must be set"), 官方从思考中段接着想, reasoning_content / content 两列分开给 token + bytes + top-20(10-26 晚实测)。
★不走 beta/completions 裸 prompt★: 它把 </think>/EOS 从 text、tokens、text_offset 里全吞掉, stop=["</think>"] 也不停(三次实测), 分界找不到; 按块续写+重采
  找分界又会系统性偏向"接着想"(含 </think> 的块被丢掉重采)—— 正是要修的病的反面。

三步(spark 仓库根, ~/v41env 的 python):
  branch --key K --student freegen.jsonl --out branch_raw.jsonl --hf HF [--every 4096 --max-tokens 8192 --jobs 4]
      student = think_freegen_probe.py 的产物(id q reasoning closed …, 服务默认采样生成的学生思考)。每条在思考第 every·k 个 token 处切一个分支点
      (没收口的条末尾再加一个: 它撞上限时的状态正是最偏的), 官方续写 ≤max_tokens 位。
      输出一行一个分支: id q cut prefix(学生前缀原文) reasoning(官方续的思考) content(收口后的正文, 没收口 "") closed finish lp(与 api_teacher collect 同格式) ok。
  rows   --raw branch_raw.jsonl --out rows.jsonl
      训练行 {"prefix": 头+<think>+学生前缀, "text": 官方续写(思考 [+ </think> + 正文]), "eos": 正文写完才 true}; 配方 restore_B2ctx_*.cfg 直接拼进 data。
      教师表: branch_raw 并进 posttrain_fp_spark.sh 第 6 参的 raw(api_teacher.py ptf 按内容找块, 前缀位 one-hot 占位但训练器不算它的损失)。
  share  --hf HF --raw 名=文件 [名=文件 …] [--doc-tail 8]
      各族料的 token 账: 条数 / 总 token / 中位 / doc_tail 下真进梯度的 token —— 配比按【token】等份, 不按行数(v3 按行看 27/225 是图形, 按 token 是 78%)。
出错会怎样: key 读不到直接退; 单个分支失败重试 5 次后 ok:false(续跑只补 ok:false 的); 文字按 token bytes 拼(api_teacher.text_from_lp 的坑), 有查不到 id 的 token 就记 align_ok false。"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import http.client
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from api_teacher import hf_byte_table, post, text_from_lp  # noqa: E402
from think_trace_collect import ASST, EFFORT_BUDGET, EFFORT_PREFIX, EOS, SYS, THINK, THINK_END, USER, render_messages, user_msgs  # noqa: E402

HEAD = SYS + EFFORT_PREFIX.format(budget=EFFORT_BUDGET["high"])   # 与部署的思考档(high)同一串字节(req_render_ids.sh / server_config.c)
PREFIX_URL = "https://api.deepseek.com/beta/chat/completions"


def stream_prefix(key, model, q, prefix, max_tokens, timeout, content="", messages=None):
    """前缀续写流式拉完, 带 top-20; 返回 (finish, usage, {列: [原始分布条…]})。
    content 非空 = 正文分支(10-30): 学生的完整思考当 reasoning_content、半截正文当 content, 官方从正文中段接着写
    messages 给了 = 真实 agent 请求的整串消息(10-31), 学生前缀接在它后面; 否则单条 user q"""
    body = {"model": model, "max_tokens": max_tokens, "logprobs": True, "top_logprobs": 20, "stream": True, "stream_options": {"include_usage": True},
            "messages": user_msgs(messages or q) + [{"role": "assistant", "reasoning_content": prefix, "content": content, "prefix": True}]}
    ent = {"reasoning_content": [], "content": []}; usage = finish = None
    with post(PREFIX_URL, key, body, timeout) as r:
        for raw in r:
            ln = raw.decode("utf-8", "replace").strip()
            if not ln.startswith("data:") or ln[5:].strip() == "[DONE]":
                continue
            ev = json.loads(ln[5:]); usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                lp = ch.get("logprobs") or {}
                for k in ent: ent[k].extend(lp.get(k) or [])
                finish = ch.get("finish_reason") or finish
    return finish, usage, ent


def branch_one(key, model, item, max_tokens, timeout, b2id, id2b):
    delay = 5
    for attempt in range(5):
        try:
            t0 = time.time()
            if item.get("kind") == "content": finish, usage, ent = stream_prefix(key, model, item["q"], item["prefix_reasoning"], max_tokens, timeout, content=item["prefix"], messages=item.get("messages"))
            else: finish, usage, ent = stream_prefix(key, model, item["q"], item["prefix"], max_tokens, timeout, messages=item.get("messages"))
            lp = {}
            for k in ("reasoning_content", "content"):
                rows = []
                for e in ent[k]:
                    cid = [b2id.get(bytes(t["bytes"] or [])) for t in e.get("top_logprobs") or []]
                    rows.append([b2id.get(bytes(e["bytes"] or [])), e["logprob"], [c for c in cid if c is not None], [t["logprob"] for c, t in zip(cid, e["top_logprobs"]) if c is not None]])
                lp[k] = rows
            rs, cs = text_from_lp(lp["reasoning_content"], id2b), text_from_lp(lp["content"], id2b)
            closed = bool(lp["content"]) or (finish == "stop" and bool(lp["reasoning_content"]) and rs is not None)   # 有正文行 = 过了 </think>; 没正文就 stop = 收口后正文为空(罕见)
            return dict(item, ok=rs is not None and cs is not None, finish=finish, closed=closed, reasoning=rs or "", content=cs or "", lp=lp, usage=usage, model=model,
                        n_lp=len(lp["reasoning_content"]) + len(lp["content"]), secs=round(time.time() - t0, 1))
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError, http.client.HTTPException) as e:   # IncompleteRead 不是 OSError(见 api_teacher.collect_one)
            msg = f"HTTP {e.code} {e.read().decode('utf-8', 'replace')[:200]}" if isinstance(e, urllib.error.HTTPError) else type(e).__name__
            if attempt == 4:
                return dict(item, ok=False, finish=f"error:{msg}", reasoning="", content="", lp={})
        print(f"  [{item['id']}#{item['cut']}] {msg}, {delay}s 后重试", flush=True); time.sleep(delay); delay *= 3


def cut_points(ids, every, closed, cuts=None, end_cut=False):
    """分支点 = 每 every 个 token 一个(不含 0); 没收口的条末尾再加一个(撞上限的状态最偏)。
    cuts(10-27 计划段): 显式切点表, 只切这些位、不补末尾 —— 鹈鹕"像不像"定在思考开头几百位的部件清单里(学生写"橙色喙"处 FP 要"长/大/喉囊"),
    每 8192 位一刀从来切不到那里。
    end_cut(10-31 真实 agent 请求): 收口了的条也在思考末尾切一刀 —— 学生想完了、落笔第一个 token 走岔(aider 里直接写整份文件而不是 SEARCH/REPLACE 块),
    官方从学生想完的地方接: 立刻 </think> 再按格式落笔, 梯度正落在那一位上。"""
    if cuts: return [c for c in cuts if c < len(ids)]
    pts = list(range(every, len(ids), every))
    if (not closed or end_cut) and len(ids) and (not pts or pts[-1] != len(ids)): pts.append(len(ids))
    return pts


def cut_bytes(pieces, cut):
    """前 cut 个 token(字节串)的原文; 分界落在多字节字符中间就往后挪到能解出 UTF-8 的位置(token 边界是字节边界, 不一定是字符边界)。"""
    while cut < len(pieces):
        try: return b"".join(pieces[:cut]).decode("utf-8"), cut
        except UnicodeDecodeError: cut += 1
    return b"".join(pieces).decode("utf-8", "replace"), len(pieces)


def prefix_text(ids, cut, id2b): return cut_bytes([id2b[i] for i in ids], cut)


def cap_target(pieces, head, tail, eos):
    """靶太长只留头尾, 返回 [(并进前缀的中段字, 靶字, eos)](10-30 用户: d003→d005 料翻倍、一轮 3 小时, 每轮收益没多一格; 累计训练靠多轮新料不靠长尾)。
    头 = 切点后前 head 位(学生刚分叉, 官方该怎么接落在这); 尾 = 末 tail 位(</think> / EOS 决策位在这), 没收口的尾只是半路文字, 不留;
    中段并进 prefix 只当上下文(训练器 pt_add_text 见 prefix 不出梯度, 只前向走一遍)。d006 实账: 官方整轨迹 + 收口位分支的长尾占靶字 76%。"""
    n = len(pieces)
    if n <= head + tail: return [("", cut_bytes(pieces, n)[0], eos)]
    out = [("", cut_bytes(pieces, head)[0], False)]
    mid, mc = cut_bytes(pieces, n - tail)
    if eos and mc < n: out.append((mid, b"".join(pieces[mc:]).decode("utf-8"), True))
    return out


def msgs_of(r): return {"messages": r["messages"]} if r.get("messages") else {}   # 真实 agent 请求的消息串跟着行走(学生条 → 分支项 → 训练行的前缀)


def head_of(r): return render_messages(r["messages"]) if r.get("messages") else HEAD + USER + r["q"] + ASST + THINK   # 训练行前缀的头: 多条消息按服务端渲染, 否则老写法


def cmd_branch(a):
    key = open(a.key, encoding="utf-8").read().strip(); tok, b2id = hf_byte_table(a.hf); id2b = {i: b for b, i in b2id.items()}
    items = []
    for ln in open(a.student, encoding="utf-8"):
        if not ln.strip(): continue
        r = json.loads(ln)
        if not r.get("reasoning"): continue
        if a.content_fracs:   # 正文分支(10-30 用户"质感和审美语料"): 学生收口且有正文的条, 正文按 token 数在这些比例处各切一刀, 官方接着把正文写完
            # 为什么: 学生写 <circle> 当头、坐标粗、没有渐变的地方, 官方整轨迹里没有一位梯度落在那里(曝光偏差, 与规划位同病); 分支把官方的坐标 / 渐变 / 描边教在学生自己的前文上
            if not r.get("closed") or not r.get("content"): continue
            cids = tok.encode(r["content"], add_special_tokens=False).ids
            for fr in (float(x) for x in a.content_fracs.split(",")):
                cut = int(len(cids) * fr)
                if cut < 8 or cut >= len(cids) - 8: continue
                pfx, cut2 = prefix_text(cids, cut, id2b)
                items.append({"id": r["id"], "q": r["q"], "cut": cut2, "kind": "content", "student_tokens": len(cids), "student_closed": True, "prefix_reasoning": r["reasoning"], "prefix": pfx, "src": "cbranch", **msgs_of(r)})
            continue
        ids = tok.encode(r["reasoning"], add_special_tokens=False).ids
        for cut in cut_points(ids, a.every, bool(r.get("closed")), [int(x) for x in a.cuts.split(",")] if a.cuts else None, a.end_cut):
            pfx, cut2 = prefix_text(ids, cut, id2b)
            items.append({"id": r["id"], "q": r["q"], "cut": cut2, "student_tokens": len(ids), "student_closed": bool(r.get("closed")), "prefix": pfx, "src": "branch", **msgs_of(r)})
    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out, encoding="utf-8"):
            r = json.loads(ln)
            if r.get("ok"): done.add((r["id"], r["cut"]))
    todo = [it for it in items if (it["id"], it["cut"]) not in done]
    print(f"branch: 学生条 {sum(1 for _ in open(a.student))}, 分支点 {len(items)}(每 {a.every} 位一个), 已有 ok {len(done)}, 本次 {len(todo)}, {a.jobs} 路, 续写上限 {a.max_tokens}", flush=True)
    lock = threading.Lock(); t0 = time.time(); nok = ncl = nfin = 0
    with open(a.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(branch_one, key, a.model, it, a.max_tokens, a.timeout, b2id, id2b) for it in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result(); nok += bool(r["ok"]); ncl += bool(r.get("closed")); nfin += r.get("finish") == "stop" and bool(r.get("content"))
            with lock: f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            print(f"  {i}/{len(todo)} [{r['id']}#{r['cut']}] {'ok' if r['ok'] else '★' + str(r['finish'])} 续 {r.get('n_lp', 0)} tok"
                  f"{' 收口' if r.get('closed') else ''}{' 正文写完' if r.get('finish') == 'stop' and r.get('content') else ''} {r.get('secs', '?')} s, 累计 {time.time() - t0:.0f} s", flush=True)
    print(f"branch 完成: ok {nok}/{len(todo)}, 其中收口 {ncl}, 正文写完 {nfin} → {a.out}", flush=True)


def cmd_rows(a):
    """轨迹 → 训练行(prefix 只当上下文, text 是靶)。吃三种原料: 官方整轨迹(collect, 没有 prefix 字段 = 从头)、思考段分支、正文分支(kind=content)。
    --cap 头,尾(要 --hf): 靶超过 头+尾 位就按 cap_target 拆成头行 + 尾行; lp 里有不认识的 id 或拼不回原文的条不拆(计数报出来)。"""
    n = ne = nsplit = nnocap = ntr = 0; id2b = tf = None
    if a.cap:
        head, tail = map(int, a.cap.split(",")); _, b2id = hf_byte_table(a.hf); id2b = {i: b for b, i in b2id.items()}
        assert a.traces, "--cap 要 --traces(截头轨迹落这里, 拼进配教师表的原料)"; tf = open(a.traces, "w", encoding="utf-8")

    def pieces_of(lp, text, with_think, closed):   # 靶的逐 token 字节串; 拼不回 text 就不拆
        if id2b is None: return None
        try:
            p = [id2b[x[0]] for x in lp.get("reasoning_content", [])] if with_think else []
            if with_think and closed: p.append(THINK_END.encode())
            if closed or not with_think: p += [id2b[x[0]] for x in lp.get("content", [])]
        except KeyError: return None
        return p if cut_bytes(p, len(p))[0] == text else None

    with open(a.out, "w", encoding="utf-8") as f:
        for ln in open(a.raw, encoding="utf-8"):
            r = json.loads(ln)
            if not r.get("ok") or not r.get("lp"): continue
            if r.get("kind") == "content":   # 正文分支: 前缀 = 头 + 题 + <think>学生完整思考</think> + 学生半截正文, 靶 = 官方接着写的正文(写完才带 EOS)
                text = r["content"]
                if not text or r.get("reasoning"): continue   # 官方从正文中段接笔不该再有思考; 有就是 API 行为变了, 整条丢
                eos = r.get("finish") == "stop"
                prefix = head_of(r) + r["prefix_reasoning"] + THINK_END + r["prefix"]
                assert EOS not in text and THINK_END not in text, (r["id"], r["cut"])
                pieces = pieces_of(r["lp"], text, False, True)
            else:
                closed = r.get("closed", bool(r.get("content")))   # 官方整轨迹没有 closed 字段: 写了正文就是收过口
                text = r["reasoning"] + (THINK_END + r["content"] if closed else "")
                if not text: continue
                eos = closed and r.get("finish") == "stop"   # 正文写完 API 才 stop; 撞上限的正文 / 没收口的思考末尾都不补 EOS
                prefix = head_of(r) + r.get("prefix", "")
                assert EOS not in text and text.count(THINK_END) <= 1, (r["id"], r.get("cut"))
                pieces = pieces_of(r["lp"], text, True, closed)
            if id2b is not None and pieces is None: nnocap += 1
            parts = cap_target(pieces, head, tail, eos) if pieces is not None else [("", text, eos)]
            nsplit += len(parts) > 1
            for mid, t, e in parts:
                row = {"prefix": prefix + mid, "text": t}
                if not e: row["eos"] = False
                f.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1; ne += e
            if tf is not None and len(parts[0][1]) < len(text):   # 头被截: 配教师表要一条同样截短的轨迹(api_teacher.py cmd_ptf 按整块找、一条轨迹只认领一条序列;
                lp = r["lp"]; hc = cut_bytes(pieces, head)[1]; nr = len(lp.get("reasoning_content", []))   # 尾行连着前缀里的中段整块都在, 对原轨迹; 头行对这条)
                if r.get("kind") == "content": tr = {"reasoning": "", "content": parts[0][1], "lp": {"reasoning_content": [], "content": lp["content"][:hc]}}
                elif hc <= nr: tr = {"reasoning": parts[0][1], "content": "", "lp": {"reasoning_content": lp["reasoning_content"][:hc], "content": []}}
                else: tr = {"reasoning": r["reasoning"], "content": parts[0][1][len(r["reasoning"]) + len(THINK_END):], "lp": {"reasoning_content": lp["reasoning_content"], "content": lp["content"][:hc - nr - 1]}}
                tf.write(json.dumps({"id": f"{r['id']}#cap{head}", "q": r["q"], "src": r.get("src", "api"), "ok": True, "head": True, "finish": "length", **tr}, ensure_ascii=False) + "\n"); ntr += 1
    if tf is not None: tf.close()
    print(f"rows: {n} 行 → {a.out}(带 EOS 的 {ne}" + (f"; 靶只留头 {head} 尾 {tail} 位: 拆了 {nsplit} 条, 拼不回原文没拆 {nnocap} 条, 截头轨迹 {ntr} 条 → {a.traces}" if id2b else "")
          + "); 训练器见 prefix 字段把它只当上下文(core_ptrain_data.c pt_add_text)")


def cmd_head(a):
    """官方轨迹只留思考的前 n 个 token(10-27 计划段料): 文字按 API 的 token 字节拼回、lp 同步截, content 清空, 标 head。
    为什么: doc_tail 只训每篇末几段, 思考开头的部件清单(决定画成什么样)从没进过梯度; 截出来的头单独成一篇, 训练器整篇都训。
    ok 留 true 是给 api_teacher.py ptf 认(它只配 ok 的轨迹, content 空就只配思考块); chunks 见 head 不接 </think>、不补 EOS。
    --raw 文件:N —— N>0 从该文件等距取 N 条(不挑内容), 0 = 全取。lp 里有不认识的 id 的条整条跳过(拼不回原文)。
    不看原条的 ok(10-30 d003 实撞: 12 条规划位题里 1 条正文撞 65536 上限 finish=length, ok=false 被跳过, 其实思考前 2048 位和 lp 都完整):
    头只用思考开头, 原条收没收口无关; 只排除 API 报错的条(finish=error:*, lp 为空)。"""
    _, b2id = hf_byte_table(a.hf); id2b = {i: b for b, i in b2id.items()}; n = skip = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for spec in a.raw:
            path, k = (spec.rsplit(":", 1) + ["0"])[:2] if ":" in spec else (spec, "0")
            R = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            R = [r for r in R if not str(r.get("finish", "")).startswith("error") and r.get("lp") and r["lp"].get("reasoning_content")]
            if int(k) > 0: R = R[::max(1, len(R) // int(k))][:int(k)]
            for r in R:
                rows = r["lp"]["reasoning_content"]; ids = [x[0] for x in rows]
                if any(i not in id2b for i in ids[:a.n + 8]): skip += 1; continue
                txt, cut = prefix_text(ids, min(a.n, len(ids)), id2b)
                f.write(json.dumps({"id": f"{r['id']}#h{a.n}", "q": r["q"], "src": r.get("src", "api"), "reasoning": txt, "content": "", "ok": True, "head": True,
                                    "finish": "length", "lp": {"reasoning_content": rows[:cut], "content": []}}, ensure_ascii=False) + "\n"); n += 1
    print(f"head: {n} 条 × 思考前 {a.n} 位 → {a.out}(跳过 {skip} 条: lp 里有不认识的 id)")


def cmd_evalcut(a):
    """留出集官方轨迹 → 评估用的两段(10-28 自进化): 每条出 id#h(思考前 --head 位, 同 cmd_head)与 id#t(思考末 --tail 位 + </think> + 正文前 --content 位)。
    为什么: 整条轨迹当训练器 eval= 时, 18 篇 27 万位 + ctxlen 前缀预填, 第 0 步评估一次 ~18 分钟, 一轮装不下(10-28 第一轮轮末评估撞 40 分钟训练超时, ckpt 没落盘)。
    两段各管一件: 开头 = 计划(画成什么样定在这), 末尾 = 决策位(</think>)与终止位(正文写得下就带 EOS); 中段细节由编程判决份的置信度分桶量。
    lp 按 token 同步截, api_teacher.py ptf 照常按块配 —— 不需要本地 FP 教师。正文被截的条标 content_cut, chunks 不给它补 EOS(补在正文中间 = 教它写到一半就停)。"""
    _, b2id = hf_byte_table(a.hf); id2b = {i: b for b, i in b2id.items()}; n = skip = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for r in (json.loads(l) for l in open(a.raw, encoding="utf-8") if l.strip()):
            if not (r.get("ok") and r.get("lp")): continue
            rr, cc = r["lp"]["reasoning_content"], r["lp"]["content"]
            if any(x[0] not in id2b for x in rr + cc): skip += 1; continue
            txt, cut = prefix_text([x[0] for x in rr], min(a.head, len(rr)), id2b)
            f.write(json.dumps({"id": f"{r['id']}#h{a.head}", "q": r["q"], "src": r.get("src", "api"), "reasoning": txt, "content": "", "ok": True, "head": True,
                                "finish": "length", "lp": {"reasoning_content": rr[:cut], "content": []}}, ensure_ascii=False) + "\n")
            s0 = max(0, len(rr) - a.tail)
            while s0 < len(rr):   # 起点落在多字节字符中间就往后挪(token 边界是字节边界, 不一定是字符边界)
                try: rt = b"".join(id2b[x[0]] for x in rr[s0:]).decode("utf-8"); break
                except UnicodeDecodeError: s0 += 1
            ct, ccut = prefix_text([x[0] for x in cc], min(a.content, len(cc)), id2b) if cc else ("", 0)
            f.write(json.dumps({"id": f"{r['id']}#t{a.tail}", "q": r["q"], "src": r.get("src", "api"), "reasoning": rt, "content": ct, "ok": True,
                                "finish": r.get("finish"), "content_cut": ccut < len(cc), "lp": {"reasoning_content": rr[s0:], "content": cc[:ccut]}}, ensure_ascii=False) + "\n"); n += 1
    print(f"evalcut: {n} 条 × (思考前 {a.head} + 思考末 {a.tail} / 正文前 {a.content}) → {a.out}(跳过 {skip} 条: lp 里有不认识的 id)")


def cmd_share(a):
    tok, _ = hf_byte_table(a.hf); S = 1022
    print(f"| 族 | 条 | 总 token | 中位 | 最长 | doc_tail={a.doc_tail} 进梯度 token |")
    print("|---|---|---|---|---|---|")
    tot = {}
    for spec in a.raw:
        name, path = spec.split("=", 1); lens = []
        for ln in open(path, encoding="utf-8"):
            if not ln.strip(): continue
            r = json.loads(ln)
            t = r.get("text") or ((r.get("reasoning") or "") + (r.get("content") or ""))
            if not t: continue
            lens.append(len(tok.encode(t, add_special_tokens=False).ids))
        lens.sort(); tr = sum(min(L, a.doc_tail * S) if a.doc_tail else L for L in lens)
        tot[name] = (len(lens), sum(lens), tr)
        print(f"| {name} | {len(lens)} | {sum(lens)} | {lens[len(lens) // 2] if lens else 0} | {lens[-1] if lens else 0} | {tr} |")
    T = sum(v[1] for v in tot.values()) or 1; G = sum(v[2] for v in tot.values()) or 1
    print("份额(总 token / 进梯度): " + ", ".join(f"{k} {100 * v[1] / T:.0f}% / {100 * v[2] / G:.0f}%" for k, v in tot.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("branch"); b.add_argument("--key", required=True); b.add_argument("--student", required=True); b.add_argument("--out", required=True); b.add_argument("--hf", required=True)
    b.add_argument("--model", default="deepseek-flash"); b.add_argument("--every", type=int, default=4096); b.add_argument("--max-tokens", type=int, default=8192)
    b.add_argument("--jobs", type=int, default=4); b.add_argument("--timeout", type=int, default=3600); b.add_argument("--cuts", default="")
    b.add_argument("--content-fracs", default="", help="正文分支: 学生正文按 token 数在这些比例处切(如 0.25,0.5,0.75), 官方接着写完; 给了就不做思考段分支")
    b.add_argument("--end-cut", action="store_true", help="收口了的学生条也在思考末尾切一刀(落笔第一个 token 走岔的病, 10-31 aider)")
    r = sp.add_parser("rows"); r.add_argument("--raw", required=True); r.add_argument("--out", required=True); r.add_argument("--hf", default="")
    r.add_argument("--cap", default="", help="头,尾(token 数): 靶超过头+尾就只留头 head 位 + 末 tail 位(带 EOS 的才留尾), 中段并进 prefix 只当上下文; 要 --hf 与 --traces")
    r.add_argument("--traces", default="", help="--cap 截了头的行各配一条截短轨迹写到这里(拼进 ptf 的原料, 否则头行配不上教师表被剔; 10-30 d006 实撞 17 条)")
    s = sp.add_parser("share"); s.add_argument("--hf", required=True); s.add_argument("--raw", nargs="+", required=True); s.add_argument("--doc-tail", type=int, default=8)
    h = sp.add_parser("head"); h.add_argument("--hf", required=True); h.add_argument("--raw", nargs="+", required=True); h.add_argument("--out", required=True); h.add_argument("--n", type=int, default=2048)
    e = sp.add_parser("evalcut"); e.add_argument("--hf", required=True); e.add_argument("--raw", required=True); e.add_argument("--out", required=True)
    e.add_argument("--head", type=int, default=1024); e.add_argument("--tail", type=int, default=512); e.add_argument("--content", type=int, default=1536)
    a = ap.parse_args()
    {"branch": cmd_branch, "rows": cmd_rows, "share": cmd_share, "head": cmd_head, "evalcut": cmd_evalcut}[a.cmd](a)


if __name__ == "__main__":
    main()
