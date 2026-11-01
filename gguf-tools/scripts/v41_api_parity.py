#!/usr/bin/env python3
"""v41_api_parity.py — 拿官方 API 的逐位 top-20 logprob 当外部真值, 给 FP 教师 / file 教师 / 引擎 打分(2026-10-25)。

为什么要它: FP 教师(dq_fp4 精确专家)对引擎的 y 逐层只是"VQ 噪声"量级, 看不出单层 bug, 但它抄不了长文、前 64 位就有
15~19 nat 的尖峰。到底是 FP 前向错, 还是引擎碰巧对? 只有官方 API 能裁。协议照 tests/test-vectors(贪心 + top_logprobs 20,
thinking 关): API 贪心续写 N 位 → 把 (渲染好的提示 ids + 续写 ids) 喂教师/引擎做 teacher-forcing → 第 k 步的分布与 API 第 k 步比。

子命令(都不碰数值链, 只做编排/比对):
  fetch   --key FILE --prompt TEXT --out JSON [--model deepseek-flash] [--max-tokens 96]
  ids     --json JSON --prompt-ids FILE --hf HFDIR --out IDS      # FILE = `ds4 --dump-tokens -p` 的首行 [id, …]
  compare --json JSON --ids IDS --nprompt N --hf HFDIR name=logits.bin ...   # logits = 原始 f32 [S][V]
密钥只读文件、只进请求头, 任何路径都不打印。"""
import argparse, json, sys, urllib.request, urllib.error
from pathlib import Path

API_URL = "https://api.deepseek.com/chat/completions"


def load_tok(hf):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(hf, trust_remote_code=True)


def tok_to_id(tok, text, bts):
    """API 给的是解码后的文本(+字节), 词表里一个 piece 对应一个 id; 先按文本 encode, 不是单 id 再按字节映射成 piece 查。"""
    if text is None:                  # top_logprobs 里有的条只有 bytes 没有 token 文本(半个 UTF-8 字符)
        text = bytes(bts).decode("utf-8", "replace") if bts else ""
    enc = tok.encode(text, add_special_tokens=False) if text else []
    if len(enc) == 1:
        return enc[0]
    if bts:
        # GPT-2 式字节→可见字符映射(与 tokenizers 的 ByteLevel 同表), 自己写一份: transformers 的导出路径随版本变
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]; n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b); cs.append(256 + n); n += 1
        b2u = dict(zip(bs, [chr(c) for c in cs]))
        piece = "".join(b2u[b] for b in bts)
        i = tok.convert_tokens_to_ids(piece)
        if i is not None and i != tok.unk_token_id:
            return i
    return None


def cmd_fetch(a):
    key = Path(a.key).read_text().strip()
    body = {"model": a.model, "messages": [{"role": "user", "content": a.prompt}], "temperature": 0,
            "max_tokens": a.max_tokens, "logprobs": True, "top_logprobs": 20, "thinking": {"type": "disabled"}}
    req = urllib.request.Request(API_URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"★API HTTP {e.code}: {e.read()[:300]!r}★")
    steps = resp["choices"][0]["logprobs"]["content"]
    Path(a.out).write_text(json.dumps({"request": body, "response": resp}, ensure_ascii=False, indent=1))
    print(f"[fetch] model={resp.get('model')} 续写 {len(steps)} 位, finish={resp['choices'][0].get('finish_reason')}")
    print("[fetch] 文本:", repr(resp["choices"][0]["message"]["content"][:300]))


def cmd_ids(a):
    tok = load_tok(a.hf)
    j = json.load(open(a.json))
    steps = j["response"]["choices"][0]["logprobs"]["content"]
    first = Path(a.prompt_ids).read_text().splitlines()[0]
    pids = [int(t) for t in first.strip("[] \n").split(",") if t.strip()]
    if not pids or pids[0] != tok.bos_token_id:
        sys.exit(f"★提示 ids 没以 BOS({tok.bos_token_id}) 开头: {pids[:5]} —— ds4 --dump-tokens 要喂渲染好的文本(以 BOS 开头)★")
    cont, bad = [], 0
    for it in steps:
        i = tok_to_id(tok, it["token"], it.get("bytes"))
        if i is None:
            bad += 1; print(f"[ids] ★映射不到: {it['token']!r} bytes={it.get('bytes')}★"); i = 0
        cont.append(i)
    text_api = "".join(it["token"] for it in steps)
    text_dec = tok.decode(cont)
    print(f"[ids] 提示 {len(pids)} + 续写 {len(cont)} = {len(pids)+len(cont)}; 映射失败 {bad}; 解码回文 {'同' if text_dec == text_api else '★不同★'}")
    if text_dec != text_api:
        print("  api:", repr(text_api[:200])); print("  dec:", repr(text_dec[:200]))
    Path(a.out).write_text(" ".join(str(i) for i in pids + cont) + "\n")
    print(f"[ids] nprompt={len(pids)} → {a.out}")


def cmd_compare(a):
    import numpy as np
    tok = load_tok(a.hf)
    j = json.load(open(a.json)); steps = j["response"]["choices"][0]["logprobs"]["content"]
    ids = [int(t) for t in Path(a.ids).read_text().split()]
    S = len(ids); npm = a.nprompt
    models = {}
    for spec in a.logits:
        nm, p = spec.split("=", 1)
        hd = np.fromfile(p, np.uint32, count=2)          # --score-out / 教师 --out 同一格式: 8 字节头 <S,V> + f32 [S][V](anchor_metrics read_student)
        Sf, V = int(hd[0]), int(hd[1])
        assert Sf == S, f"{nm}: 头里 S={Sf} ≠ ids 数 {S}"
        lg = np.fromfile(p, np.float32, offset=8).reshape(S, V)
        m = lg.max(1, keepdims=True); lp = lg - m - np.log(np.exp(lg - m).sum(1, keepdims=True))
        models[nm] = lp
    names = list(models)
    print(f"S={S} nprompt={npm} 步数={len(steps)} 模型={names}")
    print("步 | API 贪心 token      lp_api | " + " | ".join(f"{nm}: argmax(同?) lp_tok" for nm in names))
    stat = {nm: {"same": 0, "nll": 0.0, "dlp": [], "n": 0} for nm in names}
    for k, it in enumerate(steps):
        row = npm + k - 1          # 第 row 位的 logits 预测 ids[row+1] = 第 k 步的 token
        if npm + k >= S:
            break
        tid = ids[npm + k]
        line = f"{k:3d} | {it['token']!r:18} {it['logprob']:7.3f} |"
        # API 的 top_logprobs 里 -9999 是占位(没给值), 和 tests/test-vectors 的 fetch 一样剔掉
        tops = [(tok_to_id(tok, t["token"], t.get("bytes")), t["logprob"]) for t in it["top_logprobs"][:5] if t["logprob"] > -1000]
        for nm in names:
            lp = models[nm][row]; am = int(lp.argmax()); same = am == tid
            st = stat[nm]; st["n"] += 1; st["same"] += same; st["nll"] += -lp[tid]
            st["dlp"] += [abs(lp[i] - l) for i, l in tops if i is not None]
            line += f" {tok.decode([am])!r:14}({'✓' if same else '✗'}) {lp[tid]:7.3f} |"
        print(line)
    print("\n汇总(对官方 API):")
    for nm in names:
        st = stat[nm]; d = np.array(st["dlp"])
        print(f"  {nm:8} Same top {100*st['same']/st['n']:5.1f}%  API贪心token平均NLL {st['nll']/st['n']:.3f}  "
              f"API top-5 |Δlogprob| 均值 {d.mean():.3f} 中位 {np.median(d):.3f} p95 {np.percentile(d,95):.3f}")


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sp = ap.add_subparsers(dest="cmd", required=True)
f = sp.add_parser("fetch"); f.add_argument("--key", required=True); f.add_argument("--prompt", required=True)
f.add_argument("--out", required=True); f.add_argument("--model", default="deepseek-flash"); f.add_argument("--max-tokens", type=int, default=96)
i = sp.add_parser("ids"); i.add_argument("--json", required=True); i.add_argument("--prompt-ids", required=True)
i.add_argument("--hf", required=True); i.add_argument("--out", required=True)
c = sp.add_parser("compare"); c.add_argument("--json", required=True); c.add_argument("--ids", required=True)
c.add_argument("--nprompt", type=int, required=True); c.add_argument("--hf", required=True); c.add_argument("logits", nargs="+")
args = ap.parse_args()
{"fetch": cmd_fetch, "ids": cmd_ids, "compare": cmd_compare}[args.cmd](args)
