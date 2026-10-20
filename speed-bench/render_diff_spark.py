#!/usr/bin/env python3
"""render_diff_spark.py — 官方 encoding.py 渲染 vs 我们服务端渲染的逐 token 对拍(2026-10-20, 用户"再看看原模型仓库")。

怎么用(spark 仓库根):
  ~/v41env/bin/python speed-bench/render_diff_spark.py hf/DeepSeek-V4.1-Flash-1017 /tmp/bugmd/r3_req.json /tmp/bugmd/r3_prompt.ids
    <hf 目录>      出厂权重目录: 用它的 encoding/encoding.py 与 tokenizer.json
    <请求 JSON>    trace 里原样记的 OpenAI 格式请求(messages + 顶层 tools + think)
    <我们的 ids>   trace "--- token ids" 块里 prompt: 后那一行(空格分隔)

做什么: 把请求按官方口径喂 encode_messages(顶层 tools 挂到 system 消息上 —— 官方 render_message 从 msg["tools"] 取; think:true ⇒
  thinking_mode="thinking", reasoning_effort 不传 = 官方默认 "high"=75), 官方 tokenizer.json 分词, 与我们的 id 逐位比, 打印第一处分叉
  前后各 12 个 token 的两边原文, 以及两边总长。全同 ⇒ 模板不是问题; 不同 ⇒ 那一处就是要修的 bug。
为什么要它: 10-18 对官方 1017 的模板只是读过代码, 从没对这条真实请求(39 个工具 + 系统提示)逐 token 对过。
出错会怎样: encoding.py 导入失败多半是 hf 目录给错; tokenizers 缺就装 ~/v41env(0.23 已在)。
"""
import importlib.util
import json
import os
import sys


def main():
    hf, req_path, ids_path = sys.argv[1], sys.argv[2], sys.argv[3]
    spec = importlib.util.spec_from_file_location("encoding", os.path.join(hf, "encoding", "encoding.py"))
    enc = importlib.util.module_from_spec(spec); spec.loader.exec_module(enc)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(hf, "tokenizer.json"))

    d = json.load(open(req_path))
    msgs = [dict(m) for m in d["messages"]]
    tools = d.get("tools")
    if tools:
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["tools"] = tools
        else:
            msgs.insert(0, {"role": "system", "content": "", "tools": tools})
    mode = "thinking" if d.get("think") else "chat"
    effort = d.get("reasoning_effort")
    prompt = enc.encode_messages(msgs, thinking_mode=mode, reasoning_effort=effort)
    ref = tok.encode(prompt, add_special_tokens=False).ids
    ours = [int(x) for x in open(ids_path).read().split()]
    print("官方渲染 %d 字符 → %d token; 我们 %d token; thinking_mode=%s effort=%s" % (len(prompt), len(ref), len(ours), mode, effort or "默认(75)"))
    n = min(len(ref), len(ours))
    k = next((i for i in range(n) if ref[i] != ours[i]), None)
    if k is None and len(ref) == len(ours):
        print("★逐 token 全同★")
        return 0
    if k is None:
        k = n
        print("★前 %d 位全同, 只差长度: 官方多 %d / 我们多 %d★" % (n, len(ref) - n, len(ours) - n))
    else:
        print("★第 %d 位分叉: 官方 %d(%r) vs 我们 %d(%r)★" % (k, ref[k], tok.decode([ref[k]]), ours[k], tok.decode([ours[k]])))
    a, b = max(0, k - 12), k + 12
    print("官方 [%d:%d]: %r" % (a, b, tok.decode(ref[a:b])))
    print("我们 [%d:%d]: %r" % (a, b, tok.decode(ours[a:b])))
    # 分叉之后还有没有再对上(只是插入/缺一段): 往后找我们的 20 个 token 在官方里下一次出现的位置
    win = ours[k + 1:k + 21]
    if win:
        for j in range(k, len(ref) - len(win)):
            if ref[j:j + len(win)] == win:
                print("我们第 %d 位起的 20 个 token 在官方第 %d 位再次对上 ⇒ 官方比我们多/少 %d 个 token 的一段" % (k + 1, j, j - (k + 1)))
                break
        else:
            print("分叉后 20 个 token 的窗在官方后文里找不到 ⇒ 不是单段插入/缺失")
    print("官方渲染全文 → /tmp/bugmd/render_official.txt")
    open("/tmp/bugmd/render_official.txt", "w").write(prompt)
    # 全文对齐: 两边解码回文本是否全同; 同则按字符边界对齐, 统计"分词边界不同"的段与最常见的组合
    tr, to = tok.decode(ref), tok.decode(ours)
    print("解码回文本: %s(官方 %d 字符 / 我们 %d 字符)" % ("全同" if tr == to else "★不同★", len(tr), len(to)))
    if tr != to:
        k2 = next((i for i in range(min(len(tr), len(to))) if tr[i] != to[i]), min(len(tr), len(to)))
        print("文本第 %d 字符分叉: 官方 %r / 我们 %r" % (k2, tr[max(0, k2 - 40):k2 + 40], to[max(0, k2 - 40):k2 + 40]))
        return 1
    def bounds(ids):
        b, off = [], 0
        for t in ids:
            off += len(tok.decode([t])); b.append(off)
        return b
    br, bo = bounds(ref), bounds(ours)
    common = sorted(set(br) & set(bo))
    from collections import Counter
    pat = Counter(); nseg = 0
    prev = 0; ir = io = 0
    for c in common:
        jr = ir
        while jr < len(ref) and br[jr] <= c: jr += 1
        jo = io
        while jo < len(ours) and bo[jo] <= c: jo += 1
        if ref[ir:jr] != ours[io:jo]:
            nseg += 1
            pat[(tok.decode(ref[ir:jr]), tuple(tok.decode([t]) for t in ours[io:jo]))] += 1
        ir, io = jr, jo
    print("分词边界不同的段: %d 处(共同边界 %d); 最常见 12 种(官方切法 → 我们的切法):" % (nseg, len(common)))
    for (a, b), n in pat.most_common(12):
        print("  %4d × %r → %r" % (n, a, list(b)))
    print("末尾 24 token 官方: %r" % tok.decode(ref[-24:]))
    print("末尾 24 token 我们: %r" % tok.decode(ours[-24:]))
    print("末尾 id 官方 %s / 我们 %s" % (ref[-8:], ours[-8:]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
