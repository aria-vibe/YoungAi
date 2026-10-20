#!/usr/bin/env python3
"""unicode_pretok_gen.py — 生成 BPE 预切分要的 Unicode 类别表 + 预切分金标用例(2026-10-20)。

为什么: 官方 tokenizer.json 的预切分正则用的是 Unicode 类别(\\p{L} \\p{M} \\p{N} \\p{P} \\p{S} 与 \\s),
引擎 core_bpe.c 原来只认 ASCII 标点/数字, 把一切非 ASCII 当字母 —— 中文标点("。\\n" ")。" "」\\n")被切开,
一条 11k 的中文系统提示有 27 处 token 与官方不同(10-20 实测, speed-bench/render_diff_spark.py)。
产出(两份都是生成件, 别手改):
  src/common/ds4_unicode_tab.h   四张区间表: 字母 L / 组合标记 M / 数字 N / 标点+符号 PS(来自跑它的 Python 的 unicodedata)
  tests/unit/pretok_cases.h      一组字符串在官方 tokenizers 预切分器(tokenizer.json 的 pre_tokenizer)下的切片边界, ds4_unit 逐条对拍
用法(spark, v41env 有 tokenizers): ~/v41env/bin/python gguf-tools/scripts/unicode_pretok_gen.py hf/DeepSeek-V4.1-Flash-1017
出错会怎样: 没有 tokenizers 只能出类别表, 用例表跳过并提示。
"""
import os
import sys
import unicodedata

CASES = [
    ")。\n", "。\n\n", "朋友。\n\n## 你怎么做事", "说\"不\"。", "「好」\n", "a)、b", "本机)。简洁、直接、像靠谱的朋友。\n\n",
    "    int x = 1;\n", "abc123def 12345 １２３４", "héllo wörld", "a → b ★ c", "价格：¥100，约 3.5%。",
    "😀 emoji\n", "x y　z", "  \n\n  foo", "tab\tsep\n", "I'm don't", "é combining", "...\n>;\n",
    "中文，English。混排！\n", "《书名》【标注】（括号）", "a\r\nb\r\n", "\n\n", " ", "", "ĉ control", "—— 破折号 …… 省略",
]


def ranges(pred):
    out, start = [], None
    for cp in range(0x110000):
        ok = pred(cp)
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append((start, cp - 1)); start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def cat(cp):
    return unicodedata.category(chr(cp))


def emit_table(f, name, rs):
    f.write("static const uint32_t %s[][2] = {\n" % name)
    for i in range(0, len(rs), 6):
        f.write("    " + " ".join("{0x%X,0x%X}," % r for r in rs[i:i + 6]) + "\n")
    f.write("};\n")


def main():
    hf = sys.argv[1] if len(sys.argv) > 1 else None
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tabs = {
        "uc_tab_L": ranges(lambda c: cat(c)[0] == "L"),
        "uc_tab_M": ranges(lambda c: cat(c)[0] == "M"),
        "uc_tab_N": ranges(lambda c: cat(c)[0] == "N"),
        "uc_tab_PS": ranges(lambda c: cat(c)[0] in "PS"),
    }
    p = os.path.join(root, "src/common/ds4_unicode_tab.h")
    with open(p, "w") as f:
        f.write("/* ds4_unicode_tab.h — 生成件(gguf-tools/scripts/unicode_pretok_gen.py, Unicode %s), 别手改。\n"
                " * 每表 [start, end] 闭区间升序; ds4_pretok.c 二分查。*/\n#include <stdint.h>\n" % unicodedata.unidata_version)
        for k, v in tabs.items():
            emit_table(f, k, v)
    print("类别表 → %s: %s" % (p, {k: len(v) for k, v in tabs.items()}))
    if not hf:
        print("没给 hf 目录, 用例表跳过"); return
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("没有 tokenizers, 用例表跳过"); return
    tok = Tokenizer.from_file(os.path.join(hf, "tokenizer.json"))
    q = os.path.join(root, "tests/unit/pretok_cases.h")
    with open(q, "w") as f:
        f.write("/* pretok_cases.h — 生成件(gguf-tools/scripts/unicode_pretok_gen.py): 官方 tokenizer.json 预切分器的切片边界, 别手改。\n"
                " * 每条 = 原文 + 各片字节长度(0 结尾)。*/\n")
        f.write("static const struct { const char *text; int lens[64]; } pretok_cases[] = {\n")
        for s in CASES:
            pieces = tok.pre_tokenizer.pre_tokenize_str(s)
            lens, prev = [], 0
            raw = s.encode("utf-8")
            by_char = all(b <= len(s) for _, (_, b) in pieces) and (not pieces or pieces[-1][1][1] == len(s))
            for _, (a, b) in pieces:   # 偏移按字符(tokenizers 默认)或字节, 两种都换算成字节长度
                assert a == prev, (s, pieces); prev = b
                lens.append(len(s[a:b].encode("utf-8")) if by_char else b - a)
            assert prev == (len(s) if by_char else len(raw)), (s, pieces)
            assert sum(lens) == len(raw), (s, pieces, lens)
            # 八进制转义定长 3 位, 不像 \x 会把后面的十六进制字符吞进去
            lit = "".join("\\%03o" % c if c < 0x20 or c == 0x22 or c == 0x5c or c >= 0x7f else chr(c) for c in raw)
            f.write('    {"%s", {%s}},\n' % (lit, ", ".join(str(x) for x in lens + [0])))
        f.write("};\n")
    print("用例表 → %s: %d 条" % (q, len(CASES)))


if __name__ == "__main__":
    main()
