#!/usr/bin/env python3
# judge_rows_tokclass.py — 把 anchor_metrics --row-out 的逐位置读数按【目标 token 的类】分桶(2026-10-24); 同日加: 把一份 HTML 切成
#   部署态 ids + 逐 token 内容区标签(--html 模式), 分桶时用 --regions 按内容区而不是字面类分。
#
# 用法 A(字面类分桶, spark ~/ds4-main):
#   ~/v41env/bin/python gguf-tools/scripts/judge_rows_tokclass.py --hf hf/DeepSeek-V4.1-Flash-1017 --ids <ids 文件> [--from N] [--regions <标签文件>] <臂名=A.rows> [...]
#   --from N: 只看第 N 行起(鹈鹕轨迹: 提示 11358 位, 生成从 11358 起); --regions: 一行一个标签、与 ids 同行数, 用它代替字面类
# 用法 B(造料): --hf … --html <页面> --out <ids 路径>   → 写 <ids>(BOS + 全文 token, 一行一个) 与 <ids>.regions(一行一个标签)
#   内容区标签: coord = 数值类属性值(d / points / transform / viewBox / cx cy r x y width height … 的引号内), attr = 其他属性值(fill / id / class …),
#   markup = 标签名 / 属性名 / 尖括号等号引号, text = 标签外的文字, css = <style> 内, js = <script> 内, comment = <!-- -->, space = 纯空白 token, bos
# 输出: 每个臂一张表 —— 类 / 位置数 / 占比 / Mean KLD / Σmin / Same top / ref 命中 / stu 命中; 多臂时末尾加 "B 对 A 的 ΔKLD% 按类"。
#
# 为什么要这把尺: 09-20 量过"该出数字的位置 top-1 51.5% vs 其余 65.4%"(旧底座); 10-24 用户判断"编程里的数学/美术成分没覆盖到, 鹈鹕 SVG 效果一般"。
#   字面类分桶(10-24 中午)说明损失不是数字专属, 最大一桶是 word; 但字面类量不到"是在画鹈鹕的嘴还是在写 CSS", 所以加内容区: 官方原件写的鹈鹕页面
#   teacher-forced 过三臂, 看量化在哪一区掉得最多, 直接回答"美术成分是不是被量化吃掉了"。
# 对齐: .rows 第 p 行是读完第 p 个 token 后对第 p+1 个 token 的预测(anchor_metrics: 命中 = argmax == 实际下一个 token)。本脚本用 ref 命中列自检:
#   ref_argmax == ids[p+1] 的行必须与 ref命中=1 的行一致(≥ 99%), 不一致就停 —— 对齐错一位, 分桶全是假数。
# 不做数值链: 只读文本, 算分桶均值; 判决仍只认 anchor_metrics 的五指标。
import argparse, os, re, sys
from tokenizers import Tokenizer

CLASSES = ["digit", "word", "punct", "space", "other"]
REGIONS = ["coord", "attr", "markup", "text", "css", "js", "comment", "space", "bos"]
DECISIONS = ["A_open", "A_close", "B_seg", "B_close", "C_attr", "C_close"]
NUM_ATTRS = {"d", "points", "transform", "viewbox", "cx", "cy", "r", "rx", "ry", "x", "y", "x1", "y1", "x2", "y2", "width", "height", "dx", "dy",
             "offset", "stroke-width", "opacity", "stroke-dasharray", "stroke-dashoffset", "font-size", "stddeviation", "values", "keytimes",
             "dur", "begin", "from", "to", "by", "rotate", "pathlength", "stroke-miterlimit", "fx", "fy", "k1", "k2", "k3", "k4", "scale"}
ATTR_RE = re.compile(r"""([A-Za-z_:][\w:.-]*)\s*=\s*("[^"]*"|'[^']*')""")
TAG_RE = re.compile(r"<\s*(/?)\s*([A-Za-z][\w:-]*)")


def tok_class(s):
    if any(c.isdigit() and c.isascii() for c in s): return "digit"
    if s.strip() == "" and s != "": return "space"
    if any(c.isalpha() and c.isascii() for c in s): return "word"
    if s and all((not c.isalnum()) and c.isascii() and not c.isspace() for c in s): return "punct"
    return "other"


def html_labels(t):
    """每个字符一个内容区标签(见文件头)+ 属性值字符所属的属性名(小写, 其余 "")。简单状态机: 注释 / 标签(含属性值) / script / style / 其余是文字。"""
    n = len(t); lab = ["text"] * n; an = [""] * n; i = 0; low = t.lower()
    while i < n:
        if t.startswith("<!--", i):
            j = t.find("-->", i); j = n if j < 0 else j + 3
            lab[i:j] = ["comment"] * (j - i); i = j; continue
        if t[i] == "<":
            j = t.find(">", i); j = n - 1 if j < 0 else j
            lab[i:j + 1] = ["markup"] * (j + 1 - i)
            seg = t[i:j + 1]
            for m in ATTR_RE.finditer(seg):
                a, b = m.start(2) + 1, m.end(2) - 1      # 引号内
                kind = "coord" if m.group(1).lower() in NUM_ATTRS else "attr"
                lab[i + a:i + b] = [kind] * (b - a); an[i + a:i + b] = [m.group(1).lower()] * (b - a)
            tm = TAG_RE.match(seg)
            i = j + 1
            if tm and not tm.group(1) and tm.group(2).lower() in ("script", "style"):
                name = tm.group(2).lower(); k = low.find("</" + name, i); k = n if k < 0 else k
                lab[i:k] = [("js" if name == "script" else "css")] * (k - i); i = k
            continue
        i += 1
    return lab, an


# 决策位(2026-10-25, 用户"path 为例细节丢了"): 画细节 = 在每个边界反复选"再来一个还是收", 偏差可能只有 1% 量级, 分区 Same top 看不见。
#   标签打在【实际发生的那个 token】上(.rows 第 p 行预测 ids[p+1], 分桶时取 dec[p+1]):
#   A_open/A_close  元素边界: 上一个元素刚结束("/>" 或 "</x>"), 这个 token 开新同级元素 "<x" / 收父元素 "</x"  → 元素个数
#   B_seg/B_close   路径段边界: d="…" 里一个数写完, 这个 token 是新命令字母 M/L/C/Q… / 收 Z 或引号                → 每条 path 的段数(造型)
#   C_attr/C_close  属性边界: 标签内一个属性值的引号刚收, 这个 token 开下一个属性名 / 收标签 ">" "/>"           → 每个元素的渲染细节
#   只认 token 的首个非空字符正好落在事件字符上的位置(分词把引号和 "/>" 合成一个 token 时那一处不计, 表里报的是能认出的 n)。
CLOSE_TAG_RE = re.compile(r"</[A-Za-z][\w:-]*>$")
PATH_CMD = set("MLCQSTAHVmlcqstahv")


def decision_label(t, lab, an, a, k):
    """a = token 起点, k = 首个非空字符; 返回决策位标签或 "-"."""
    c = t[k]; ps = t[:a].rstrip(); prev = ps[-1] if ps else ""
    if lab[k] == "markup":
        lt, gt = t.rfind("<", 0, k), t.rfind(">", 0, k)
        inside = lt > gt and lt < k                          # 在标签内部(不含标签自己的 "<")
        if inside and c in "\"'" and an[k - 1] and an[k - 1] != "d" and (t.startswith(">", k + 1) or t.startswith("/>", k + 1)):
            return "C_close"                                 # 分词把属性值的收尾引号和 "/>" 合成一个 token(pel2 整页如此): 事件字符是引号后那个
        if inside and prev in "\"'":
            if c == ">" or t.startswith("/>", k): return "C_close"
            if c.isalpha(): return "C_attr"
            return "-"
        if not inside and c == "<" and (ps.endswith("/>") or CLOSE_TAG_RE.search(ps)):
            if t.startswith("</", k): return "A_close"
            if k + 1 < len(t) and t[k + 1].isalpha(): return "A_open"
            return "-"
        if c == "\"" and k > 0 and an[k - 1] == "d":        # d="…" 的收尾引号
            p2 = ps.rstrip(",")
            if p2 and (p2[-1].isdigit() or p2[-1] == "."): return "B_close"
        return "-"
    if lab[k] == "coord" and an[k] == "d":
        p2 = ps.rstrip(",")
        if p2 and (p2[-1].isdigit() or p2[-1] == ".") and an[a - 1 if a else 0] == "d":
            if c in PATH_CMD: return "B_seg"
            if c in "Zz": return "B_close"
    return "-"


def build_html(tok, html, out):
    t = open(html, encoding="utf-8", errors="replace").read()
    lab, an = html_labels(t)
    enc = tok.encode(t, add_special_tokens=False)
    ids, regs, dec = [0], ["bos"], ["-"]
    for tid, (a, b) in zip(enc.ids, enc.offsets):
        piece = t[a:b]
        k = next((a + q for q, c in enumerate(piece) if not c.isspace()), None)
        ids.append(tid); regs.append("space" if k is None else lab[k]); dec.append("-" if k is None else decision_label(t, lab, an, a, k))
    open(out, "w").write("\n".join(map(str, ids)) + "\n")
    open(out + ".regions", "w").write("\n".join(regs) + "\n")
    open(out + ".decisions", "w").write("\n".join(dec) + "\n")
    cnt = {k: regs.count(k) for k in REGIONS}
    dc = {k: dec.count(k) for k in DECISIONS}
    print(f"{html}: {len(t)} 字符 → {len(ids)} 位(含 BOS) → {out}; 内容区: " + " ".join(f"{k} {v}" for k, v in cnt.items() if v)
          + "; 决策位: " + " ".join(f"{k} {v}" for k, v in dc.items()))
    return 0


def decision_table(name, R, ids, dec):
    """每个决策位标签: n / FP 与学生给实际 token 的几何均值 p / 比值; 有 --set 列时再报"收尾 token 质量"(A_open 处 = 本可以收却没收的质量)。"""
    B = {k: [0, 0.0, 0.0, 0.0, 0.0] for k in DECISIONS}   # n, Σnll_ref, Σnll_stu, Σmass_ref, Σmass_stu
    has_set = R and len(R[0]) >= 12
    for p, r in enumerate(R):
        if p + 1 >= len(ids) or r[8] < 0: continue
        k = dec[p + 1]
        if k == "-": continue
        b = B[k]; b[0] += 1; b[1] += r[8]; b[2] += r[9]
        if has_set: b[3] += r[10]; b[4] += r[11]
    import math
    print(f"== {name} 决策位: n / p_FP(几何均值) / p_学生 / 学生÷FP" + (" / 收尾质量 FP / 学生" if has_set else ""))
    for k in DECISIONS:
        b = B[k]
        if not b[0]: print(f"  {k:8s} {0:6d}  (没认出位置)"); continue
        pr, psu = math.exp(-b[1] / b[0]), math.exp(-b[2] / b[0])
        line = f"  {k:8s} {b[0]:6d}  {pr:7.4f} {psu:7.4f} {psu / pr:7.3f}"
        if has_set: line += f"   {b[3] / b[0]:7.4f} {b[4] / b[0]:7.4f}"
        print(line)


def load_rows(path):
    """.rows 一行: 位置 kld Σmin same ref命中 stu命中 ref_argmax stu_argmax [ref_nll stu_nll [set质量ref set质量stu]](anchor_metrics --row-out)"""
    R = []
    for ln in open(path):
        c = ln.split()
        if len(c) < 8: continue
        R.append((int(c[0]), float(c[1]), float(c[2]), int(c[3]), int(c[4]), int(c[5]), int(c[6]), int(c[7])) + tuple(float(x) for x in c[8:12]))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True); ap.add_argument("--ids"); ap.add_argument("--from", dest="frm", type=int, default=0)
    ap.add_argument("--regions"); ap.add_argument("--html"); ap.add_argument("--out")
    ap.add_argument("--decisions", help="<ids>.decisions: 按决策位再出一张表(要 .rows 带第 9/10 列)")
    ap.add_argument("--close-set", help="只造料: 把词表里 '</' 开头的 token id 写到这个文件(anchor_metrics --set)")
    ap.add_argument("arms", nargs="*", help="名=rows 路径")
    a = ap.parse_args()
    tok = Tokenizer.from_file(os.path.join(a.hf, "tokenizer.json"))
    if a.close_set:   # 词表里所有"收尾"token("</" 开头): anchor_metrics --set 的料, A_open 处这组质量 = "本可以收却没收"的程度
        ids = [i for i in range(tok.get_vocab_size()) if tok.decode([i]).lstrip().startswith("</")]
        open(a.close_set, "w").write("\n".join(map(str, ids)) + "\n"); print(f"{a.close_set}: {len(ids)} 个 '</' 开头的 token"); return 0
    if a.html:
        if not a.out: print("★--html 要配 --out★"); return 2
        return build_html(tok, a.html, a.out)
    if not a.ids or not a.arms: print("★要 --ids 与至少一个 臂名=rows★"); return 2
    ids = [int(x) for x in open(a.ids).read().split()]
    regs = dec = None
    if a.regions:
        regs = open(a.regions).read().split()
        if len(regs) != len(ids): print(f"★--regions {len(regs)} 行 ≠ ids {len(ids)} 行★"); return 2
    if a.decisions:
        dec = open(a.decisions).read().split()
        if len(dec) != len(ids): print(f"★--decisions {len(dec)} 行 ≠ ids {len(ids)} 行★"); return 2
    classes = REGIONS if regs else CLASSES
    arms = [(x.split("=", 1)[0], x.split("=", 1)[1]) for x in a.arms]
    tables = {}
    for name, path in arms:
        R = load_rows(path)
        chk = [(r[4] == 1) == (p + 1 < len(ids) and r[6] == ids[p + 1]) for p, r in enumerate(R) if p + 1 < len(ids)]
        agree = sum(chk) / max(1, len(chk))
        if agree < 0.99:
            print(f"★{name}: 对齐自检 {agree:.4f} < 0.99(rows {len(R)} 行, ids {len(ids)}), 停★"); return 2
        B = {k: [0, 0.0, 0.0, 0, 0, 0] for k in classes}   # n, Σkld, Σsmin, Σsame, Σref命中, Σstu命中
        for p, r in enumerate(R):
            if p < a.frm or p + 1 >= len(ids): continue
            k = regs[p + 1] if regs else tok_class(tok.decode([ids[p + 1]]))
            b = B[k]; b[0] += 1; b[1] += r[1]; b[2] += r[2]; b[3] += r[3]; b[4] += r[4]; b[5] += r[5]
        n_all = sum(b[0] for b in B.values())
        tables[name] = B
        print(f"== {name}  ({os.path.basename(path)[:80]}…)  位置 {n_all}(从第 {a.frm} 行起)")
        print(f"  {'类':7s} {'n':>7s} {'占比':>6s} {'KLD':>8s} {'Σmin':>7s} {'Same':>7s} {'ref命中':>7s} {'stu命中':>7s}")
        for k in classes:
            b = B[k]
            if b[0] == 0: continue
            print(f"  {k:7s} {b[0]:7d} {100*b[0]/n_all:5.1f}% {b[1]/b[0]:8.4f} {b[2]/b[0]:7.4f} {100*b[3]/b[0]:6.2f}% {100*b[4]/b[0]:6.2f}% {100*b[5]/b[0]:6.2f}%")
        tot = [sum(b[i] for b in B.values()) for i in range(6)]
        print(f"  {'all':7s} {tot[0]:7d} {'100.0%':>6s} {tot[1]/tot[0]:8.4f} {tot[2]/tot[0]:7.4f} {100*tot[3]/tot[0]:6.2f}% {100*tot[4]/tot[0]:6.2f}% {100*tot[5]/tot[0]:6.2f}%")
        if dec:
            if len(R[0]) < 10: print(f"★{name}: .rows 没有第 9/10 列(目标 NLL), 要新 anchor_metrics 重出★"); return 2
            decision_table(name, R, ids, dec)
    if len(arms) > 1:
        a0 = tables[arms[0][0]]
        for name, _ in arms[1:]:
            B = tables[name]; parts = []
            for k in classes:
                if a0[k][0] and B[k][0] and a0[k][1] > 0:
                    parts.append(f"{k} {100*(B[k][1]/B[k][0] - a0[k][1]/a0[k][0])/(a0[k][1]/a0[k][0]):+.1f}%")
            print(f"== {name} 对 {arms[0][0]} 的 ΔKLD 按类: " + " / ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
