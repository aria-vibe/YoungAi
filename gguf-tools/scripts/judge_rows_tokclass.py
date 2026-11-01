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


CONF = [("p>0.9", 0.0, 0.10536), ("p.5-.9", 0.10536, 0.69315), ("p<0.5", 0.69315, 1e30)]   # 按 FP 给实际下一位的 p 切: -ln 0.9 / -ln 0.5


def conf_table(name, R, frm, n_ids):
    """按 FP 对实际下一位的置信度分桶(10-26 晚): 总 Same top 被 "FP 自己也在多选" 的位撑着(设计段一半位 p<0.5),
    那一桶的错不是细节错; 细节错 = FP 笃定(p>0.9 只有一种写法)或有倾向(0.5~0.9)时学生选错 —— 判 ③ 对细节有没有用只看前两桶。"""
    B = {k: [0, 0.0, 0, 0] for k, _, _ in CONF}   # n, Σkld, Σsame, Σstu命中
    for p, r in enumerate(R):
        if p < frm or p + 1 >= n_ids or r[8] < 0: continue
        for k, lo, hi in CONF:
            if lo <= r[8] < hi: b = B[k]; b[0] += 1; b[1] += r[1]; b[2] += r[3]; b[3] += r[5]; break
    n_all = sum(b[0] for b in B.values()) or 1
    print(f"== {name} 按 FP 置信度: 桶 / n / 占比 / KLD / Same top / 错 top-1")
    for k, _, _ in CONF:
        b = B[k]
        if b[0]: print(f"  conf {k:7s} {b[0]:7d} {100*b[0]/n_all:5.1f}% {b[1]/b[0]:8.4f} {100*b[2]/b[0]:6.2f}% {100 - 100*b[2]/b[0]:6.2f}%")


def agg_list(a, tok):
    """--list 模式(10-27, 泛化留出尺): 清单一行 "id 全长 生成段起点"(think_trace_collect.py ids 的 list.txt), 每个臂给一个带 {id} {n} 的 rows 路径模板;
    把所有段的生成段合成一张表 —— 一道题几千位, 单题的 Same top 抖得厉害, 合起来才看得出侧车在没见过的题上涨没涨。
    每段先过对齐自检(同主模式), 不过就停。"""
    segs = [ln.split() for ln in open(a.list) if ln.strip()]
    for name, pat in [(x.split("=", 1)[0], x.split("=", 1)[1]) for x in a.arms]:
        T = [0, 0.0, 0]; C = {k: [0, 0.0, 0] for k, _, _ in CONF}; L = {k: [0, 0.0, 0] for k in CLASSES}
        for sid, n, frm in segs:
            ids = [int(x) for x in open(os.path.join(a.ids_dir, f"{sid}.ids")).read().split()]
            R = load_rows(pat.replace("{id}", sid).replace("{n}", n))
            chk = [(r[4] == 1) == (r[6] == ids[p + 1]) for p, r in enumerate(R) if p + 1 < len(ids)]
            if sum(chk) / max(1, len(chk)) < 0.99: print(f"★{name} {sid}: 对齐自检不过, 停★"); return 2
            for p, r in enumerate(R):
                if p < int(frm) or p + 1 >= len(ids) or r[8] < 0: continue
                for b in (T, L[tok_class(tok.decode([ids[p + 1]]))]): b[0] += 1; b[1] += r[1]; b[2] += r[3]
                for k, lo, hi in CONF:
                    if lo <= r[8] < hi: b = C[k]; b[0] += 1; b[1] += r[1]; b[2] += r[3]; break
        f = lambda b: f"n {b[0]} Same {100*b[2]/max(1,b[0]):.2f}% KLD {b[1]/max(1,b[0]):.4f}"
        print(f"== {name} 留出 {len(segs)} 段合计: {f(T)}")
        print("  conf " + " | ".join(f"{k} 错 {100 - 100*C[k][2]/max(1,C[k][0]):.2f}% (n {C[k][0]})" for k, _, _ in CONF))
        print("  类   " + " | ".join(f"{k} Same {100*L[k][2]/max(1,L[k][0]):.2f}% (n {L[k][0]})" for k in ("digit", "word", "punct")))
    return 0


def paired(a):
    """--paired 模式(10-28 判决门): 同一批位置上两臂逐位相减 —— 新臂 − 基臂的 KLD / Same top / Σmin, 均值 + 标准误。
    清单一行 "名 全长 起:止,起:止,…"(think_trace_collect.py ids --pack 的 packs.txt; 编程判决份写一行 "code 15360 0:15360"), 臂写成 名=带 {id} 的 rows 模板, 恰好两臂(基在前)。
    只数生成段: 第 p 行是对 ids[p+1] 的预测, 起 ≤ p+1 < 止 才算。标准误按每 --block 个相邻位一块取块均值(逐位不独立, 直接按位算会把误差报小;
    缺省 256; wt2 只有 512 位要配 32, 否则两块估不出误差)。
    --target ID --ids <ids>(结束位尺): 只数"下一个 token 是 ID"的位置(docend 的 25 个真 EOS), 这时配 --block 1(各篇的结尾互相隔得远);
    读数看学生给真 token 的 NLL(.rows 第 10 列): 结尾位 KLD 被别的候选摊薄, "该停的地方给停多大概率"才是要守的东西。
    为什么不比两份汇总数: 汇总的 Δ 不知道有多大是噪声; 同位相减再按块估误差, 门才能说"降/升得超过 2 个标准误"。
    出错会怎样: 两臂行数对不上(判的不是同一份 ids)就停; 末尾一行 PAIRED_JSON 给脚本读。"""
    import json, math
    (n0, p0), (n1, p1) = [(x.split("=", 1)[0], x.split("=", 1)[1]) for x in a.arms]
    B = a.block   # --ids 可以是带 {id} 的模板(多条序列各一份 ids, ptf_metrics 写的 <名字>.ids)
    load_ids = lambda sid: [int(x) for x in open(a.ids.replace("{id}", sid)).read().split()] if a.target is not None else None
    K = 5   # 每位: KLD / Same / Σmin / 学生 NLL / FP NLL —— 块里存 Δ(前四个), tot 存两臂各自的和
    blk, cur, tot, n = [], [0.0] * K, [[0.0] * K, [0.0] * K], 0
    hit10 = hit01 = 0   # Same top 一中一不中的位: 新中旧不中 / 旧中新不中(反修 A 段端到端择优同一本账, v41_e2e_select.inc.c e2e_pair)
    for ln in open(a.paired):
        if not ln.strip(): continue
        sid, _, rg = ln.split()
        R0, R1 = load_rows(p0.replace("{id}", sid)), load_rows(p1.replace("{id}", sid)); ids = load_ids(sid)
        if len(R0) != len(R1): print(f"★{sid}: {n0} {len(R0)} 行 ≠ {n1} {len(R1)} 行, 判的不是同一份 ids, 停★"); return 2
        if len(R0[0]) < 10: print(f"★{sid}: .rows 没有第 9/10 列(目标 NLL), 要新 anchor_metrics 重出★"); return 2
        for s, e in (tuple(map(int, x.split(":"))) for x in rg.split(",")):
            for p in range(max(0, s - 1), min(len(R0), e - 1)):
                if ids is not None and ids[p + 1] != a.target: continue
                v0 = (R0[p][1], R0[p][3], R0[p][2], R0[p][9], R0[p][8]); v1 = (R1[p][1], R1[p][3], R1[p][2], R1[p][9], R1[p][8])
                n += 1; cnt = n % B; hit10 += v1[1] > v0[1]; hit01 += v0[1] > v1[1]
                for i in range(K): tot[0][i] += v0[i]; tot[1][i] += v1[i]; cur[i] += v1[i] - v0[i]
                if cnt == 0: blk.append([c / B for c in cur]); cur = [0.0] * K
    if len(blk) < 2: print(f"★配对位置 {n} 个, 不足两块(--block {B}), 门判不了★"); return 2
    nb = len(blk)
    se = [math.sqrt(sum((b[i] - sum(x[i] for x in blk) / nb) ** 2 for b in blk) / (nb - 1) / nb) for i in range(K)]
    m0, m1 = [t / n for t in tot[0]], [t / n for t in tot[1]]
    d = dict(n=n, blocks=nb, kld0=m0[0], kld1=m1[0], same0=100 * m0[1], same1=100 * m1[1], smin0=m0[2], smin1=m1[2], nll0=m0[3], nll1=m1[3], ref_nll=m0[4])
    d.update(dkld=m1[0] - m0[0], dkld_se=se[0], dsame=100 * (m1[1] - m0[1]), dsame_se=100 * se[1], dsmin=m1[2] - m0[2], dsmin_se=se[2], dnll=m1[3] - m0[3], dnll_se=se[3])
    d.update(same_D=hit10 - hit01, same_d=hit10 + hit01)   # 没有真差别时 D 的标准差 ≈ √d; 门按 A 段: D < −2√d 否决
    z = lambda k: d[k] / max(d[k + "_se"], 1e-12)
    print(f"== 配对 {n1} 对 {n0}: {n} 位({nb} 块 × {B}){'' if a.target is None else f', 只数下一个 token = {a.target} 的位'}")
    print(f"  KLD   {d['kld0']:.5f} → {d['kld1']:.5f}  Δ {d['dkld']:+.5f} ± {d['dkld_se']:.5f}({z('dkld'):+.1f} SE)")
    print(f"  Same  {d['same0']:.2f}% → {d['same1']:.2f}%  Δ {d['dsame']:+.2f} ± {d['dsame_se']:.2f} pp; 新中旧不中 − 旧中新不中 = {d['same_D']:+d} 行(不一致 {d['same_d']} 行, 噪声 √d = {math.sqrt(d['same_d']):.1f})")
    print(f"  Σmin  {d['smin0']:.4f} → {d['smin1']:.4f}  Δ {d['dsmin']:+.4f} ± {d['dsmin_se']:.4f}")
    print(f"  真 token NLL  {d['nll0']:.4f} → {d['nll1']:.4f}  Δ {d['dnll']:+.4f} ± {d['dnll_se']:.4f}({z('dnll'):+.1f} SE); 参照 {d['ref_nll']:.4f}"
          + ("" if a.target is None else f"  ⇒ 真 token 几何均值 p: 参照 {math.exp(-d['ref_nll']):.4f} / {n0} {math.exp(-d['nll0']):.4f} / {n1} {math.exp(-d['nll1']):.4f}"))
    print("PAIRED_JSON " + json.dumps(d))
    return 0


def misses(a):
    """--paired 清单 --misses(10-28 段 B 连回几轮后的复盘): 看 Same top 没中的位是什么样, 回答"Same top 往 100% 推卡在哪"。
    臂 ≥ 2 个(按链上先后), 每臂: 没中的位按 FP 对【真 token】的把握分桶 —— FP 自己猜中真 token 时 p = exp(-ref_nll) 就是 FP top-1 的概率;
    FP 也没猜中真 token 的位单列(那里 FP 的 top-1 概率读不到)。首末两臂再对一次账: 修好几位 / 修坏几位 / 一直错几位, 与 KLD 按首臂十分位的变化。
    为什么: 配对门只给均值差, 看不出"没中的位是 FP 自己都拿不准的近乎平票, 还是 FP 很笃定而学生错了" —— 前者推不动也不该推, 后者才是丢掉的能力。"""
    import math
    lst = [ln.split() for ln in open(a.paired) if ln.strip()]
    arms = [(x.split("=", 1)[0], x.split("=", 1)[1]) for x in a.arms]
    P = {}   # 臂 → [(kld, same, ref_hit, p_ref_true)]
    for name, tpl in arms:
        v = []
        for sid, _, rg in lst:
            R = load_rows(tpl.replace("{id}", sid))
            for s, e in (tuple(map(int, x.split(":"))) for x in rg.split(",")):
                v += [(R[p][1], R[p][3], R[p][4], math.exp(-R[p][8])) for p in range(max(0, s - 1), min(len(R), e - 1))]
        P[name] = v
    n = len(P[arms[0][0]])
    if any(len(v) != n for v in P.values()): print("★各臂位置数对不上★"); return 2
    def bucket(r): return "FP没猜中真token" if not r[2] else ("FP p<0.5" if r[3] < 0.5 else "FP 0.5~0.9" if r[3] < 0.9 else "FP p≥0.9")
    BK = ["FP p≥0.9", "FP 0.5~0.9", "FP p<0.5", "FP没猜中真token"]
    tot = {k: sum(1 for r in P[arms[0][0]] if bucket(r) == k) for k in BK}
    print(f"== Same top 没中的位, 按 FP 对真 token 的把握分桶(共 {n} 位; 各桶总位数: " + " / ".join(f"{k} {tot[k]}" for k in BK) + ")")
    for name, _ in arms:
        v = P[name]; m = [r for r in v if not r[1]]; kt = sum(r[0] for r in v)
        print(f"  {name}: Same {100*(n-len(m))/n:.2f}%  没中 {len(m)} 位(占总 KLD {100*sum(r[0] for r in m)/kt:.1f}%)  "
              + " / ".join(f"{k} {sum(1 for r in m if bucket(r)==k)}({100*sum(1 for r in m if bucket(r)==k)/max(1,tot[k]):.1f}%)" for k in BK))
    (f0, v0), (f1, v1) = (arms[0][0], P[arms[0][0]]), (arms[-1][0], P[arms[-1][0]])
    print(f"== {f0} → {f1} 逐位对账(桶按 {f0} 的位定)")
    for k in BK:
        ix = [i for i in range(n) if bucket(v0[i]) == k]
        fix = sum(1 for i in ix if not v0[i][1] and v1[i][1]); brk = sum(1 for i in ix if v0[i][1] and not v1[i][1]); both = sum(1 for i in ix if not v0[i][1] and not v1[i][1])
        print(f"  {k:14s} 修好 {fix:4d}  修坏 {brk:4d}  净 {fix-brk:+5d}  一直错 {both:4d}")
    o = sorted(range(n), key=lambda i: v0[i][0]); D = 10
    print(f"== KLD 按 {f0} 的十分位: 首臂均值 → 末臂均值(Δ%), 末行是各分位对总 Δ 的贡献")
    rows = []
    for d in range(D):
        ix = o[d * n // D:(d + 1) * n // D]; k0 = sum(v0[i][0] for i in ix) / len(ix); k1 = sum(v1[i][0] for i in ix) / len(ix)
        rows.append((k0, k1, (k1 - k0) * len(ix) / n))
    print("  " + "  ".join(f"d{d}:{k0:.3f}→{k1:.3f}({100*(k1-k0)/max(k0,1e-9):+.0f}%)" for d, (k0, k1, _) in enumerate(rows)))
    print("  贡献 " + "  ".join(f"d{d}:{c:+.5f}" for d, (_, _, c) in enumerate(rows)) + f"  合计 {sum(c for *_, c in rows):+.5f}")
    return 0


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
    ap.add_argument("--conf", action="store_true", help="再出一张按 FP 置信度分桶的表(要 .rows 带第 9 列; 判细节只看 p>0.9 与 0.5~0.9 两桶)")
    ap.add_argument("--list", help="多段合计模式: 清单(id 全长 生成段起点), 臂写成 名=带 {id} {n} 的 rows 模板; 配 --ids-dir")
    ap.add_argument("--ids-dir", help="--list 的 ids 目录(<id>.ids)")
    ap.add_argument("--paired", help="配对模式: 清单(名 全长 起:止,…), 两臂 基=模板 新=模板(模板带 {id}), 逐位相减出 Δ ± 标准误")
    ap.add_argument("--misses", action="store_true", help="配 --paired: Same top 没中的位按 FP 把握分桶 + 首末臂修好/修坏对账 + KLD 十分位(臂 ≥ 2, 按链上先后)")
    ap.add_argument("--block", type=int, default=256, help="--paired 的标准误按几位一块")
    ap.add_argument("--target", type=int, help="--paired 只数下一个 token 是这个 id 的位(配 --ids; 结束位尺: EOS = 1)")
    ap.add_argument("arms", nargs="*", help="名=rows 路径")
    a = ap.parse_args()
    if a.paired and a.misses:
        if len(a.arms) < 2: print("★--misses 要至少两臂(按链上先后)★"); return 2
        return misses(a)
    if a.paired:
        if len(a.arms) != 2: print("★--paired 要恰好两臂: 基=模板 新=模板★"); return 2
        return paired(a)
    tok = Tokenizer.from_file(os.path.join(a.hf, "tokenizer.json"))
    if a.close_set:   # 词表里所有"收尾"token("</" 开头): anchor_metrics --set 的料, A_open 处这组质量 = "本可以收却没收"的程度
        ids = [i for i in range(tok.get_vocab_size()) if tok.decode([i]).lstrip().startswith("</")]
        open(a.close_set, "w").write("\n".join(map(str, ids)) + "\n"); print(f"{a.close_set}: {len(ids)} 个 '</' 开头的 token"); return 0
    if a.html:
        if not a.out: print("★--html 要配 --out★"); return 2
        return build_html(tok, a.html, a.out)
    if a.list:
        if not a.ids_dir or not a.arms: print("★--list 要配 --ids-dir 与至少一个 臂名=rows 模板★"); return 2
        return agg_list(a, tok)
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
        if a.conf:
            if len(R[0]) < 10: print(f"★{name}: .rows 没有第 9 列(FP 目标 NLL), 要新 anchor_metrics 重出★"); return 2
            conf_table(name, R, a.frm, len(ids))
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
