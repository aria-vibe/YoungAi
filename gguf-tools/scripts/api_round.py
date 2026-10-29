#!/usr/bin/env python3
"""api_round.py — 段 B 自进化一轮里官方 API 的两个新角色(2026-10-27 晚, 用户: "官方 api 作为裁判和语料来源, 不要泄题, 一小时一轮";
"以鹈鹕测试为例, 其实是要求整个模型能力的还原, 不要跑偏" —— 评的是全族通用题, 鹈鹕只是图形那一道)。

  questions  官方 API 现出这一轮的新题(各族 N 道), 和历史题(--seen 文件, 一行一题或 jsonl 的 q 字段)去重, 命中 --ban 词的直接丢。
             为什么要每轮新出: 固定题集训几轮就是背题; 新题让每轮的料都是没见过的, 判决用的鹈鹕/留出集永远不进料。
             --ban 是防泄题的词表(鹈鹕、鸟、骑车 + 留出集的动物名), 由调用方给, 这里不写死任何题材。
  review     官方 API 当评委: 一份回答(只看正文, 不看思考)按通用细则逐项 0~10(正确/完整/遵循指令/质量) + 总分 0~100, 输出 JSON。
             绘图题 DeepSeek API 看不到图, 评的是代码: 细则要求它先从坐标量出主体标志部位的尺寸比例(measure), 注释和命名不作数。
             同一份样本(id + 正文 md5)已经评过就跳过 —— 官方参照只评一次, 每轮只评新样本。每条记 qh(题面 md5 前 8 位), summary 按题对齐。
  summary    多份 review 文件按题(qh)对齐: 每题各臂的均分, 再报每臂"逐题对官方的还原率"的均值; 第一份当参照(官方)。
  tokclass   量化缺口按 token 类拆(编程尺 / 本轮料各一张表), 给 plan 当事实(10-28)。
  seeds      段 B 第一层"原始语料"的种子(10-28): 官方出每种语言的源文件开头 1~3 行, 给 api_teacher.py collect --raw 当裸前缀。
  plan       官方复盘(10-28): 本轮事实(编程尺分语言 KLD 与本轮变化、门、训练料 KLD、题目、历史各轮、评委评语)交给官方 API,
             它定下一轮的出题语言 / 题数 / 学习率(questions --langs --brief 吃它)。没有这一步, 每轮出题是随机的, 学完也不知道下一轮该补哪里。

用法(spark 仓库根, ~/v41env/bin/python):
  api_round.py questions --key K --round 1 --n 4 --out q.jsonl --seen a.txt b.jsonl … --ban 鹈鹕 pelican 鸟 … [--prefix evohold]
  api_round.py review --key K --raw samples.jsonl --out review.jsonl [--rubric 文件] [--jobs 4]
  api_round.py summary 官方=a.review.jsonl 现役=b.review.jsonl 本轮=c.review.jsonl
  api_round.py plan --key K --langtab lang.tsv --code-json '{…}' --wt2-json '{…}' --q q.jsonl --train "0.074 0.066" --lr 2e-5 --tag a002 \
      --verdict rollback --why … --review r.jsonl --plan-in plan_in.json --history evo/a*/facts.json --facts facts.json --out plan.json
出错会怎样: API 失败重试 5 次后这一条记 ok:false(questions 少出几道照常往下走; review 少一份就是 n 少一份, summary 里能看到 n);
评委回的不是 JSON → ok:false 并把原文前 300 字留在 raw 字段, 不猜分。key 只进请求头, 不打印。"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from think_trace_collect import API_URL, read_jsonl  # noqa: E402

# 题族: 与 mix_q_* 题集同一套分法(10-26 等份配比), 描述只讲题型, 不点名任何具体题材
FAMILIES = {
    "code": "编程实现题: 各种语言(Python/C/C++/Rust/Go/JavaScript/TypeScript/SQL/Shell 等)的中高难度任务, 要求给出完整可运行的代码",
    "math": "数学题: 推导、证明、计算、概率统计、几何, 要求写出推理过程",
    "data": "数据分析与处理题: 给出数据的结构或一小段样例, 要求分析结论或写处理/可视化代码",
    "writing": "写作题: 说明文、技术文档、文案、改写、翻译、总结",
    "svgchar": "用单个 HTML 文件(SVG/CSS/JS)画一个角色或物件的 2D 插画或动画, 角色要有清楚的外形特征和比例",
    "general": "通用决策与规划题: 给建议、做方案、权衡取舍、解释概念",
}


def api_call(key, prompt, timeout, model="deepseek-flash"):
    """非流式一问一答; 返回 content。评委/出题都是短输出, 不需要流式。"""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False}
    req = urllib.request.Request(API_URL, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.load(r)
    return res["choices"][0]["message"].get("content") or ""


def retry(fn, tag):
    delay = 5
    for attempt in range(5):
        try:
            return fn()
        except Exception as e:   # 网络/5xx/429/IncompleteRead 都在这
            if attempt == 4:
                print(f"  [{tag}] ★{type(e).__name__}: {str(e)[:120]}, 放弃★", flush=True); return None
            print(f"  [{tag}] {type(e).__name__}, {delay}s 后重试", flush=True); time.sleep(delay); delay *= 2


def json_block(text, opener, closer):
    a, b = text.find(opener), text.rfind(closer)
    if a < 0 or b <= a: return None
    try: return json.loads(text[a:b + 1])
    except json.JSONDecodeError: return None


# 编程题题面固定的尾句(10-28 a010): brief 写了"纯代码为主", 官方照样附大段中文讲解, 缺口加权后 KL 前 10% 的位 46% 落在"使用/输出/解析"这类讲解词上,
#   编程尺是纯源文件、没有讲解, 那份梯度搬不过去(30 种语言 21 种变差)。尾句由这里加, 不靠出题模型照办; 按题面中英文挑一句
CODE_ONLY = {"zh": "\n只输出一个完整的源文件(放在一个代码块里), 不要任何解释文字; 代码注释用英文。",
             "en": "\nOutput only one complete source file in a single code block, with no explanation text; write code comments in English."}


def norm(q):
    for s in CODE_ONLY.values(): q = q.replace(s, "")   # 去重比的是题面本身, 带不带尾句都算同一题
    return re.sub(r"\s+", "", q).lower()


def load_seen(paths):
    seen = []
    for p in paths or []:
        if not os.path.exists(p): continue
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if not ln or ln.startswith("#"): continue
            if ln.startswith("{"):
                q = json.loads(ln).get("q")
                if q: seen.append(q)
            else:
                seen.append(ln.replace("\\n", "\n"))
    return seen


def cmd_questions(a):
    key = open(a.key, encoding="utf-8").read().strip()
    seen = load_seen(a.seen); seen_n = {norm(q) for q in seen}
    ban = [w.lower() for w in a.ban or [] if w.strip()]   # 空词对任何题都"命中"(10-27 探针实撞: 空数组在 zsh 里展开成一个空串, 题全被丢)
    fams = a.families.split(",") if a.families else list(FAMILIES)

    def ask(fam):
        # 只把最近的 60 道历史题给它看(全给太长); 去重本身靠下面的规范化比对, 提示只是让它往远处出
        near = seen[-60:]
        # --langs/--brief 来自上一轮的官方复盘(plan): 不给时题型描述里列的头几种语言会被反复挑中(10-28 a001/a002 八道题全是 Python/C/Go/TS, 编程尺却是 30 种语言)
        steer = (f"每道题指定一种编程语言, 只从这些里选, 按顺序轮流、每种至少一道(题数够的话): {', '.join(a.langs)}。\n" if a.langs else "") + \
                (f"出题方向: {a.brief}\n" if a.brief else "")
        prompt = (f"请出 {a.n} 道全新的题目, 题型: {FAMILIES[fam]}。\n" + steer +
                  f"要求: 难度中等偏上, 每道题自成一体、一句到几句话说清; 中文英文混合出(大约各半); 题目之间题材要拉开。\n"
                  f"禁止出现以下任何内容(题材、字样都不行): {', '.join(a.ban or [])}。\n"
                  f"不要和这些已有的题相似:\n" + "\n".join(f"- {q[:160]}" for q in near) +
                  f"\n\n只输出一个 JSON 字符串数组, 每个元素是一道题的完整题面, 不要别的文字。")
        txt = retry(lambda: api_call(key, prompt, a.timeout), f"q:{fam}")
        arr = json_block(txt or "", "[", "]")
        return fam, [q for q in (arr or []) if isinstance(q, str) and q.strip()]

    out, drop = [], []
    with ThreadPoolExecutor(len(fams)) as ex:
        for fu in as_completed([ex.submit(ask, f) for f in fams]):
            fam, qs = fu.result(); k = 0
            for q in qs:
                if any(w in q.lower() for w in ban): drop.append((fam, "禁词", q)); continue
                if norm(q) in seen_n: drop.append((fam, "重复", q)); continue
                seen_n.add(norm(q))
                if fam == "code": q = q.strip() + CODE_ONLY["zh" if re.search(r"[一-鿿]", q) else "en"]
                out.append({"id": f"{a.prefix or f'evo{a.round:03d}'}_{fam}_{k:02d}", "src": fam, "q": q.strip()}); k += 1
                if k >= a.n: break
    out.sort(key=lambda r: r["id"])
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"questions: {len(out)} 题 → {a.out} (" + ", ".join(f"{s} {sum(1 for r in out if r['src'] == s)}" for s in fams) + f"); 丢 {len(drop)}")
    for fam, why, q in drop: print(f"  丢[{fam}/{why}] {q[:80]}")
    if not out: sys.exit("★一道题都没出来★")


def cmd_seeds(a):
    """段 B 第一层"原始语料"的种子(10-28 用户: "原始模型和后训练专项任务是两回事, 分两块还原, 先原始语料找回原始能力, 再任务能力"):
    官方出每种语言的源文件开头(1~3 行), api_teacher.py collect --raw 拿它当裸前缀接着写。为什么开头也让官方出: 只用四样东西(官方 API / 底座 / 段 A / 段 B),
    不碰旧料, 也不碰编程尺原文; 开头只是起个头, 料的主体是官方续写的那几百上千位。"""
    key = open(a.key, encoding="utf-8").read().strip()
    ban = [w.lower() for w in a.ban or [] if w.strip()]

    def ask(lang):
        name = lang.removeprefix("code_").replace("_", " ")
        # "中大型文件"(10-28 o003 前): 只说"源文件"时官方续写中位 253 位、最短二三十位(像示例片段, 写几行就收尾); 编程尺的窗是真仓库文件里截的 512 位
        prompt = (f"请给出 {a.n} 个 {name} 源文件的开头, 风格像真实开源仓库里的文件(该语言常见的文件头注释、包/模块声明、import/include 等); "
                  f"这些文件完整写下来都是几百行的中大型源文件(真实项目里的模块, 不是示例片段或玩具程序)。"
                  f"每个只要文件最前面 1~3 行, 停在还没写到实现的地方; 这些文件的用途、所属项目类型要彼此拉开。"
                  + (f"方向: {a.brief}" if a.brief else "")
                  + f"禁止出现: {', '.join(a.ban or [])}。只输出一个 JSON 字符串数组, 每个元素是一个文件开头的原文, 不要别的文字。")
        arr = json_block(retry(lambda: api_call(key, prompt, a.timeout), f"seed:{name}") or "", "[", "]")
        return lang, [s for s in (arr or []) if isinstance(s, str) and s.strip()]

    out, drop = [], 0
    with ThreadPoolExecutor(min(8, len(a.langs))) as ex:
        for fu in as_completed([ex.submit(ask, l) for l in a.langs]):
            lang, ss = fu.result(); k = 0
            for s in ss:
                if any(w in s.lower() for w in ban): drop += 1; continue
                # 开头以整行收尾(一个换行): 10-28 o003 试过把末尾空白全去掉, 停在 ")" 上官方没法换行(")\n" 在词表里是一个 token),
                #   只能在同一行接 " # Effectively…"; 停在换行上偶尔第一位就采到 EOS(三采一次), 那是采样, collect --raw 遇到 0 位重采
                out.append({"id": f"{a.prefix}_{lang}_{k:02d}", "src": lang, "q": s.rstrip() + "\n"}); k += 1
                if k >= a.n: break
    out.sort(key=lambda r: r["id"])
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"seeds: {len(out)} 个开头 → {a.out}({len(a.langs)} 种语言, 每种 ≤ {a.n}); 禁词丢 {drop}")
    if not out: sys.exit("★一个开头都没出来★")


# 细则是通用的(10-27 用户: 要的是整个模型能力的还原, 鹈鹕只是例子)。绘图那条来自 10-27 探针: 第一版只给几何描述时, 评委给 v4b(截图是鸭子)主体 8 分、判"像",
#   它被代码里的注释和 id/class 名(pelican / pouch)带着走; 改成先从坐标量出标志部位的尺寸写进 measure 后, 量出喙长/头径 官方 2.5~3.3 / v4b 1.18, 和截图对得上。
RUBRIC = """你是严格的评委。题目:
「{q}」
下面是一个模型的回答(只有正文)。逐项打分(每项 0~10, 10 = 完全做到), 先核对再打分, 不要相信回答对自己的描述:
1. correct 正确性: 代码能否按题意跑通(逐段推演, 指出会出错的地方)、推导与计算是否对、结论是否成立
2. complete 完整性: 题目要求的每一点是否都做到
3. follow 遵循指令: 语言、格式、约束、输出形式是否照题目要求
4. quality 质量: 清晰、简洁、专业水准
★如果题目要求画图/做动画: 代码里的注释、id、class、变量名一律不作数(写着某部位不等于画出了它), 先从坐标(路径、圆/椭圆半径、transform 后的位置与大小)
量出主体标志性部位的实际尺寸和关键比例写进 measure, correct 按画出来的东西像不像题目要求给 —— 比例明显不对、会被认成另一种相似的东西, correct 给 3 分以下。非绘图题 measure 写空串。★
最后给 overall(0~100 综合判断)和一句话 verdict。
只输出一个 JSON 对象: {{"measure":"…","correct":n,"complete":n,"follow":n,"quality":n,"overall":n,"verdict":"…"}}

回答:
{a}"""


def cmd_review(a):
    key = open(a.key, encoding="utf-8").read().strip()
    rubric = open(a.rubric, encoding="utf-8").read() if a.rubric else RUBRIC
    done = {(r["id"], r["md5"]) for r in read_jsonl(a.out) if r.get("ok")}
    todo = []
    for r in read_jsonl(a.raw):
        ans = r.get("content") or ""
        h = hashlib.md5(ans.encode("utf-8")).hexdigest()
        if (r["id"], h) not in done: todo.append((r, h, ans))
    print(f"review: {a.raw} 待评 {len(todo)} 份(已评 {len(done)})", flush=True)

    def one(item):
        r, h, ans = item; t0 = time.time()
        if not ans:   # 没写出正文(到上限没收口)= 用户什么也没拿到, 记 0 分不跳过(10-27: 跳过会让这一臂少一道题, 还原率反而虚高)
            return {"id": r["id"], "qh": hashlib.md5(r["q"].encode("utf-8")).hexdigest()[:8], "src": r.get("src", ""), "md5": h, "ok": True, "secs": 0.0, "bytes": 0,
                    "closed": False, **{k: 0 for k in KEYS}, "measure": "", "verdict": f"没写出正文(finish={r.get('finish')}, 思考 {len(r.get('reasoning') or '')} 字)"}
        txt = retry(lambda: api_call(key, rubric.format(q=r["q"], a=ans), a.timeout), f"rv:{r['id']}")
        js = json_block(txt or "", "{", "}")
        ok = isinstance(js, dict) and isinstance(js.get("overall"), (int, float))
        rec = {"id": r["id"], "qh": hashlib.md5(r["q"].encode("utf-8")).hexdigest()[:8], "src": r.get("src", ""), "md5": h, "ok": ok,
               "secs": round(time.time() - t0, 1), "bytes": len(ans.encode("utf-8")), "closed": r.get("closed", True)}
        if ok: rec.update(js)
        else: rec["raw"] = (txt or "")[:300]
        return rec

    with open(a.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.jobs) as ex:
        for fu in as_completed([ex.submit(one, it) for it in todo]):
            rec = fu.result(); f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            print(f"  [{rec['id']}] " + (f"overall {rec['overall']} 正确 {rec.get('correct')} 完整 {rec.get('complete')} 遵循 {rec.get('follow')} 质量 {rec.get('quality')}"
                                         f"{' | ' + str(rec.get('measure'))[:90] if rec.get('measure') else ''} | {str(rec.get('verdict'))[:60]}" if rec["ok"] else f"★评委没回 JSON★ {rec['raw'][:80]!r}"), flush=True)


KEYS = ["overall", "correct", "complete", "follow", "quality"]


def by_item(path):
    """{qh: (src, {键: 该题各份的均值}, 份数)}"""
    g = {}
    for r in read_jsonl(path):
        if r.get("ok") and r.get("qh"): g.setdefault(r["qh"], []).append(r)
    return {h: (R[0].get("src") or R[0]["id"], {k: sum(float(r.get(k) or 0) for r in R) / len(R) for k in KEYS}, len(R)) for h, R in g.items()}


def cmd_summary(a):
    arms = [(s.split("=", 1)[0], by_item(s.split("=", 1)[1])) for s in a.files]
    ref = arms[0][1]; items = sorted(ref, key=lambda h: ref[h][0])
    print("| 题 | " + " | ".join(f"{n} overall(份)" for n, _ in arms) + " |"); print("|---|" + "---|" * len(arms))
    for h in items:
        print(f"| {ref[h][0]} {h} | " + " | ".join(f"{m[h][1]['overall']:.0f}({m[h][2]})" if h in m else "—" for _, m in arms) + " |")
    out = {}
    common = [h for h in items if all(h in m for _, m in arms)]   # 只在每臂都评到的题上算, 题数不同的臂不混比
    for n, m in arms:
        hs = common
        out[n] = {"items": len(hs), "overall": sum(m[h][1]["overall"] for h in hs) / max(1, len(hs)),
                  "ratio": 100 * sum(m[h][1]["overall"] / max(1e-9, ref[h][1]["overall"]) for h in hs) / max(1, len(hs))}
        for k in KEYS[1:]: out[n][k] = sum(m[h][1][k] for h in hs) / max(1, len(hs))
    print("| 臂 | 题数 | overall 均 | 逐题对官方还原率 | " + " | ".join(KEYS[1:]) + " |"); print("|---|---|---|---|" + "---|" * (len(KEYS) - 1))
    for n, v in out.items(): print(f"| {n} | {v['items']} | {v['overall']:.1f} | {v['ratio']:.1f}% | " + " | ".join(f"{v[k]:.1f}" for k in KEYS[1:]) + " |")
    if a.json:   # 机读一行, 链脚本拿它做去留判断
        print("SUMMARY_JSON " + json.dumps(out, ensure_ascii=False))


# 复盘提示 = 本层怎么学(头) + 共用的事实(身) + 要它回什么(尾)。两层(10-28 用户: "原始模型和后训练专项任务是两回事, 分两块还原"):
#   task = 任务层(出编程题 → 对话里作答); raw = 第一层原始能力(写源文件开头 → 裸续写, 不套对话模板)。门两层同一套(⑤, 照反修 A 段 + wt2 底线)
GATE = """- 编程尺: 30 种语言各 512 个 token 的真实代码(从没进过训练), 学生与原始模型的 Mean KLD 越低越像你; 新的修正件要让编程 KLD 降过 2 个标准误,
  且第一名猜中(Same top)不能明显退(新中旧不中的位 − 旧中新不中的位 < −2√两者不一致的位数 就否决), 才留下, 否则退回上一套;
- wt2: 512 个 token 的通用英文, 新修正件的 Same top 不低于 80% 就行(KLD 只报)。
第一目标: 编程尺 Same top 往 100% 推(学生每一位的第一名都和原始模型一样; 之前最好 84.8%, 现役 {s0:.2f}%, 本轮新修正件 {s1:.2f}%)。每轮在现役那套的基础上接着学。"""
PLAN_HEAD = {
    "task": "你在给一个量化模型安排下一轮自学习(任务层)。它的修正件(段 B)每轮这样学: 你出几道编程题 → 你(官方模型)写出完整解答 → 学生按你逐个 token 的概率分布学一遍 → 用两把尺判:\n"
            "料只能是你现出的新题。",
    "raw": "你在给一个量化模型安排下一轮自学习(第一层: 原始能力)。它的修正件(段 B)每轮这样学: 你给选定的每种语言写几个真实开源仓库风格的源文件开头(1~3 行) →"
           " 你在不套对话模板的裸续写模式下把每个文件接着写下去(最多 {rawtok} 个 token, 写完自己停) → 学生按你逐个 token 的概率分布学一遍 → 用两把尺判:\n"
           "料只能是你现写的新文件; 两把尺本身就是裸文档(编程尺 = 真实源文件, wt2 = 百科原文), 这一层练的正是这种接续文档的能力。",
}
PLAN_BODY = """
现役在编程尺每种语言上的 KLD(高 = 离你远)和本轮新修正件对现役的变化(负 = 变好, ± 是标准误;
每种语言只有 512 位, 变化不到 2 个标准误的分不清是学到了还是抽样起伏):
{langtab}
本轮练了的语言平均变化 {dplan}, 没练的语言平均变化 {drest}。

本轮({tag}): {qlabel}
{qs}
训练参数 {params}; 训练过程: {tstat}; 训练料自己的 KLD {k0} → {k1}; 门: {verdict}({why})
用时: {times}
编程尺总 KLD {c0:.5f} → {c1:.5f}(Δ {cd:+.5f} ± {cse:.5f}); wt2 {w0:.5f} → {w1:.5f}(Δ {wd:+.5f} ± {wse:.5f})

以前各轮(计划 → 结果):
{hist}

官方评委对现役自己写的编程留出题的评语: {review}

量化缺口落在哪类 token 上(编程尺是现役对原件, 本轮料是第 0 步对你):
{classes}

{traindoc}
{note}时间预算: 一轮从出料到判完、复盘完要在 {budget} 分钟内。训练时长大致跟 料的总 token 数 × epochs 成正比(料按 maxlen 位一窗切, batch 窗一步); 判决每轮固定。
按上面的用时估下一轮总时长, 填进 est_min。
"""
# 训练参数也由复盘定(10-28 用户: "复盘不能仅仅限于训练语料, 训练参数也很重要, 把参数也要总结提出优化, 脚本里面动态获取这些参数")。
#   只放现有代码真支持、改了有意义的; 不放的写明为什么, 让它知道边界: 秩 = 体积(铁律: 加体积是不得已), 层子集 = 训练时没练的层段 B 不在前向里(要先改代码),
#   clip = 实测 gnorm ~0.15 远低于 1.0 从没触发。键: (类型, 下限, 上限, 可选值, 含义与已知证据)
TRAIN_SPEC = {
    "lr":      (float, 5e-6, 5e-5, None, "学习率(Adam 每步走的距离只看它不看梯度大小; 10-28 a001 lr 2e-4 把训练料自己的 KLD 推高 29% 回退, 2e-5 同料过门)"),
    "epochs":  (int, 1, 3, None, "同一份料过几遍(历来都是 1 遍; 多遍 = 同料多学, 训练料 KLD 会更低, 但可能背料)"),
    "batch":   (int, 1, 8, None, "每步几窗(梯度在窗间累加; 窗越多每步方向越稳、步数越少)"),
    "maxlen":  (int, 512, 2048, (512, 1024, 2048), "一窗多长(位; 长窗每位看到的上文更全、每步更慢; 编程尺每种语言是 512 位一窗)"),
    "gap":     (float, 0.0, 0.9, None, "缺口加权分位: 0 = 每位等权; Q > 0 = 第 0 步 KL 在 Q 分位以上的位权重 1、其余 rest(10-28 a011 同料 rest 0/0.3/1 编程 +2.9/+1.4/−0.1 SE —— 只练缺口位反而差)"),
    "rest":    (float, 0.0, 1.0, None, "gap > 0 时缺口以外的位的权重"),
}
TRAIN_RAW = {"raw_tok": (int, 512, 4096, None, "第一层每个文件续写的上限位数(官方写完会自己停; 历来 1536, 续写中位 250~400 位)")}
TRAIN_FIXED = "秩 64(加秩 = 加体积, 不放); 训练全部 40 层(层子集现有代码不支持); clip 1.0(gnorm ~0.15 从没触发)"


def train_spec(layer):
    return dict(TRAIN_SPEC, **(TRAIN_RAW if layer == "raw" else {}))


def train_doc(layer, cur):
    return ("训练参数你也要一起定(写进 train 对象; 现役这轮用的值在上面\"训练参数\"里):\n"
            + "\n".join(f"  {k}: {d}; 范围 {('/'.join(map(str, ch)) if ch else f'{lo:g}~{hi:g}')}" for k, (_, lo, hi, ch, d) in train_spec(layer).items())
            + f"\n  固定不调: {TRAIN_FIXED}。没理由改的参数就照现在的值写回去。")


def check_train(layer, tr):
    """计划里的 train 对象逐项核范围; 返回错处列表(空 = 合格)"""
    if not isinstance(tr, dict): return ["train 要是 JSON 对象"]
    bad = []
    for k, (ty, lo, hi, ch, _) in train_spec(layer).items():
        v = tr.get(k)
        ok = isinstance(v, (int, float)) and not isinstance(v, bool) and (ty is float or float(v).is_integer()) and lo <= v <= hi and (ch is None or v in ch)
        if not ok: bad.append(f"train.{k} 要在 {('/'.join(map(str, ch)) if ch else f'{lo:g}~{hi:g}')}, 你给的是 {v}")
    return bad


def train_stat(train_log):
    """训练过程的实数: 步数、每步秒数、gnorm 中位与最大(lr 合不合适看它; clip 1.0)"""
    if not train_log or not os.path.exists(train_log): return "(没记)"
    st = [ln.split() for ln in open(train_log) if ln.startswith("step ") and " gnorm " in ln]
    if not st: return "(没记)"
    g = sorted(float(x[x.index("gnorm") + 1]) for x in st); t = float(st[-1][st[-1].index("t") + 1])
    return f"{len(st)} 步, 每步 {t / len(st):.1f} 秒, gnorm 中位 {g[len(g) // 2]:.3g} 最大 {g[-1]:.3g}"
PLAN_TAIL = {
    "task": """请定下一轮, 只输出一个 JSON 对象:
{{"langs": [从上面 30 种里选 1~30 种, 写表里的名字], "q": 题数 4~32, "train": {{训练参数, 见上}}, "est_min": 你估的下一轮总分钟数, "brief": "给出题人的一段话: 这些题该练什么样的代码(不要点名具体题目)", "why": "改了什么、为什么会变好, 依据上面哪些数字"}}""",
    "raw": """请定下一轮, 只输出一个 JSON 对象:
{{"langs": [从上面 30 种里选 1~30 种, 写表里的名字], "q": 每种语言写几个文件开头 1~8, "train": {{训练参数, 见上}}, "est_min": 你估的下一轮总分钟数, "brief": "给写开头的人的一段话: 这些文件该是什么样的项目、模块、规模(不要点名具体文件)", "why": "改了什么、为什么会变好, 依据上面哪些数字"}}""",
}
QRANGE = {"task": (4, 32), "raw": (1, 8)}   # task = 题数; raw = 每种语言的开头数(总数 = q × 语言数)


def round_times(times_path, train_log):
    """本轮各段用时(<work>/times.tsv: 一行"段名 起始秒", 末行 end)与训练细账(train.log 的 step 行: 第几步、累计秒; epoch 行: 行数)。
    给复盘看的是实数: 各段分钟、训练步数与每步秒数、训练料行数; 拿不到的段就不写, 不估。返回 (给人看的一句, 总分钟或 None)"""
    parts, total = [], None
    if times_path and os.path.exists(times_path):
        T = [ln.split() for ln in open(times_path) if len(ln.split()) == 2]
        segs = [(T[i][0], (int(T[i + 1][1]) - int(T[i][1])) / 60) for i in range(len(T) - 1)]
        parts += [f"{k} {m:.1f} 分" for k, m in segs]
        if len(T) >= 2: total = (int(T[-1][1]) - int(T[0][1])) / 60; parts.append(f"合计 {total:.1f} 分")
    if train_log and os.path.exists(train_log):
        steps = [ln.split() for ln in open(train_log) if ln.startswith("step ") and " t " in ln]
        if steps:
            n, t = int(steps[-1][1]), float(steps[-1][steps[-1].index("t") + 1])
            parts.append(f"训练 {n} 步共 {t / 60:.1f} 分(每步 {t / n:.1f} 秒, 不含装载与首末评估)")
    return ("; ".join(parts) or "(没记)"), total


def cmd_plan(a):
    """官方复盘(10-28 用户: "飞轮没有转动起来 …… 下一次是需要什么语料, 调整什么参数, 否则下一次训练其实还是随机取料"):
    把本轮的事实(编程尺分语言 KLD 与本轮变化、门、训练料 KLD、题目、历史各轮、评委评语)交给官方 API, 由它定下一轮的出题语言 / 题数 / 学习率。
    这里只摆事实和校验它的回答, 不替它判: 语言必须在编程尺的 30 种里、题数 4~32(10-28 放宽: 一轮只练 ≤8 种语言时, 没练到的二十几种每轮被带偏 +0.0013~+0.0028, 把练到的 −0.0024~−0.0032 抵掉)、学习率 5e-6~5e-5, 越界或不是 JSON 就把错处告诉它重答, 两次都不对就停。
    产物: --out(计划) 与 --facts(本轮事实, 下一轮复盘读它当历史)。"""
    key = open(a.key, encoding="utf-8").read().strip()
    tab = [ln.split() for ln in open(a.langtab, encoding="utf-8") if ln.strip()]
    langs = [t[0] for t in tab]
    langtab = "\n".join(f"  {t[0]:<20} 现役 {float(t[1]):.4f}  本轮 {float(t[3]):+.4f} ± {float(t[4]):.4f}" for t in tab)
    c, w = json.loads(a.code_json), json.loads(a.wt2_json)
    qs = [r["q"] for r in read_jsonl(a.q)]
    hist = []
    for p in sorted(a.history or []):
        if os.path.abspath(p) == os.path.abspath(a.facts) or not os.path.exists(p): continue
        h = json.load(open(p, encoding="utf-8"))
        hist.append(f"  {h['tag']}: 计划语言 {','.join(h.get('langs') or ['(没定, 随机出题)'])}, {h['nq']} 题, lr {h['lr']} → {h['verdict']}; 编程 Δ {h['cd']:+.5f} ± {h['cse']:.5f}, wt2 Δ {h['wd']:+.5f} ± {h['wse']:.5f}"
                    + (f"; 计划语言上平均 Δ {h['dplan']:+.4f}" if h.get("dplan") is not None else "")
                    + (f", 其余语言 {h['drest']:+.4f}" if h.get("drest") is not None else "")   # 30 种全练(a010)就没有"其余", 是 None
                    + (f"; 训练参数 {json.dumps(h['train'], ensure_ascii=False)}" if h.get("train") else "")
                    + (f"; 训练过程 {h['tstat']}" if h.get("tstat") else "")
                    + (f"; 官方解答 {h['tok']} token; 用时 {h['times']}" if h.get("times") else ""))
    rv = [r for r in read_jsonl(a.review) if r.get("ok")] if a.review and os.path.exists(a.review) else []
    review = "; ".join(f"overall {r['overall']}, {r.get('verdict')}" for r in rv) or "(没有)"
    k0, k1 = a.train.split()
    times, _ = round_times(a.times, a.train_log)
    R = [r for r in read_jsonl(a.raw) if r.get("ok")] if a.raw and os.path.exists(a.raw) else []
    th, ct = (sum(len((r.get("lp") or {}).get(k) or []) for r in R) for k in ("reasoning_content", "content"))
    tok = th + ct
    # 料的成分(10-28 a001~a003: 正文只占 11~13%, 其余是思考段的文字, 编程尺量的却是原始代码文件) —— 摆给它看, 不替它下结论
    times = (f"官方续写共 {tok} token({len(R)} 个文件); " if a.layer == "raw" else f"官方解答共 {tok} token(思考段 {th} / 正文 {ct}, 正文占 {100 * ct / max(1, tok):.1f}%); ") + times
    mine = json.load(open(a.plan_in, encoding="utf-8")) if a.plan_in and os.path.exists(a.plan_in) else {}
    # 本轮练了哪些语言: 第一层按本轮实际出的开头数(每个开头记着 src = 语言), 不看计划 —— 10-28 o001~o004 没有计划、30 种全练,
    #   按计划算成了"(没定, 随机出题)", 官方复盘看不出"全练四轮全过、收窄那轮回退", o005 后又给了更窄的计划。任务层的题 src 是题族, 仍按计划
    used = sorted({r.get("src") for r in read_jsonl(a.q) if r.get("src") in langs}) if a.layer == "raw" else (mine.get("langs") or [])
    pl = set(used); dp = [float(t[3]) for t in tab if t[0] in pl]; dr = [float(t[3]) for t in tab if t[0] not in pl]
    fmt = lambda v: f"{sum(v) / len(v):+.4f}({len(v)} 种)" if v else "(这轮没按计划选语言)"
    L0 = a.layer
    head = PLAN_HEAD[L0].format(rawtok=a.raw_tok) + "\n" + GATE.format(s0=c["same0"], s1=c["same1"])
    cur = json.loads(a.params) if a.params else {"lr": float(a.lr)}; tstat = train_stat(a.train_log)
    prompt = (head + PLAN_BODY + PLAN_TAIL[L0]).format(params=json.dumps(cur, ensure_ascii=False), tstat=tstat, traindoc=train_doc(L0, cur),langtab=langtab, tag=a.tag, qs="\n".join(f"  - {q[:200]!r}" if L0 == "raw" else f"  - {q[:200]}" for q in qs[:40]),
                         qlabel="源文件开头" if L0 == "raw" else "题目", k0=k0, k1=k1, verdict=a.verdict, why=a.why,
                         c0=c["kld0"], c1=c["kld1"], cd=c["dkld"], cse=c["dkld_se"], w0=w["kld0"], w1=w["kld1"], wd=w["dkld"], wse=w["dkld_se"],
                         hist="\n".join(hist) or "  (这是第一轮有计划的)", review=review, times=times, budget=a.budget,
                         dplan=fmt(dp), drest=fmt(dr) if dr else ("(全部语言都练了, 没有没练的)" if dp else "—"), note=(a.note + "\n") if a.note else "",
                         classes=open(a.classes, encoding="utf-8").read().rstrip() if a.classes and os.path.exists(a.classes) else "  (没有)")
    plan = None; ask = prompt
    for _ in range(2):
        js = json_block(retry(lambda: api_call(key, ask, a.timeout), "plan") or "", "{", "}")
        bad = [] if isinstance(js, dict) else ["不是 JSON 对象"]
        if not bad:
            L = js.get("langs"); q = js.get("q")
            if not (isinstance(L, list) and 1 <= len(L) <= len(langs) and all(x in langs for x in L)): bad.append(f"langs 要 1~{len(langs)} 个、都在表里, 你给的是 {L}")
            qa, qb = QRANGE[L0]
            if not (isinstance(q, int) and qa <= q <= qb): bad.append(f"q 要 {qa}~{qb} 的整数, 你给的是 {q}")
            bad += check_train(L0, js.get("train"))
            if not isinstance(js.get("brief"), str) or not isinstance(js.get("why"), str): bad.append("brief / why 要是字符串")
            em = js.get("est_min")
            if not (isinstance(em, (int, float)) and 0 < em <= a.budget): bad.append(f"est_min 要是你估的总分钟数且不超过预算 {a.budget}, 你给的是 {em}")
        if not bad: plan = js; break
        ask = prompt + "\n\n你上一次的回答不合格: " + "; ".join(bad) + "。按格式重答。"
    # 本轮事实先落盘(计划没出来也留着, 下一轮复盘照样当历史读)
    facts ={"tag": a.tag, "layer": a.layer, "langs": used or None, "nq": len(qs), "lr": a.lr, "verdict": a.verdict, "cd": c["dkld"], "cse": c["dkld_se"], "wd": w["dkld"], "wse": w["dkld_se"],
             "dplan": sum(dp) / len(dp) if dp else None, "drest": sum(dr) / len(dr) if dr and dp else None, "tok": tok, "times": times.split("; ", 1)[1],
             "est_min": mine.get("est_min"), "train": cur, "tstat": tstat}
    json.dump(facts, open(a.facts, "w", encoding="utf-8"), ensure_ascii=False)
    if not plan: sys.exit("★官方复盘两次都没给出合格的计划★")
    plan["from"] = a.tag; plan["lr"] = plan["train"]["lr"]   # 任务层老脚本读顶层 lr
    json.dump(plan, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"plan → {a.out}: 语言 {', '.join(plan['langs'])}; {plan['q']} {'个开头/种' if a.layer == 'raw' else '题'}; 训练参数 {json.dumps(plan['train'], ensure_ascii=False)}; 它估下一轮 {plan['est_min']} 分(预算 {a.budget})\n"
          f"  本轮用时: {times}\n  出题方向: {plan['brief']}\n  依据: {plan['why']}")


def tok_class(s):
    """一个 token 的文字归哪类(只看字面, 不看语言): 标识符/词、标点符号、空白缩进、数字、其他"""
    if s.strip() == "": return "空白缩进"
    if re.fullmatch(r"\s*[A-Za-z_][A-Za-z0-9_]*", s): return "标识符/词"
    if re.fullmatch(r"\s*\d+", s): return "数字"
    if re.fullmatch(r"\s*[^\w\s]+", s): return "标点符号"
    return "其他"


def class_table(tok, pairs):
    """[(目标 id, KL)] → {类: (占全部位, 占 KL 前 10% 的位, 占总 KL)}; 前 10% 按 KL 从大到小数"""
    n = len(pairs); tot = sum(k for _, k in pairs) or 1e-12
    top = set(sorted(range(n), key=lambda i: -pairs[i][1])[: max(1, n // 10)])
    C = [tok_class(tok.decode([t])) for t, _ in pairs]; out = {}
    for c in sorted(set(C)):
        P = [i for i in range(n) if C[i] == c]
        out[c] = (len(P) / n, sum(1 for i in P if i in top) / len(top), sum(pairs[i][1] for i in P) / tot)
    return out, sum(pairs[i][1] for i in top) / tot


def cmd_tokclass(a):
    """量化缺口落在哪类 token 上: 编程尺(现役 rows 第 2 列 KLD, 第 p 行预测 ids[p+1])与本轮料(--restore-b 落的 kl0.tsv)各一张表, 写 --out 给 plan 读。
    为什么(10-28): 尺上 KL 前 10% 的位 68% 是标识符/词, 料是题 + 短答案, 标识符是官方自己起的通用名, 练不到这块 —— 料的缺口和尺的缺口对不上, 修了也搬不过去"""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.hf, "tokenizer.json"))
    ids = [int(x) for x in open(a.ruler_ids).read().split()]
    kl = [float(ln.split()[1]) for ln in open(a.ruler_rows) if ln.strip()]
    ruler = [(ids[p + 1], kl[p]) for p in range(min(len(kl), len(ids) - 1))]
    tabs = [("编程尺(现役, 对原件)", ruler)]
    if a.mat and os.path.exists(a.mat):
        tabs.append(("本轮料(第 0 步, 对官方)", [(int(x), float(y)) for x, y in (ln.split() for ln in open(a.mat) if ln.strip())]))
    lines = []
    for name, pairs in tabs:
        t, topshare = class_table(tok, pairs)
        lines.append(f"{name}: {len(pairs)} 位, KL 前 10% 的位占总 KL {100 * topshare:.1f}%")
        lines += [f"  {c:<6} 占全部位 {100 * v[0]:5.1f}% / 占 KL 前 10% 的位 {100 * v[1]:5.1f}% / 占总 KL {100 * v[2]:5.1f}%" for c, v in t.items()]
    open(a.out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("questions"); q.add_argument("--key", required=True); q.add_argument("--round", type=int, required=True); q.add_argument("--n", type=int, default=4)
    q.add_argument("--out", required=True); q.add_argument("--seen", nargs="*"); q.add_argument("--ban", nargs="*"); q.add_argument("--families", default=""); q.add_argument("--prefix", default="")
    q.add_argument("--timeout", type=int, default=600); q.add_argument("--langs", nargs="*"); q.add_argument("--brief", default="")
    p = sp.add_parser("plan"); p.add_argument("--key", required=True); p.add_argument("--langtab", required=True); p.add_argument("--code-json", required=True)
    p.add_argument("--wt2-json", required=True); p.add_argument("--q", required=True); p.add_argument("--train", required=True); p.add_argument("--lr", required=True)
    p.add_argument("--tag", required=True); p.add_argument("--verdict", required=True); p.add_argument("--why", default=""); p.add_argument("--review")
    p.add_argument("--plan-in"); p.add_argument("--history", nargs="*"); p.add_argument("--facts", required=True); p.add_argument("--out", required=True)
    p.add_argument("--timeout", type=int, default=900); p.add_argument("--times"); p.add_argument("--train-log"); p.add_argument("--raw")
    p.add_argument("--classes", help="tokclass 写的表: 缺口落在哪类 token 上(尺 / 料)")
    p.add_argument("--note", default="", help="下一轮和本轮不一样的地方(调用方给的事实, 原样摆给它)")
    p.add_argument("--budget", type=int, required=True, help="一轮总分钟数的上限(调用方给, 计划的 est_min 不许超)")
    p.add_argument("--layer", choices=["task", "raw"], default="task", help="task = 任务层(出题 → 对话作答); raw = 第一层原始能力(开头 → 裸续写)")
    p.add_argument("--raw-tok", type=int, default=1536, help="raw 层每个文件续写的上限位数(摆给它看)")
    p.add_argument("--params", default="", help="本轮实际用的训练参数(JSON, 脚本给; 键见 TRAIN_SPEC)")
    r = sp.add_parser("review"); r.add_argument("--key", required=True); r.add_argument("--raw", required=True); r.add_argument("--out", required=True)
    r.add_argument("--rubric"); r.add_argument("--jobs", type=int, default=4); r.add_argument("--timeout", type=int, default=900)
    t = sp.add_parser("tokclass"); t.add_argument("--hf", required=True); t.add_argument("--ruler-ids", required=True); t.add_argument("--ruler-rows", required=True)
    t.add_argument("--mat"); t.add_argument("--out", required=True)
    s = sp.add_parser("summary"); s.add_argument("files", nargs="+"); s.add_argument("--json", action="store_true")
    sd = sp.add_parser("seeds"); sd.add_argument("--key", required=True); sd.add_argument("--langs", nargs="+", required=True); sd.add_argument("--n", type=int, default=1)
    sd.add_argument("--out", required=True); sd.add_argument("--ban", nargs="*"); sd.add_argument("--prefix", default="raw"); sd.add_argument("--timeout", type=int, default=600); sd.add_argument("--brief", default="", help="第一层计划(plan --layer raw)给的方向")
    a = ap.parse_args()
    {"questions": cmd_questions, "review": cmd_review, "summary": cmd_summary, "plan": cmd_plan, "tokclass": cmd_tokclass, "seeds": cmd_seeds}[a.cmd](a)


if __name__ == "__main__":
    main()
