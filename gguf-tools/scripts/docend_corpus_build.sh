#!/bin/bash
# docend_corpus_build.sh — "文档结尾"判决料(2026-09-30, 用户"先 1"): 整篇文档 + 真正的 EOS, 量量化把"停"这个决策压掉几倍。
#
# 为什么要它: 现役所有拟合料/判决料都是从文档中间切出来的 128~512 token 窗, 一个文档结尾都没有 —— EOS 这种最基本的
#   "停"决策从来没进过任何一份拟合或判决。09-30 在真实 CFO 序列上量到: FP 在"该收口"的位置给 </think> 22%~49%,
#   量化学生只给 1%~6%(引擎与官方代码同权重同数)。这份料回答"这是思考专属还是决策 token 的通病": 拿整篇开源文档
#   按预训练样子 <BOS> 正文 <EOS> 拼起来, 在每个 EOS 位置比 FP 与学生给 EOS 的概率(bugmd_ids_tools.py tokprobe)。
# 料: gguf-tools/data/corpus/{law,med,sci,code}/*.jsonl(一行一篇 {"text"}), 每个子域取 PER_SUB 篇、篇长在 [MIN_TOK, MAX_TOK] token
#   之间的整篇(不截断: 截断的文档没有真结尾)。同一子域按行号等距取, 不挑内容。
# 产物: gguf/go-onebit/docend/<NAME>.ids(一行一个 id; 每篇 = BOS + 正文 + EOS 拼接) + <NAME>.manifest.txt(篇号 来源 行号 起点 EOS位置 篇长)
#   + 给了 JSONL_OUT 就再写一份 {"text": 原文} 一行一篇 —— 同一批篇子直接当后训练 ③ 的 text 料(docs/think_close.md §2 的 A 份)。
# 用法: bash gguf-tools/scripts/docend_corpus_build.sh [PER_SUB=1] [MIN_TOK=300] [MAX_TOK=1200] [OFFSET=0] [NAME=docend] [JSONL_OUT=] [EXCLUDE=]   (spark 本机跑, 分词器走 ~/v41env)
#   OFFSET = 每子域候选里从第几篇起等距取(10-22): 候选 ≈ 8×PER_SUB 篇、步长 ≈ 8, OFFSET 1~7 与 OFFSET 0 不重叠 —— 训练料与留出判决料要两套不同的篇子。
#   EXCLUDE = 逗号分隔的已有 manifest, 这些篇(子域+源行号)不再入候选(10-22 下午实撞: PER_SUB 不同时步长不同, 光靠 OFFSET 错不开,
#     docend_tr2 与判决篇子重了 4 篇 / 与 docend_tr 重了 8 篇; 判决篇子进训练 = 判决失效)。
# 出错会怎样: 某子域凑不够 PER_SUB 篇 → 打 ★ 少收, 不硬停(判决料不要求每域等量); 分词器装不上 → 直接退。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
PER_SUB="${1:-1}"; MIN_TOK="${2:-300}"; MAX_TOK="${3:-1200}"; OFFSET="${4:-0}"; NAME="${5:-docend}"; JSONL_OUT="${6:-}"; EXCLUDE="${7:-}"
OUT="$ROOT/gguf/go-onebit/docend"; mkdir -p "$OUT"
~/v41env/bin/python - "$ROOT" "$OUT" "$PER_SUB" "$MIN_TOK" "$MAX_TOK" "$OFFSET" "$NAME" "$JSONL_OUT" "$EXCLUDE" <<'PYEOF'
import glob, json, os, sys
root, out, per_sub, min_tok, max_tok = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
offset, name, jsonl_out = int(sys.argv[6]), sys.argv[7], sys.argv[8]
excl = set()
for mp in filter(None, sys.argv[9].split(",")):
    for s in open(mp, encoding="utf-8"):
        if s.startswith("#") or not s.strip(): continue
        f = s.split(); excl.add((f[1], int(f[2])))
if excl: print("排除清单: %d 篇(来自 %s)" % (len(excl), sys.argv[9]))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.join(root, "hf/DeepSeek-V4.1-Flash"), trust_remote_code=True)
BOS, EOS = tok.bos_token_id, tok.eos_token_id
assert BOS is not None and EOS is not None, "分词器没有 BOS/EOS"
# 子域清单: 中英各半, 散文为主, 代码只收四种"文件有自然结尾"的
SUBS = ["law/law_case_zh", "law/law_statute_en", "law/law_contract_en",
        "med/med_encyc_zh", "med/med_paper_en", "med/med_case_en",
        "sci/sci_paper_zh", "sci/sci_arxiv_en", "sci/sci_paper_hss_zh",
        "code/code_python", "code/code_markdown", "code/code_shell", "code/code_sql"]
ids_all, manifest, texts, pos = [], [], [], 0
for sub in SUBS:
    path = os.path.join(root, "gguf-tools/data/corpus", sub + ".jsonl")
    if not os.path.exists(path):
        print("★缺 %s, 跳过★" % path); continue
    lines = open(path, encoding="utf-8").read().split("\n")
    cands = []
    for ln, s in enumerate(lines):
        if not s.strip(): continue
        try: text = json.loads(s)["text"]
        except Exception: continue
        if not text.strip() or (sub, ln) in excl: continue
        enc = tok.encode(text, add_special_tokens=False)
        if min_tok <= len(enc) <= max_tok: cands.append((ln, enc, text))
        if len(cands) >= per_sub * 8: break          # 够挑了(等距取 per_sub 篇), 不用把整个子域都分词
    if not cands:
        print("★%s 没有 [%d,%d] token 的整篇★" % (sub, min_tok, max_tok)); continue
    step = max(1, len(cands) // per_sub)
    picked = cands[offset::step][:per_sub]
    if len(picked) < per_sub: print("★%s 只凑到 %d/%d 篇★" % (sub, len(picked), per_sub))
    for ln, enc, text in picked:
        doc = [BOS] + enc + [EOS]
        manifest.append((len(manifest), sub, ln, pos, pos + len(doc) - 1, len(enc)))   # EOS 的绝对位置 = pos+len(doc)-1
        ids_all.extend(doc); pos += len(doc); texts.append(text)
with open(os.path.join(out, name + ".ids"), "w") as f: f.write("\n".join(map(str, ids_all)) + "\n")
with open(os.path.join(out, name + ".manifest.txt"), "w", encoding="utf-8") as f:
    f.write("# 篇号 子域 源行号 起点(BOS位置) EOS位置 正文token数; BOS=%d EOS=%d; 总 token %d; offset %d\n" % (BOS, EOS, len(ids_all), offset))
    for m in manifest: f.write("%d %s %d %d %d %d\n" % m)
if jsonl_out:
    with open(jsonl_out, "w", encoding="utf-8") as f:
        for t in texts: f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    print("text 料: %d 篇 → %s" % (len(texts), jsonl_out))
print("%s.ids: %d 篇, %d token(BOS %d / EOS %d) → %s" % (name, len(manifest), len(ids_all), BOS, EOS, out))
for m in manifest: print("  篇 %2d  %-22s 行 %5d  起 %6d  EOS 位 %6d  正文 %4d tok" % m)
PYEOF
