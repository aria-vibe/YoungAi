#!/bin/bash
# restore_B2_mix_spark.sh — 还原侧车 段 B 第四版的料链(2026-10-26 晚): 6 族按 token 等份的官方轨迹 + 分支续写料, 再交给 restore_B2_api_spark.sh 训。
#   要修的三件事(fable5 10-26 晚): ① 思考收不了口 = 曝光偏差 → 分支续写(学生前缀零梯度 + 官方续写当靶, api_branch.py)
#     ② 编程/wt2 退 = 图形 token 份额 78%→100% 的零和 → 按 token 等份配 6 族(编程/数学/数据/写作/通用决策/图形), 图形只占它那一份
#     ③ 细节的规划层靠官方各族轨迹里"每个部件带数字规划"的风格跨域迁移 —— 这一条是要实测的赌注, 不保证。
# 五步(任一步非 0 停, 尾行 B2MIX_EXIT <rc>; --from/--to 分步; 步 ② 要模型服务开着, 步 ⑤ 自己停模型):
#   ① 题 + 官方轨迹: think_trace_collect.py questions --set 四族 → mix_q.jsonl; api_teacher.py collect(边生成边拿 top-20)→ mix_lp_raw.jsonl
#   ② 学生生成: 每族各取 N 题(--student-n, 缺省 6; 通用从 think198、图形从 api27 的题里取), think_freegen_probe.py 对着服务生成 → student_v4.jsonl
#   ③ 分支续写: api_branch.py branch(每 4096 位一个分支点, 官方续 ≤8192 位)→ branch_v4_raw.jsonl; rows → restore_clean/branch_v4.jsonl
#   ④ 料: mix 整篇(chunks --max-chunks 1) + 图形 api27 前 G 篇(--graphics-n, 缺省 12) + think198 全部 + 分支行 → restore_clean/restore_v4.jsonl;
#         教师原料 = 四份 lp raw 拼成 v4_lp_raw.jsonl; api_branch.py share 打 token 份额账; 可选 --audit-ids 对判决 ids 做 16-gram 泄题审计(只报不拦)
#   ⑤ restore_B2_api_spark.sh --from 3 --to 4 --cfg restore_B2ctx_v4.cfg --teacher-raw v4_lp_raw.jsonl(③ 训 + ④ 选轮并包 + 段 C 五指标)
# 用法(spark 仓库根): nohup gguf-tools/scripts/restore_B2_mix_spark.sh [--from 1 --to 5] [--student-n 6] [--graphics-n 12] [--student-max-tokens 24576] [--audit-ids f.ids …] > /tmp/bugmd/b2mix_<时间>.log 2>&1 &
#   1 小时版(10-26 晚用户: "时间长不代表指标好, 压到 1 小时"; 训练 39 s/步 × 每步 4 段, 1 小时 = 料 ~200 段): --cfg restore_B2ctx_v4q.cfg(doc_tail 4)
#     --mix-n 2(每族官方 2 题) --student-n 1 --student-max-tokens 12288 --branch-max-tokens 4096 --graphics-n 6 --think-n 40 --reuse-teacher gguf/v41/posttrain/restore-B2ctx-v3
#     ① 与 ② 本来就并行(① 只打 API, ② 只占本地服务); --reuse-teacher 把 v3 的保持料教师表(part.001, 63 条, 内容键配)拷过来省 842 s。
#   ★1 小时版的坑(10-26 夜审料): --branch-max-tokens 4096 把官方续写截在思考中段, 9 条分支 7 条没收口 = 教"接着想", 与要修的病反着来;
#     --student-max-tokens 12288 让分支点到不了 2 万位后真正收不了口的地方。分支料的续写上限要够官方自己收口(32768), 总量用 --branch-every 控。
#   ②③ 都按 id 续跑: 换学生(侧车变了)或换上限重跑前, 把 student_v4.jsonl / branch_v4_raw.jsonl 挪开, 否则旧条原样复用。
#   ⓪(10-27, --from 0): --heldout-q <题集> 采泛化留出集的官方轨迹 → --heldout-dir(缺省 gguf/go-onebit/code/svgheld)的 ids + list.txt(已有就跳过);
#     --baseline 先拿现役侧车跑一遍段 C(只判不并包), rows 另存 *.base.rows —— 新一趟要在留出集上比它好才算泛化。留出 ids 自动进 ④ 的泄题审计。
#   计划段版(10-27 晚, --plan-head N --branch-cuts 128,384,…): 料只要思考开头 —— 官方轨迹截前 N 位(api_branch.py head) + 学生开头的分支续写。
#     为什么: FP 探针 10-27 —— 官方写"鹈鹕(身体、长脖、"处 FP 给"喉"0.88、量化 0.00; 学生写"橙色喙"处 FP 只给 0.06~0.31 而学生 0.35~0.91;
#     画成什么样定在开头几百位的部件清单, doc_tail(末段)和每 8192 位一刀的分支都碰不到。配 cfg 的 init= 从现役 ③ 续训, 末段学到的不丢。
# 出错会怎样: 服务没起 → ② 连接失败直接停(不回退到别的采样); API 失败的题 ok:false 续跑只补它们; 料里任何一行训练器解析不了整趟停(core_ptrain_data.c)。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
FROM=1; TO=5; SN=6; GN=12; SMAX=24576; AUDIT=(); MIXN=0; THINKN=0; BMAX=8192; BEVERY=4096; REUSE=""; OCFG=""; AGN=0; SETALL=()
HQ=""; HD=gguf/go-onebit/code/svgheld; BASE=0; PLANH=0; BCUTS=""
while [ $# -gt 0 ]; do case "$1" in
    --from) FROM="${2:?}"; shift 2;; --to) TO="${2:?}"; shift 2;; --student-n) SN="${2:?}"; shift 2;; --graphics-n) GN="${2:?}"; shift 2;;
    --student-max-tokens) SMAX="${2:?}"; shift 2;; --mix-n) MIXN="${2:?}"; shift 2;; --think-n) THINKN="${2:?}"; shift 2;; --branch-max-tokens) BMAX="${2:?}"; shift 2;;
    --reuse-teacher) REUSE="${2:?}"; shift 2;; --cfg) OCFG="${2:?}"; shift 2;; --agent-n) AGN="${2:?}"; shift 2;; --branch-every) BEVERY="${2:?}"; shift 2;;
    --set-all) SETALL+=("${2:?}"); shift 2;;   # 族名=题集(10-27): 再加一族官方轨迹, 整族都采, 不受 --mix-n 抽题(SVG 角色插画族)
    --heldout-q) HQ="${2:?}"; shift 2;; --heldout-dir) HD="${2:?}"; shift 2;; --baseline) BASE=1; shift;;
    --plan-head) PLANH="${2:?}"; shift 2;; --branch-cuts) BCUTS="${2:?}"; shift 2;;
    --audit-ids) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do AUDIT+=("$1"); shift; done;;
    *) echo "★不认识的参数 $1★"; exit 2;; esac; done
stage(){ [ "$FROM" -le "$1" ] && [ "$TO" -ge "$1" ]; }
PY="$HOME/v41env/bin/python"; HF=hf/DeepSeek-V4.1-Flash-1017; PD=gguf-tools/data/posttrain; RC=$PD/restore_clean; RO=gguf/go-onebit/restore
KEY=$PD/deepseek_api.key; CFG=${OCFG:-$PD/restore_B2ctx_v4.cfg}
M(){ echo "[b2mix $(date '+%m-%d %H:%M:%S')] $*"; }
die(){ M "★$*★"; echo "B2MIX_EXIT 1"; exit 1; }
for f in "$KEY" "$HF/tokenizer.json" "$CFG" "$RC/api27_v3.jsonl" "$RC/think198_v3.jsonl" "$RO/api27_lp_raw.jsonl" "$RO/think198_lp_raw.jsonl"; do [ -s "$f" ] || die "缺 $f"; done
for s in code math data writing; do [ -s "$PD/mix_q_$s.txt" ] || die "缺题集 $PD/mix_q_$s.txt"; done

stage1q(){   # 题集(同步: ② 的学生题单要读它, 不能和 ① 的采集一起丢后台)
    M "① 题集 → $PD/mix_q.jsonl$([ "$MIXN" -gt 0 ] && echo "(每族抽 $MIXN 题)")"
    "$PY" gguf-tools/scripts/think_trace_collect.py questions --set "code=$PD/mix_q_code.txt" "math=$PD/mix_q_math.txt" "data=$PD/mix_q_data.txt" "writing=$PD/mix_q_writing.txt" "${SETALL[@]}" --out "$PD/mix_q.jsonl" || return 1
    if [ "$MIXN" -gt 0 ]; then   # 每族等距抽 N 题(不挑内容), 1 小时版用; --set-all 的族整族留着
        "$PY" - "$MIXN" "$PD/mix_q.jsonl" "${SETALL[@]%%=*}" <<'EOF2' || return 1
import json, sys
n = int(sys.argv[1]); keep = set(sys.argv[3:]); fam = {}
for ln in open(sys.argv[2], encoding="utf-8"): r = json.loads(ln); fam.setdefault(r["src"], []).append(r)
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for k, rows in fam.items():
        for r in (rows if k in keep else rows[::max(1, len(rows) // n)][:n]): f.write(json.dumps(r, ensure_ascii=False) + "\n")
EOF2
    fi
}
stage1c(){   # 官方轨迹采集(只打 API, 后台跑, 与 ② 并行)
    M "① 官方轨迹(带 top-20, 4 路, 上限 65536) → $RO/mix_lp_raw.jsonl"
    "$PY" gguf-tools/scripts/api_teacher.py collect --key "$KEY" --questions "$PD/mix_q.jsonl" --out "$RO/mix_lp_raw.jsonl" --hf "$HF" --max-tokens 65536 --jobs 4 || return 1
    nok=$("$PY" -c "import json,sys; print(sum(1 for l in open('$RO/mix_lp_raw.jsonl') if json.loads(l).get('ok')))"); M "① 完成: ok $nok / $(wc -l < "$PD/mix_q.jsonl") 题"
}
if stage 0; then   # ⓪ 泛化留出尺(10-27 用户: "能不能泛化作为重要"): 训练料里一道都没有的题, 官方轨迹做成段 C 的 ids; 再量现役侧车的基线
    if [ -n "$HQ" ] && [ ! -s "$HD/list.txt" ]; then
        M "⓪ 留出集 $HQ → 官方轨迹 → $HD/*.ids"
        "$PY" gguf-tools/scripts/think_trace_collect.py questions --set "$(basename "$HD")=$HQ" --out "$RO/heldout_q.jsonl" || die "留出题集失败"
        "$PY" gguf-tools/scripts/think_trace_collect.py collect --key "$KEY" --questions "$RO/heldout_q.jsonl" --out "$RO/heldout_raw.jsonl" --max-tokens 65536 --jobs 6 || die "留出采集失败"
        "$PY" gguf-tools/scripts/think_trace_collect.py ids --hf "$HF" --raw "$RO/heldout_raw.jsonl" --out-dir "$HD" || die "留出 ids 失败"
    fi
    if [ "$BASE" = 1 ]; then   # 基线 = 现役侧车(还没被这一趟换掉)在鹈鹕设计段 + 留出集上的读数; rows 另存 .base.rows, 这一趟段 C 会盖同名文件
        M "⓪ 基线: 现役侧车只判不并包(restore_sidecar_chain_spark.sh --judge-only)"
        bash gguf-tools/scripts/restore_sidecar_chain_spark.sh --judge-only --cfg "$CFG" --heldout "$HD" || die "基线判决失败"
        n=0; for f in gguf/v41judge/stu_"$(basename "$HD")"_*_amp_*.rows gguf/v41judge/stu_pelican_think_*_amp_*.rows; do [ -s "$f" ] && cp "$f" "${f%.rows}.base.rows" && n=$((n+1)); done
        M "⓪ 基线 rows 另存 $n 份(*.base.rows)"
    fi
fi
[ -s "$HD/list.txt" ] && for f in "$HD"/*.ids; do AUDIT+=("$f"); done   # 留出 ids 自动进泄题审计(④)
S1PID=""
if stage 1; then stage1q || die "题集失败"; stage1c > "$RO/mix_stage1.log" 2>&1 & S1PID=$!; M "① 采集后台跑(日志 $RO/mix_stage1.log), 与 ② 并行"; fi
if stage 2; then
    M "② 学生生成: 每族 $SN 题, 上限 $SMAX, 服务 127.0.0.1:8001"
    # ⓪ 的基线判决停过模型再装回, 装回到 idle 要 ~80 s(/api/models/load 先回 ok 再加载): 等它进 idle 再发, 10 分钟不进就是没装回来
    for _ in $(seq 120); do curl -s --noproxy '*' --max-time 5 http://127.0.0.1:8001/metrics | grep -q '"state":"idle"' && break; sleep 5; done
    curl -s --noproxy '*' --max-time 5 http://127.0.0.1:8001/v1/models >/dev/null || die "模型服务没起(127.0.0.1:8001)"
    "$PY" - "$SN" "$PD/mix_q.jsonl" "$RC/api27_v3.jsonl" "$RO/think198_lp_raw.jsonl" "$PD/student_q.jsonl" <<'EOF' || die "学生题单失败"
import json, sys
n = int(sys.argv[1]); fam = {}
for ln in open(sys.argv[2], encoding="utf-8"):
    r = json.loads(ln); fam.setdefault(r["src"], []).append({"id": r["id"], "q": r["q"]})
U, A = "<｜User｜>", "<｜Assistant｜>"
for ln in open(sys.argv[3], encoding="utf-8"):   # 图形: 训练用的 27 题(text 行里取题)
    t = json.loads(ln).get("text", ""); u, s = t.find(U), t.find(A)
    if 0 <= u < s: fam.setdefault("graphics", []).append({"id": f"graphics_{len(fam.get('graphics', [])):03d}", "q": t[u + len(U):s]})
for ln in open(sys.argv[4], encoding="utf-8"):   # 通用决策: think198 里的 gen_ 题(HumanEval 那 100 题太短, 学生几百位就收口, 不是要修的病)
    r = json.loads(ln)
    if r["id"].startswith("gen_") and r.get("ok"): fam.setdefault("general", []).append({"id": r["id"], "q": r["q"]})
with open(sys.argv[5], "w", encoding="utf-8") as f:
    for k in ["code", "math", "data", "writing", "general", "graphics"] + sorted(set(fam) - {"code", "math", "data", "writing", "general", "graphics"}):   # --set-all 加的族排在后面
        rows = fam.get(k, []); step = max(1, len(rows) // n)
        for r in rows[::step][:n]: f.write(json.dumps(dict(r, src=k), ensure_ascii=False) + "\n")   # 等距取, 不挑内容
        print(f"  {k}: {len(rows)} 题里取 {min(n, len(rows))}")
EOF
    "$PY" gguf-tools/scripts/think_freegen_probe.py --questions "$PD/student_q.jsonl" --n 0 --jobs 2 --max-tokens "$SMAX" --out "$RO/student_v4.jsonl" || die "学生生成失败"
    M "② 完成: $(wc -l < "$RO/student_v4.jsonl") 条学生轨迹, 收口 $("$PY" -c "import json; print(sum(1 for l in open('$RO/student_v4.jsonl') if json.loads(l).get('closed')))")"
fi
if [ -n "$S1PID" ]; then wait "$S1PID" || die "① 失败(看 $RO/mix_stage1.log)"; tail -3 "$RO/mix_stage1.log"; fi
if stage 3; then
    M "③ 分支续写: $([ -n "$BCUTS" ] && echo "切点 $BCUTS" || echo "每 $BEVERY 位一个分支点"), 官方续 ≤$BMAX 位, 4 路"
    "$PY" gguf-tools/scripts/api_branch.py branch --key "$KEY" --student "$RO/student_v4.jsonl" --out "$RO/branch_v4_raw.jsonl" --hf "$HF" --every "$BEVERY" --cuts "$BCUTS" --max-tokens "$BMAX" --jobs 4 || die "branch 失败"
    M "③ 分支收口: $("$PY" -c "import json; R=[json.loads(l) for l in open('$RO/branch_v4_raw.jsonl')]; R=[r for r in R if r.get('ok')]; print(f\"ok {len(R)} 条, 官方续写收口 {sum(1 for r in R if r.get('closed'))} 条\")")"
    "$PY" gguf-tools/scripts/api_branch.py rows --raw "$RO/branch_v4_raw.jsonl" --out "$RC/branch_v4.jsonl" || die "rows 失败"
fi
if stage 4 && [ "$PLANH" -gt 0 ]; then   # 计划段版: 官方各族轨迹的思考前 PLANH 位 + 分支行, 不要整篇
    M "④ 料(计划段): mix + 图形前 $GN 篇 + think198 抽 $THINKN 条的思考前 $PLANH 位 + 分支行 → $RC/restore_v4.jsonl"
    "$PY" gguf-tools/scripts/api_branch.py head --hf "$HF" --n "$PLANH" --out "$RO/head_raw.jsonl" \
        --raw "$RO/mix_lp_raw.jsonl" "$RO/api27_lp_raw.jsonl:$GN" "$RO/think198_lp_raw.jsonl:$THINKN" || die "head 失败"
    "$PY" gguf-tools/scripts/think_trace_collect.py chunks --hf "$HF" --raw "$RO/head_raw.jsonl" --out "$RC/head_v4.jsonl" --max-tok 1000000 --max-chunks 1 || die "chunks 失败"
    cat "$RC/head_v4.jsonl" "$RC/branch_v4.jsonl" > "$RC/restore_v4.jsonl"
    cat "$RO/head_raw.jsonl" "$RO/branch_v4_raw.jsonl" > "$RO/v4_lp_raw.jsonl"
    M "料 $(wc -l < "$RC/restore_v4.jsonl") 行 = 头 $(wc -l < "$RC/head_v4.jsonl") + 分支 $(wc -l < "$RC/branch_v4.jsonl"); 教师原料 $(wc -l < "$RO/v4_lp_raw.jsonl") 条"
elif stage 4; then
    M "④ 料: mix 整篇 + 图形前 $GN 篇 + think198 + 分支行 → $RC/restore_v4.jsonl"
    "$PY" gguf-tools/scripts/think_trace_collect.py chunks --hf "$HF" --raw "$RO/mix_lp_raw.jsonl" --out "$RC/mix_v4.jsonl" --max-tok 1000000 --max-chunks 1 || die "chunks 失败"
    head -n "$GN" "$RC/api27_v3.jsonl" > "$RC/graphics_v4.jsonl"
    if [ "$THINKN" -gt 0 ]; then awk -v n="$THINKN" -v t="$(wc -l < "$RC/think198_v3.jsonl")" 'NR % int(t / n + 0.999) == 1' "$RC/think198_v3.jsonl" | head -n "$THINKN" > "$RC/think_v4.jsonl"; else cp "$RC/think198_v3.jsonl" "$RC/think_v4.jsonl"; fi   # 通用短题等距抽 N 行(wt2 的锚)
    # agent 多轮行(api_agent.py rows 的产物 + 它的 raw)有就并进去: Mac 侧 dsh_agent_collect.sh --ship → spark api_agent.py resample / rows
    AG=(); AGR=(); [ -s "$RC/agent_v4.jsonl" ] && [ -s "$RO/agent_v4_raw.jsonl" ] && { AG=("$RC/agent_v4.jsonl"); AGR=("$RO/agent_v4_raw.jsonl"); }
    if [ "${#AG[@]}" -gt 0 ] && [ "$AGN" -gt 0 ]; then   # --agent-n: 等距抽 N 行(每行要单独预填 ~2 万位前缀, 接不了力, 比接力段贵; 1 小时版 32 行 ≈ 8 步); raw 整份留着, 多出的记录 ptf 不理会
        t=$(wc -l < "${AG[0]}"); awk -v n="$AGN" -v t="$t" 'NR % int(t / n + 0.999) == 1' "${AG[0]}" | head -n "$AGN" > "$RC/agent_v4.pick.jsonl"; AG=("$RC/agent_v4.pick.jsonl")
    fi
    cat "$RC/mix_v4.jsonl" "$RC/graphics_v4.jsonl" "$RC/think_v4.jsonl" "$RC/branch_v4.jsonl" "${AG[@]}" > "$RC/restore_v4.jsonl"
    cat "$RO/mix_lp_raw.jsonl" "$RO/api27_lp_raw.jsonl" "$RO/think198_lp_raw.jsonl" "$RO/branch_v4_raw.jsonl" "${AGR[@]}" > "$RO/v4_lp_raw.jsonl"
    M "料 $(wc -l < "$RC/restore_v4.jsonl") 行 = mix $(wc -l < "$RC/mix_v4.jsonl") + 图形 $(wc -l < "$RC/graphics_v4.jsonl") + think198 $(wc -l < "$RC/think_v4.jsonl") + 分支 $(wc -l < "$RC/branch_v4.jsonl") + agent $([ "${#AG[@]}" -gt 0 ] && wc -l < "${AG[0]}" || echo 0); 教师原料 $(wc -l < "$RO/v4_lp_raw.jsonl") 条"
    "$PY" gguf-tools/scripts/api_branch.py share --hf "$HF" --doc-tail "$(grep '^doc_tail=' "$CFG" | cut -d= -f2)" \
        --raw "mix=$RC/mix_v4.jsonl" "graphics=$RC/graphics_v4.jsonl" "think198=$RC/think_v4.jsonl" "branch=$RC/branch_v4.jsonl" $([ "${#AG[@]}" -gt 0 ] && echo "agent=${AG[0]}")
    if [ "${#AUDIT[@]}" -gt 0 ]; then   # 泄题审计: 料与判决 ids 共享的 16-gram(样板句也会命中, 看具体片段再判; 只报不拦)
        "$PY" - "$HF" "$RC/restore_v4.jsonl" "${AUDIT[@]}" <<'EOF'
import json, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1] + "/tokenizer.json"); NG = 16; jp = {}
for p in sys.argv[3:]:
    ids = [int(x) for x in open(p).read().split()]
    for i in range(len(ids) - NG + 1): jp[tuple(ids[i:i + NG])] = p
hits = {}
for n, ln in enumerate(open(sys.argv[2], encoding="utf-8")):
    r = json.loads(ln); ids = tok.encode(r.get("text", ""), add_special_tokens=False).ids
    gs = {tuple(ids[i:i + NG]) for i in range(len(ids) - NG + 1)} & set(jp)
    if gs: hits[n] = gs
print(f"泄题审计: {len(hits)} 行命中判决 ids 的 {NG}-gram")
for n, gs in list(hits.items())[:8]: g = next(iter(gs)); print(f"  行 {n}: {len(gs)} 个, 如 {tok.decode(list(g))[:70]!r} ← {jp[g]}")
EOF
    fi
fi
if stage 5; then
    OUT=$(grep '^out=' "$CFG" | cut -d= -f2-)
    if [ -n "$REUSE" ]; then   # 保持料的教师表复用(posttrain_fp_spark.sh 按"有 .ptf 的份保留 + 内容键配"接: 拷 000 以外的份, API 那份 000 本趟重打)
        mkdir -p "$OUT"; n=0
        for p in "$REUSE"/teacher.part.[0-9][0-9][0-9]; do [ -s "$p.ptf" ] && [ "${p##*.}" != 000 ] && { cp "$p" "$p.ptf" "$OUT"/; n=$((n + 1)); }; done
        M "复用 $REUSE 的 $n 份教师表(保持料), 本地 FP 教师只补没对上的"
    fi
    M "⑤ 训 + 段 C: restore_B2_api_spark.sh --from 3 --to 4 --cfg $CFG --teacher-raw $RO/v4_lp_raw.jsonl"
    bash gguf-tools/scripts/restore_B2_api_spark.sh --from 3 --to 4 --cfg "$CFG" --teacher-raw "$RO/v4_lp_raw.jsonl" || die "训练链失败"
fi
M "完成"; echo "B2MIX_EXIT 0"
