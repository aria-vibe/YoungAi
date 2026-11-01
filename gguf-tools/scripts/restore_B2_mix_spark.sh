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
# ★自进化一轮(10-27 晚, 用户: "一个小时一轮, 官方 api 当裁判和语料来源, 不要泄题, 对标现在的 b 段训练→加载→官方评审";
#   "以鹈鹕测试为例, 其实是要求整个模型能力的还原, 不要跑偏" —— 尺和去留门都按全族通用能力, 鹈鹕只是行为尺里图形那一道)★
#   --work gguf/go-onebit/evo/rNNN 打开一轮模式: 这一轮的题/轨迹/学生/分支/料/cfg/日志全落在这个目录(id 带轮号, 不和历史续跑串), 只走计划段料(--plan-head)。
#   ⓪ 固定通用留出集 gguf/go-onebit/evo/hold(只建一次): 官方出 6 族各 3 道(永不进料, 之后每轮出题都与它去重)+ 官方完整轨迹;
#      两把官方尺都从它来 —— 分布尺: 整条轨迹当训练器 eval=(泛化留出), 第 0 步(= 现役)与轮末(= 本轮)对官方 top-20 的 KL 逐题配对 + 决策位(</think>/EOS)KL;
#      行为尺: 编程/数学/数据/写作/通用各 1 道 + 鹈鹕 1 道, 模型自己写, 官方评委对每道题打分, 逐题除以官方答案的分 = 还原率(官方参照只评一次)。
#   ① --evo-q N: api_round.py questions 让官方现出各族 N 道新题(与全部历史题 + 留出集去重, 禁词: 鸟/鹈鹕/骑车 + 泛化留出集动物名), 再 api_teacher.py collect 拿带 top-20 的轨迹
#   ② 学生只做这一轮的新题(不掺旧题集) ④ 头料 = 新题 + 旧图形/通用各少量回放(--graphics-n / --think-n, 0 = 不回放); 含禁词的料行整行剔掉
#   ⑤ 从现役侧车续训: 训练前把侧车根目录的 amp 拷成 <work>/init 当 init=(cfg 模板默认 restore_B2plan_v6.cfg, 只换 data/out/init, 加 eval=); 段 C 只判编程判决份 + wt2
#   ⑥ 行为尺 + 去留: 留 = 分布尺(泛化留出 gen_kl)轮末低于第 0 步 且 决策位 KL 不升 且 编程判决份 Same top 掉不过 0.3 pp(15360 位 1 SE) 且 行为尺还原率掉不过 5 个点
#      (n=6 道单份, 评委噪声未量, 5 点是暂定容差); 否则侧车换回 <work>/init 并重装模型(新 amp 挪 <work>/amp_rejected, 不删)。
#      细节还原度 = 编程判决份按 FP 置信度分桶(p>0.9 / 0.5~0.9 两桶的错 top-1), 现役与本轮的 rows 按侧车指纹存 gguf/go-onebit/evo/rows/。
#   现役的行为分按侧车 amp 指纹缓存在 gguf/go-onebit/evo/beh/<md5>.*(没有就在 ② 后生成); 现役的编程/wt2 读数从 gguf/go-onebit/evo/ledger.tsv 按指纹查,
#     没有就要 --base-log <判过这套侧车的段 C 日志>(取第 2 / 4 个 Same top = 编程 / wt2 的侧车臂)。每轮结果追加进 ledger.tsv。
#   ⑦ 清理(10-27 用户: "每次跑完记得清理临时文件, 不然空间不够"; 单跑: --work <目录> --from 7 --to 7): 只删本轮产生的、能重算或别处有副本的 ——
#      段 C 两臂的学生 logits 本体与逐层转储(编程 7.9 GB/臂; .rows 与日志留着)、本轮教师表(teacher.*)、本轮的拼接料、与 init 同内容的 amp_prev_*、
#      init / amp_rejected(各自在某个 ckpt_e* 或侧车根目录有同 md5 的副本才删)。教师锚(teacher_*.bin, FP 参考前向)一个不碰。
#   超时(分钟, --tmo 段=分钟 覆盖): 每段一个硬上限, 撞了 rc 124 停车报出来, 不重试(超时说明哪段比预想的长, 要查)。训练/段 C 的超时套在停模型包装里, 杀掉后模型照样装回。
# 出错会怎样: 服务没起 → ② 连接失败直接停(不回退到别的采样); API 失败的题 ok:false 续跑只补它们; 料里任何一行训练器解析不了整趟停(core_ptrain_data.c)。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
FROM=""; TO=""; SN=6; GN=12; SMAX=24576; AUDIT=(); MIXN=0; THINKN=0; BMAX=8192; BEVERY=4096; REUSE=""; OCFG=""; AGN=0; SETALL=()
HQ=""; HD=gguf/go-onebit/code/svgheld; BASE=0; PLANH=0; BCUTS=""
WORK=""; EVOQ=0; HJ=""; BASELOG=""
# 训练 55 = 导序列 1.5 + 装模型 1 + 第 0 步评估 ~8.5 + 训练 ~19.5(256 题 64 步) + 轮末评估 ~8 + 余量(10-28 第一轮实测; ckpt 在轮末评估之后才落盘, 超时 = 整轮白训)
declare -A TMO=([q]=5 [collect]=20 [student]=12 [branch]=10 [train]=55 [judge]=20 [beh]=35 [review]=20)
while [ $# -gt 0 ]; do case "$1" in
    --from) FROM="${2:?}"; shift 2;; --to) TO="${2:?}"; shift 2;; --student-n) SN="${2:?}"; shift 2;; --graphics-n) GN="${2:?}"; shift 2;;
    --student-max-tokens) SMAX="${2:?}"; shift 2;; --mix-n) MIXN="${2:?}"; shift 2;; --think-n) THINKN="${2:?}"; shift 2;; --branch-max-tokens) BMAX="${2:?}"; shift 2;;
    --reuse-teacher) REUSE="${2:?}"; shift 2;; --cfg) OCFG="${2:?}"; shift 2;; --agent-n) AGN="${2:?}"; shift 2;; --branch-every) BEVERY="${2:?}"; shift 2;;
    --set-all) SETALL+=("${2:?}"); shift 2;;   # 族名=题集(10-27): 再加一族官方轨迹, 整族都采, 不受 --mix-n 抽题(SVG 角色插画族)
    --heldout-q) HQ="${2:?}"; shift 2;; --heldout-dir) HD="${2:?}"; shift 2;; --baseline) BASE=1; shift;;
    --plan-head) PLANH="${2:?}"; shift 2;; --branch-cuts) BCUTS="${2:?}"; shift 2;;
    --work) WORK="${2:?}"; shift 2;; --evo-q) EVOQ="${2:?}"; shift 2;; --heldout-judge) HJ="${2:?}"; shift 2;;
    --base-log) BASELOG="${2:?}"; shift 2;; --tmo) TMO[${2%%=*}]="${2#*=}"; shift 2;;
    --audit-ids) shift; while [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; do AUDIT+=("$1"); shift; done;;
    *) echo "★不认识的参数 $1★"; exit 2;; esac; done
[ -n "$WORK" ] && { FROM=${FROM:-0}; TO=${TO:-7}; }; FROM=${FROM:-1}; TO=${TO:-5}   # 一轮模式缺省 ⓪~⑦
stage(){ [ "$FROM" -le "$1" ] && [ "$TO" -ge "$1" ]; }
PY="$HOME/v41env/bin/python"; HF=hf/DeepSeek-V4.1-Flash-1017; PD=gguf-tools/data/posttrain; RC=$PD/restore_clean; RO=gguf/go-onebit/restore
KEY=$PD/deepseek_api.key; CFG=${OCFG:-$PD/restore_B2ctx_v4.cfg}
# 文件布局: 缺省 = 历来的固定名(②③ 按 id 续跑); --work = 一轮一个目录
QJ=$PD/mix_q.jsonl; S1LOG=$RO/mix_stage1.log; MIXRAW=$RO/mix_lp_raw.jsonl; STQ=$PD/student_q.jsonl; STU=$RO/student_v4.jsonl
BRRAW=$RO/branch_v4_raw.jsonl; BRROW=$RC/branch_v4.jsonl; HEADRAW=$RO/head_raw.jsonl; HEADROW=$RC/head_v4.jsonl; DATA=$RC/restore_v4.jsonl; LPRAW=$RO/v4_lp_raw.jsonl
EVO=gguf/go-onebit/evo; mapfile -t PKS < gguf/serve_pick.txt; ARM="${PKS[1]#"$ROOT"/}"
if [ -n "$WORK" ]; then
    [ "$PLANH" -gt 0 ] || { echo "★--work 只走计划段料, 要给 --plan-head★"; exit 2; }
    [ -n "$OCFG" ] || CFG=$PD/restore_B2plan_v6.cfg
    mkdir -p "$WORK" "$EVO/hold" "$EVO/beh" "$EVO/rows"; RN=$((10#$(basename "$WORK" | tr -dc 0-9)))
    QJ=$WORK/q.jsonl; S1LOG=$WORK/collect.log; MIXRAW=$WORK/api_raw.jsonl; STQ=$WORK/student_q.jsonl; STU=$WORK/student.jsonl
    BRRAW=$WORK/branch_raw.jsonl; BRROW=$WORK/branch_rows.jsonl; HEADRAW=$WORK/head_raw.jsonl; HEADROW=$WORK/head_rows.jsonl; DATA=$WORK/restore.jsonl; LPRAW=$WORK/lp_raw.jsonl
fi
T(){ local k=$1; shift; timeout --kill-after=30 "${TMO[$k]}m" "$@"; local rc=$?; [ "$rc" = 124 ] && M "★$k 段超时(${TMO[$k]} 分钟)★"; return "$rc"; }
ampmd5(){ md5sum "$ARM/amp_L00.bin" 2>/dev/null | cut -c1-12; }
PELQ=$RO/peldist_api.jsonl   # 单轮鹈鹕: 官方 8 份(10-27 分布对照)的题面与正文 —— 行为尺里图形那一道, 正文当评委参照
HOLD=$EVO/hold
CODEROWS(){ echo "gguf/v41judge/stu_code_judge_n15360_eng_$(basename "${PKS[0]}" .gguf)$1.rows"; }   # 编程判决份逐位读数: 参数空 = 裸臂, _amp_<侧车目录名> = 侧车臂
M(){ echo "[b2mix $(date '+%m-%d %H:%M:%S')] $*"; }
die(){ M "★$*★"; echo "B2MIX_EXIT 1"; exit 1; }
for f in "$KEY" "$HF/tokenizer.json" "$CFG" "$RC/api27_v3.jsonl" "$RC/think198_v3.jsonl" "$RO/api27_lp_raw.jsonl" "$RO/think198_lp_raw.jsonl"; do [ -s "$f" ] || die "缺 $f"; done
for s in code math data writing; do [ -s "$PD/mix_q_$s.txt" ] || die "缺题集 $PD/mix_q_$s.txt"; done

stage1q(){   # 题集(同步: ② 的学生题单要读它, 不能和 ① 的采集一起丢后台)
    M "① 题集 → $QJ$([ "$MIXN" -gt 0 ] && echo "(每族抽 $MIXN 题)")"
    "$PY" gguf-tools/scripts/think_trace_collect.py questions --set "code=$PD/mix_q_code.txt" "math=$PD/mix_q_math.txt" "data=$PD/mix_q_data.txt" "writing=$PD/mix_q_writing.txt" "${SETALL[@]}" --out "$QJ" || return 1
    if [ "$MIXN" -gt 0 ]; then   # 每族等距抽 N 题(不挑内容), 1 小时版用; --set-all 的族整族留着
        "$PY" - "$MIXN" "$QJ" "${SETALL[@]%%=*}" <<'EOF2' || return 1
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
    M "① 官方轨迹(带 top-20, 4 路, 上限 65536) → $MIXRAW"
    "$PY" gguf-tools/scripts/api_teacher.py collect --key "$KEY" --questions "$QJ" --out "$MIXRAW" --hf "$HF" --max-tokens 65536 --jobs 4 || return 1
    nok=$("$PY" -c "import json,sys; print(sum(1 for l in open('$MIXRAW') if json.loads(l).get('ok')))"); M "① 完成: ok $nok / $(wc -l < "$QJ") 题"
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
waitidle(){ for _ in $(seq 120); do curl -s --noproxy '*' --max-time 5 http://127.0.0.1:8001/metrics | grep -q '"state":"idle"' && return 0; sleep 5; done; return 1; }
# 就绪 = idle 且 工作台对这个子进程的冒烟已过(最新一份 serve_*.log 有"冒烟 过")。★只看 idle 不够★(10-28 实撞): 子进程一监听就是 idle,
#   ⑥ 抢在冒烟前发了 3 道行为尺题, 服务端串行, 冒烟排在后面 10 分钟超时 → 主进程判冒烟失败杀了子进程, 行为尺连接被拒。
waitready(){ local f; waitidle || return 1
    for _ in $(seq 120); do f=$(ls -t gguf/v41/posttrain/ui_logs/serve_*.log 2>/dev/null | head -1)
        grep -q "冒烟 过" "$f" 2>/dev/null && return 0; grep -q "冒烟 ★失败★\|★没起来\|★余量" "$f" 2>/dev/null && { M "★模型子进程没起来: $(grep -h "★" "$f" | tail -1)★"; return 1; }; sleep 5; done; return 1; }
beh_gen(){   # 现在装着的侧车做行为尺 6 道 → $EVO/beh/<amp 指纹>.jsonl, 官方评委打分 → .review.jsonl(按指纹缓存, 同一套侧车只生成一次)
    local fp; fp=$(ampmd5); [ -n "$fp" ] || { M "★侧车里没有 amp★"; return 1; }; local o="$EVO/beh/$fp"
    M "行为尺: 侧车 $fp 做 $(wc -l < "$HOLD/beh_q.jsonl") 道(已有的跳过) → $o.jsonl"
    # 每题上限 65536 位 = 官方最长(鹈鹕 3.6 万)的 1.8 倍: 到上限没收口记 finish=length、评 0 分(用户什么也没拿到)。只能说"到上限没停", 不说"不停"。
    #   10-27 实撞: 不设上限时现役 v6 的鹈鹕写到 8.26 万位没收口, 段超时一杀 freegen 什么都没落盘, 整轮停车。服务端同时只跑一个请求, 3 路 = 排队。
    T beh "$PY" gguf-tools/scripts/think_freegen_probe.py --questions "$HOLD/beh_q.jsonl" --n 0 --jobs 3 --max-tokens 65536 --timeout 3000 --out "$o.jsonl" || return 1
    T review "$PY" gguf-tools/scripts/api_round.py review --key "$KEY" --raw "$o.jsonl" --out "$o.review.jsonl" --jobs 16 || return 1
}
hold_evalrows(){   # 留出集 → 训练器 eval= 行: 每条只取 思考前 1024 位(计划) + 思考末 512 位 / </think> / 正文前 1536 位(决策位与终止位), 共 ~6 万位
    # 10-28 实撞: 整条轨迹(18 篇 27 万位)当 eval, 第 0 步评估 ~18 分钟, 轮末评估撞 40 分钟训练超时, ckpt 没落盘
    "$PY" gguf-tools/scripts/api_branch.py evalcut --hf "$HF" --raw "$HOLD/raw.jsonl" --out "$HOLD/eval_raw.jsonl" --head 1024 --tail 512 --content 1536 || return 1
    "$PY" gguf-tools/scripts/think_trace_collect.py chunks --hf "$HF" --raw "$HOLD/eval_raw.jsonl" --out "$HOLD/eval_rows.jsonl" --max-tok 1000000 --max-chunks 1 || return 1
}
hold_prep(){   # 留出集的官方轨迹 → eval 行 + 行为尺官方参照评分(后台跑, 只打 API)
    T collect "$PY" gguf-tools/scripts/api_teacher.py collect --key "$KEY" --questions "$HOLD/q.jsonl" --out "$HOLD/raw.jsonl" --hf "$HF" --max-tokens 65536 --jobs 8 --timeout 1200 || return 1
    hold_evalrows || return 1
    "$PY" - "$HOLD/raw.jsonl" "$PELQ" "$HOLD/beh_q.jsonl" "$HOLD/beh_official.jsonl" <<'EOFB' || return 1
import json, sys
raw = {r["id"]: r for r in (json.loads(l) for l in open(sys.argv[1], encoding="utf-8")) if r.get("ok")}
with open(sys.argv[4], "w", encoding="utf-8") as f:
    for ln in open(sys.argv[3], encoding="utf-8"):
        q = json.loads(ln)
        if q["src"] == "pelican":   # 官方 8 份都当参照(同题面 ⇒ 同 qh, summary 取均值)
            for r in (json.loads(l) for l in open(sys.argv[2], encoding="utf-8")): f.write(json.dumps(dict(r, src="pelican"), ensure_ascii=False) + "\n")
        elif q["id"] in raw: f.write(json.dumps(raw[q["id"]], ensure_ascii=False) + "\n")
        else: print(f"★行为尺 {q['id']} 没有官方轨迹★")
EOFB
    T review "$PY" gguf-tools/scripts/api_round.py review --key "$KEY" --raw "$HOLD/beh_official.jsonl" --out "$HOLD/official.review.jsonl" --jobs 16 || return 1
    M "留出集: 官方轨迹 ok $("$PY" -c "import json; print(sum(1 for l in open('$HOLD/raw.jsonl') if json.loads(l).get('ok')))") / $(wc -l < "$HOLD/q.jsonl"), eval 行 $(wc -l < "$HOLD/eval_rows.jsonl")"
}
mapfile -t BAN < <(grep -v '^#' "$PD/evo_ban.txt" 2>/dev/null | grep -v '^$')
HPID=""
if [ -n "$WORK" ] && stage 0; then   # ⓪ 固定通用留出集(题只出一次; 轨迹/评分续跑) + 记现役侧车指纹
    if [ ! -s "$HOLD/q.jsonl" ]; then
        SEEN=("$PD"/mix_q_*.txt "$PD"/hold_q_*.txt); for f in "$EVO"/r*/q.jsonl; do [ -s "$f" ] && SEEN+=("$f"); done
        M "⓪ 通用留出集: 官方出 6 族各 3 道 → $HOLD/q.jsonl(永不进料)"
        T q "$PY" gguf-tools/scripts/api_round.py questions --key "$KEY" --round 0 --prefix evohold --n 3 --out "$HOLD/q.jsonl" --seen "${SEEN[@]}" --ban "${BAN[@]}" || die "留出集出题失败"
        # 行为尺 = 编程/数学/数据/写作/通用各取第 1 道 + 鹈鹕(图形那一道用鹈鹕: 有官方 8 份参照, 也是对外的展示题)
        "$PY" - "$HOLD/q.jsonl" "$PELQ" "$HOLD/beh_q.jsonl" <<'EOFQ' || die "行为尺题单失败"
import json, sys
seen, out = set(), []
for r in (json.loads(l) for l in open(sys.argv[1], encoding="utf-8")):
    if r["src"] in ("code", "math", "data", "writing", "general") and r["src"] not in seen: seen.add(r["src"]); out.append(r)
out.append({"id": "pelican", "src": "pelican", "q": json.loads(open(sys.argv[2], encoding="utf-8").readline())["q"]})
with open(sys.argv[3], "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
EOFQ
    fi
    # 跳过要条数对齐(10-27 实撞: 参照评到 12/13 撞了超时, 重发时只看"文件非空"就跳过, 鹈鹕参照少一份)
    nref=$(grep -c '"ok": true' "$HOLD/official.review.jsonl" 2>/dev/null); nref=${nref:-0}; nwant=$(wc -l < "$HOLD/beh_official.jsonl" 2>/dev/null || echo 1)
    if [ "$nref" -lt "$nwant" ] || [ ! -s "$HOLD/raw.jsonl" ]; then hold_prep > "$HOLD/prep.log" 2>&1 & HPID=$!; M "⓪ 留出集官方轨迹 + 参照评分后台跑(日志 $HOLD/prep.log)"; fi
    [ -z "$HPID" ] && [ ! -s "$HOLD/eval_rows.jsonl" ] && { hold_evalrows || die "留出集 eval 行失败"; }   # 轨迹已在、eval 行还没出(换切法后)就当场出
    ampmd5 > "$WORK/base.fp"
fi
S1PID=""
if stage 1 && [ "$EVOQ" -gt 0 ]; then
    M "① 官方现出新题: 每族 $EVOQ 道(第 $RN 轮) → $QJ"
    SEEN=("$PD"/mix_q_*.txt "$PD"/hold_q_*.txt "$HOLD/q.jsonl"); for f in "$EVO"/r*/q.jsonl; do [ -s "$f" ] && [ "$f" != "$QJ" ] && SEEN+=("$f"); done
    # 禁词 = 判决题材(鹈鹕/鸟/骑车) + 泛化留出集的动物(evo_ban.txt, 一行一个); 题集已在(中断后重发)就沿用, 已采的轨迹按 id 续
    if [ -s "$QJ" ]; then M "① 题集已在($(wc -l < "$QJ") 道), 沿用"
    else T q "$PY" gguf-tools/scripts/api_round.py questions --key "$KEY" --round "$RN" --n "$EVOQ" --out "$QJ" --seen "${SEEN[@]}" --ban "${BAN[@]}" || die "出题失败"; fi
    stage1e(){ M "① 官方轨迹(带 top-20, 8 路) → $MIXRAW"; T collect "$PY" gguf-tools/scripts/api_teacher.py collect --key "$KEY" --questions "$QJ" --out "$MIXRAW" --hf "$HF" --max-tokens 65536 --jobs 8 --timeout 1200 || return 1
               M "① 完成: ok $("$PY" -c "import json; print(sum(1 for l in open('$MIXRAW') if json.loads(l).get('ok')))") / $(wc -l < "$QJ") 题"; }
    stage1e > "$S1LOG" 2>&1 & S1PID=$!; M "① 采集后台跑(日志 $S1LOG), 与 ② 并行"
elif stage 1; then stage1q || die "题集失败"; stage1c > "$S1LOG" 2>&1 & S1PID=$!; M "① 采集后台跑(日志 $S1LOG), 与 ② 并行"; fi
if stage 2; then
    M "② 学生生成: 每族 $SN 题, 上限 $SMAX, 服务 127.0.0.1:8001"
    # ⓪ 的基线判决停过模型再装回, 装回到 idle 要 ~80 s(/api/models/load 先回 ok 再加载): 等它进 idle 再发, 10 分钟不进就是没装回来
    waitready || die "模型服务没就绪(127.0.0.1:8001: idle + 工作台冒烟过)"
    if [ -n "$WORK" ]; then SG=/dev/null; ST=/dev/null; else SG=$RC/api27_v3.jsonl; ST=$RO/think198_lp_raw.jsonl; fi   # 一轮模式: 学生只做这一轮的新题
    "$PY" - "$SN" "$QJ" "$SG" "$ST" "$STQ" <<'EOF' || die "学生题单失败"
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
    T student "$PY" gguf-tools/scripts/think_freegen_probe.py --questions "$STQ" --n 0 --jobs 2 --max-tokens "$SMAX" --timeout 1200 --out "$STU" || die "学生生成失败"
    M "② 完成: $(wc -l < "$STU") 条学生轨迹, 收口 $("$PY" -c "import json; print(sum(1 for l in open('$STU') if json.loads(l).get('closed')))")"
fi
BEHB=""
if [ -n "$WORK" ] && [ "$FROM" -le 5 ] && [ ! -s "$EVO/beh/$(cat "$WORK/base.fp").review.jsonl" ]; then   # 现役的行为分没缓存: 后台生成, 与 ③④(只打 API)并行, ⑤ 停模型前等它
    [ -n "$HPID" ] && { wait "$HPID" || die "⓪ 留出集准备失败(看 $HOLD/prep.log)"; HPID=""; }   # 行为尺题单与官方参照要先在
    beh_gen > "$WORK/beh_base.log" 2>&1 & BEHB=$!; M "⓪ 现役侧车 $(cat "$WORK/base.fp") 的行为尺后台生成(日志 $WORK/beh_base.log)"
fi
if [ -n "$S1PID" ]; then wait "$S1PID" || die "① 失败(看 $S1LOG)"; tail -3 "$S1LOG"; fi
if stage 3; then
    M "③ 分支续写: $([ -n "$BCUTS" ] && echo "切点 $BCUTS" || echo "每 $BEVERY 位一个分支点"), 官方续 ≤$BMAX 位, 4 路"
    T branch "$PY" gguf-tools/scripts/api_branch.py branch --key "$KEY" --student "$STU" --out "$BRRAW" --hf "$HF" --every "$BEVERY" --cuts "$BCUTS" --max-tokens "$BMAX" --jobs 6 --timeout 600 || die "branch 失败"
    M "③ 分支收口: $("$PY" -c "import json; R=[json.loads(l) for l in open('$BRRAW')]; R=[r for r in R if r.get('ok')]; print(f\"ok {len(R)} 条, 官方续写收口 {sum(1 for r in R if r.get('closed'))} 条\")")"
    "$PY" gguf-tools/scripts/api_branch.py rows --raw "$BRRAW" --out "$BRROW" || die "rows 失败"
fi
if stage 4 && [ "$PLANH" -gt 0 ]; then   # 计划段版: 官方各族轨迹的思考前 PLANH 位 + 分支行, 不要整篇
    M "④ 料(计划段): 新题 + 图形 $GN 篇 + think198 抽 $THINKN 条的思考前 $PLANH 位 + 分支行 → $DATA"
    HR=("$MIXRAW"); if [ -n "$WORK" ]; then [ "$GN" -gt 0 ] && HR+=("$RO/api27_lp_raw.jsonl:$GN"); [ "$THINKN" -gt 0 ] && HR+=("$RO/think198_lp_raw.jsonl:$THINKN")   # 一轮模式 0 = 不回放
    else HR+=("$RO/api27_lp_raw.jsonl:$GN" "$RO/think198_lp_raw.jsonl:$THINKN"); fi
    "$PY" gguf-tools/scripts/api_branch.py head --hf "$HF" --n "$PLANH" --out "$HEADRAW" --raw "${HR[@]}" || die "head 失败"
    "$PY" gguf-tools/scripts/think_trace_collect.py chunks --hf "$HF" --raw "$HEADRAW" --out "$HEADROW" --max-tok 1000000 --max-chunks 1 || die "chunks 失败"
    cat "$HEADROW" "$BRROW" > "$DATA"
    cat "$HEADRAW" "$BRRAW" > "$LPRAW"
    M "料 $(wc -l < "$DATA") 行 = 头 $(wc -l < "$HEADROW") + 分支 $(wc -l < "$BRROW"); 教师原料 $(wc -l < "$LPRAW") 条"
    if [ -n "$WORK" ]; then   # 泄题门: 出题时已按禁词丢过题, 这里拦官方轨迹/学生前缀里自己冒出来的(比如画别的动物时顺口提到鸟) —— 含禁词的行整行剔掉, 不进梯度
        BANRE=$(grep -v '^#' "$PD/evo_ban.txt" | grep -v '^$' | paste -sd'|')
        grep -n -o -i -E "$BANRE" "$DATA" | head -5 | sed 's/^/    剔: 行 /'
        grep -v -i -E "$BANRE" "$DATA" > "$DATA.clean"; M "④ 泄题门: 含禁词的行 $(( $(wc -l < "$DATA") - $(wc -l < "$DATA.clean") )) 行剔掉, 剩 $(wc -l < "$DATA.clean")"; mv "$DATA.clean" "$DATA"
    fi
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
if stage 5 && [ -n "$WORK" ]; then   # 一轮模式: 从现役侧车续训 —— init = 现在侧车根目录的 amp(⓪ 记的指纹那套), cfg 由模板只换 data/out/init
    [ -n "$HPID" ] && { wait "$HPID" || die "⓪ 留出集准备失败(看 $HOLD/prep.log)"; }
    [ -n "$BEHB" ] && { M "⑤ 等现役行为尺生成收工(训练要停模型)"; wait "$BEHB" || { tail -5 "$WORK/beh_base.log"; die "现役行为尺失败(看 $WORK/beh_base.log)"; }; grep -E "^\| |\[evohold|\[pelican" "$WORK/beh_base.log"; }
    # 现役的编程判决份 rows 按指纹留底(段 C 会盖同名文件); 盘上那份就是最后一次判的侧车 = 现役(第一轮: v6 10-27 19:04 判的)
    [ -s "$EVO/rows/$(cat "$WORK/base.fp").code.rows" ] || cp "$(CODEROWS "_amp_$(basename "$ARM")")" "$EVO/rows/$(cat "$WORK/base.fp").code.rows" || die "现役编程 rows 留底失败"
    [ "$(ampmd5)" = "$(cat "$WORK/base.fp")" ] || die "侧车 amp 指纹变了(⓪ 记的 $(cat "$WORK/base.fp"), 现在 $(ampmd5)), 有别人动过侧车"
    mkdir -p "$WORK/init"; cp "$ARM"/amp_L*.bin "$WORK/init/" || die "拷 init 失败"
    # base.fnv = ③ 解在哪个反修态(段 A)上的指纹; 段 A 不变, 从任一份同段 A 训出来的目录拷(没有它训练器只警告, 读数不核对)
    for b in "$REUSE/base.fnv" gguf/v41/posttrain/restore-evo-*/base.fnv; do [ -s "$b" ] && { cp "$b" "$WORK/init/"; break; }; done
    OUT=gguf/v41/posttrain/restore-evo-$(basename "$WORK")
    sed -e "s#^data=.*#data=$DATA#" -e "s#^out=.*#out=$OUT#" -e "s#^init=.*#init=$WORK/init#" "$CFG" > "$WORK/train.cfg"
    grep -q '^init=' "$WORK/train.cfg" || echo "init=$WORK/init" >> "$WORK/train.cfg"
    grep -v '^eval=' "$WORK/train.cfg" > "$WORK/train.cfg.t" && echo "eval=$HOLD/eval_rows.jsonl" >> "$WORK/train.cfg.t" && mv "$WORK/train.cfg.t" "$WORK/train.cfg"   # 分布尺: 通用留出集当泛化留出
    cat "$LPRAW" "$HOLD/eval_raw.jsonl" > "$WORK/lp_all.jsonl"; LPRAW=$WORK/lp_all.jsonl   # 留出集的官方 top-20 也要进教师表(eval 行按它算 KL)
    CFG=$WORK/train.cfg
fi
if stage 5; then
    OUT=$(grep '^out=' "$CFG" | cut -d= -f2-)
    if [ -n "$REUSE" ]; then   # 保持料的教师表复用(posttrain_fp_spark.sh 按"有 .ptf 的份保留 + 内容键配"接: 拷 000 以外的份, API 那份 000 本趟重打)
        mkdir -p "$OUT"; n=0
        for p in "$REUSE"/teacher.part.[0-9][0-9][0-9]; do [ -s "$p.ptf" ] && [ "${p##*.}" != 000 ] && { cp "$p" "$p.ptf" "$OUT"/; n=$((n + 1)); }; done
        M "复用 $REUSE 的 $n 份教师表(保持料), 本地 FP 教师只补没对上的"
    fi
    EX=(); [ -n "$WORK" ] && EX=(--tmo-train "${TMO[train]}" --tmo-judge "${TMO[judge]}" --heldout "${HJ:-none}" --pelseg 0)   # 一轮模式缺省不判 6 段留出集(19.8 万位, 一轮装不下), 要判就 --heldout-judge 目录
    M "⑤ 训 + 段 C: restore_B2_api_spark.sh --from 3 --to 4 --cfg $CFG --teacher-raw $LPRAW ${EX[*]}"
    if ! bash gguf-tools/scripts/restore_B2_api_spark.sh --from 3 --to 4 --cfg "$CFG" --teacher-raw "$LPRAW" "${EX[@]}" 2>&1 | tee "${WORK:-/tmp}/train_judge.log"; then
        # 训练前侧车的 amp 被挪走了(restore_B2_api_spark.sh ③), 半路失败模型装回的是没有段 B 的侧车 —— 一轮模式把 init 拷回去重装, 现役不受影响
        if [ -n "$WORK" ] && ! ls "$ARM"/amp_L*.bin >/dev/null 2>&1; then cp "$WORK/init"/amp_L*.bin "$ARM/"; bash gguf-tools/scripts/with_model_stopped_spark.sh true > "$WORK/reload.log" 2>&1; M "失败后侧车换回 init 并重装"; fi
        die "训练链失败"
    fi
fi
if stage 6 && [ -n "$WORK" ]; then   # ⑥ 行为尺 + 去留(全族通用能力; 鹈鹕只是行为尺里的一道)
    OUT=$(grep '^out=' "$CFG" | cut -d= -f2-); BFP=$(cat "$WORK/base.fp"); waitready || die "模型没装回就绪(idle + 冒烟过)"
    NFP=$(ampmd5); [ "$NFP" != "$BFP" ] || die "侧车 amp 指纹没变($NFP), 新一轮没并进去"
    cp "$(CODEROWS "_amp_$(basename "$ARM")")" "$EVO/rows/$NFP.code.rows" || die "本轮编程 rows 留底失败"
    beh_gen > "$WORK/beh_new.log" 2>&1 || { tail -5 "$WORK/beh_new.log"; die "⑥ 行为尺生成/评审失败(看 $WORK/beh_new.log)"; }
    M "⑥ 行为尺本轮各题:"; grep -E "^\| |\[evohold|\[pelican" "$WORK/beh_new.log"
    # 别 tee /dev/stderr: Linux 上按路径重开日志并截断, 前面的日志全丢(10-28 在 restore_B_api_round_spark.sh 实撞)
    SO=$("$PY" gguf-tools/scripts/api_round.py summary "官方=$HOLD/official.review.jsonl" "现役=$EVO/beh/$BFP.review.jsonl" "本轮=$EVO/beh/$NFP.review.jsonl" --json); printf '%s\n' "$SO" >&2
    SJ=$(grep '^SUMMARY_JSON' <<<"$SO" | cut -d' ' -f2-)
    read -r RB RN2 < <("$PY" -c "import json,sys; d=json.loads(sys.argv[1]); print(round(d['现役']['ratio'], 2), round(d['本轮']['ratio'], 2))" "$SJ")
    # 分布尺(官方 top-20, 通用留出集 = 训练器 eval= 泛化留出): train.log 的 step 0 / epoch 1 行的 gen_kl; decision 行 = 决策位 KL
    #   ★不用 gain/se★: 那是普通留出的逐题配对, 给了 eval= 时普通留出是 0 题, gain 恒 0(10-28 第一轮实撞); 泛化留出训练器只出均值, 没有逐题配对
    read -r E0 EK < <(awk '/^step 0 /{for(i=1;i<NF;i++) if($i=="gen_kl") a=$(i+1)} /^epoch 1 /{for(i=1;i<NF;i++) if($i=="gen_kl") b=$(i+1)} END{print a, b}' "$OUT/train.log")
    G=$(awk -v a="$E0" -v b="$EK" 'BEGIN{printf "%.5f", a-b}'); GS="-"
    read -r D0 D1 DN < <(awk '/^decision e00/{a=$4; n=$6} /^decision e01/{b=$4} END{print a, b, n}' "$OUT/train.log")
    # 编程 / wt2 读数: 段 C 依次打 编程裸 / 编程侧车 / wt2 裸 / wt2 侧车, 取第 2 / 4 个 Same top
    st(){ grep -o 'Same top token  = [0-9.]*' "$1" | awk -v k="$2" 'NR==k{print $NF}'; }
    CN=$(st "$WORK/train_judge.log" 2); WN=$(st "$WORK/train_judge.log" 4)
    read -r CB WB < <(awk -F'\t' -v f="$BFP" '$3==f{c=$4; w=$6} $2==f{c=$5; w=$7} END{print c, w}' "$EVO/ledger.tsv" 2>/dev/null)   # 现役 = 某轮留下的新侧车, 或某轮回退后的旧侧车
    [ -z "$CB" ] && [ -n "$BASELOG" ] && { CB=$(st "$BASELOG" 2); WB=$(st "$BASELOG" 4); }
    [ -n "$CN" ] && [ -n "$CB" ] && [ -n "$E0" ] && [ -n "$EK" ] && [ -n "$D1" ] && [ -n "$RN2" ] || die "读数缺(编程 '$CN'/'$CB' 分布 gain '$G' 决策位 '$D1' 行为 '$RN2'; 现役没进 ledger 就给 --base-log)"
    M "⑥ 分布尺(通用留出 $(wc -l < "$HOLD/eval_rows.jsonl") 段对官方 top-20): 泛化留出 KL 第 0 步(现役) $E0 → 轮末 $EK(降 $G, $(awk -v a="$E0" -v b="$EK" 'BEGIN{printf "%+.1f%%", (b/a-1)*100}')); 决策位($DN 位)KL $D0 → $D1"
    M "⑥ 行为尺(官方评委, 逐题对官方的还原率): 现役 $RB% → 本轮 $RN2%; 编程判决份 Same $CB → $CN; wt2 $WB → $WN(±1.8 pp)"
    M "⑥ 细节(编程判决份按 FP 置信度分桶, 看 p>0.9 / 0.5~0.9 两桶的错 top-1):"
    "$PY" gguf-tools/scripts/judge_rows_tokclass.py --hf "$HF" --ids gguf/go-onebit/code/judge.ids --conf "裸=$(CODEROWS "")" "现役=$EVO/rows/$BFP.code.rows" "本轮=$EVO/rows/$NFP.code.rows" 2>&1 | grep -E "^== |^  conf " | sed 's/^/    /'
    grep -E "^ds4: \[ptrain\] (第 0 步|★?轮 1|料:)" "$OUT/train.out" 2>/dev/null | sed 's/^ds4: \[ptrain\] /    /'
    keep=$(awk -v g="$G" -v d0="$D0" -v d1="$D1" -v cb="$CB" -v cn="$CN" -v rb="$RB" -v rn="$RN2" 'BEGIN{print (g > 0 && d1 <= d0 && cn >= cb - 0.3 && rn >= rb - 5) ? 1 : 0}')
    if [ "$keep" = 1 ]; then V=keep; M "⑥ 留: 本轮侧车 $NFP 成为现役"
    else
        V=rollback; mkdir -p "$WORK/amp_rejected"; mv "$ARM"/amp_L*.bin "$WORK/amp_rejected/" && cp "$WORK/init"/amp_L*.bin "$ARM/" || die "回退拷 amp 失败"
        printf '# 回退(自进化 %s 没过去留门): 段 B 换回 %s/init(指纹 %s), 本轮 amp 在 %s/amp_rejected, %s\n' "$(basename "$WORK")" "$WORK" "$BFP" "$WORK" "$(date '+%Y-%m-%d %H:%M')" >> "$ARM/manifest.txt"
        bash gguf-tools/scripts/with_model_stopped_spark.sh true > "$WORK/reload.log" 2>&1; waitready || die "回退后模型没装回"
        [ "$(ampmd5)" = "$BFP" ] || die "回退后指纹不对"; M "⑥ 回退: 侧车换回 $BFP 并已重装"
    fi
    # ledger 列: 轮 现役指纹 本轮指纹 编程本轮 编程现役 wt2本轮 wt2现役 分布降幅 - 决策位e00 决策位e01 行为本轮% 行为现役% 去留 (分布两列 = 泛化留出 gen_kl 第 0 步减轮末的降幅, "-")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(basename "$WORK")" "$BFP" "$NFP" "$CN" "$CB" "$WN" "$WB" "$G" "$GS" "$D0" "$D1" "$RN2" "$RB" "$V" >> "$EVO/ledger.tsv"
    M "ledger 追加: $(tail -1 "$EVO/ledger.tsv")"
fi
if stage 7 && [ -n "$WORK" ]; then   # ⑦ 清理
    OUT=gguf/v41/posttrain/restore-evo-$(basename "$WORK"); GB=$(basename "${PKS[0]}" .gguf); AB=$(basename "$ARM"); before=$(df -B1 --output=avail . | tail -1)
    for t in code_judge_n15360 g7_wt2_n512; do for sfx in "" "_amp_$AB"; do rm -f "gguf/v41judge/stu_${t}_eng_${GB}${sfx}.bin" "gguf/v41judge/stu_${t}_eng_${GB}${sfx}.bin."*; done; done
    rm -f "$OUT"/teacher.ids "$OUT"/teacher.ptf "$OUT"/teacher.part.* "$OUT"/teacher.rest "$WORK/lp_all.jsonl" "$WORK/lp_raw.jsonl"
    m00(){ md5sum "$1/amp_L00.bin" 2>/dev/null | cut -c1-32; }
    has_copy(){   # $1 的 amp 在侧车根目录或某个 ckpt_e* 里有同内容的另一份(逐层比 40 个文件)
        local d c; for d in "$ARM" gguf/v41/posttrain/*/ckpt_e*; do [ "$d" -ef "$1" ] && continue; [ "$(m00 "$d")" = "$(m00 "$1")" ] || continue
            c=0; for f in "$1"/amp_L*.bin; do cmp -s "$f" "$d/$(basename "$f")" || { c=1; break; }; done; [ "$c" = 0 ] && return 0; done; return 1; }
    for d in "$ARM"/amp_prev_*; do [ -d "$d" ] && [ -d "$WORK/init" ] && [ "$(m00 "$d")" = "$(m00 "$WORK/init")" ] && has_copy "$d" && rm -r "$d" && M "⑦ 删 $d(与 init 同, 别处有副本)"; done
    for d in "$WORK/init" "$WORK/amp_rejected"; do [ -d "$d" ] && has_copy "$d" && rm -r "$d" && M "⑦ 删 $d(别处有同内容副本)"; done
    M "⑦ 清理完: 腾出 $(( ($(df -B1 --output=avail . | tail -1) - before) / 1000000000 )) GB, 盘余 $(df -h --output=avail . | tail -1 | tr -d ' ')"
fi
M "完成"; echo "B2MIX_EXIT 0"
