#!/bin/bash
# bench_suite_spark.sh — 在 spark 上按 unsloth 口径把六项任务基准顺序跑一遍, 每项一个时间预算, 产物进 gguf/v41/bench/<日期>/。
# 对着 ds4-server 直连口 127.0.0.1:8001(工作台子进程, 现役侧车已挂)。跑的时候模型服务被占满, 穿透那边的聊天会排队。
# 用法: bench_suite_spark.sh <日期标签> [预算分钟=50] [并发=3] [任务列表...]
#   默认任务顺序 aime25 gpqa_diamond lcb_v6 ifeval mmlu_pro hmmt_feb26(每项预算到点不再发新请求, 在跑的请求跑完为止,
#   所以每项实际 ≈ 预算 + 最后一批的尾巴). LCB 跑完顺手在本机判分(lcb_grade.py); IFEval 判分要在 Mac 上跑(ifeval_grade.py)。
#   每题最多样本数照 unsloth: AIME/HMMT 16, GPQA 8, MMLU-Pro 2, IFEval 2, LCB 3 —— 预算内先把每题各跑 1 个再跑第 2 个。
set -uo pipefail
cd "$(dirname "$0")/../.."
TAG="${1:?日期标签}"; BUDGET="${2:-50}"; JOBS="${3:-3}"; shift 3 2>/dev/null || shift $#
TASKS=("$@"); [ ${#TASKS[@]} -gt 0 ] || TASKS=(aime25 gpqa_diamond lcb_v6 ifeval mmlu_pro hmmt_feb26)
OUT="gguf/v41/bench/$TAG"; mkdir -p "$OUT"
declare -A SAMPLES=([aime25]=16 [hmmt_feb26]=16 [gpqa_diamond]=8 [mmlu_pro]=2 [ifeval]=2 [lcb_v6]=3)
curl -s --noproxy '*' -m 10 http://127.0.0.1:8001/v1/models > /dev/null || { echo "8001 没在服务" >&2; exit 2; }
echo "# $(date '+%F %T') 起跑 tag=$TAG 预算=${BUDGET}min 并发=$JOBS 任务=${TASKS[*]}" | tee -a "$OUT/chain.log"
for spec in "${TASKS[@]}"; do
    # 任务既可以是名字(gguf-tools/bench/data/<名>.jsonl), 也可以是 路径[:样本数](10-31 复查进了训练料的题: 单独的小题集, 不走排除清单, 样本数自定)
    t=$spec; DATA="gguf-tools/bench/data/$t.jsonl"; NS="${SAMPLES[$t]:-1}"; EXCL=()
    if [[ "$spec" == */* ]]; then DATA="${spec%%:*}"; t=$(basename "$DATA" .jsonl); [[ "$spec" == *:* ]] && NS="${spec##*:}"; EXCL=(--exclude ""); fi
    echo "# $(date '+%F %T') ▶ $t" | tee -a "$OUT/chain.log"
    python3 gguf-tools/scripts/bench_suite.py --data "$DATA" --out "$OUT/$t" \
        --budget-min "$BUDGET" --jobs "$JOBS" --samples "$NS" "${EXCL[@]}" 2>&1 | tee -a "$OUT/$t.log"
    if [ "$t" = lcb_v6 ]; then
        python3 gguf-tools/scripts/lcb_grade.py --out "$OUT/$t" --tests gguf-tools/bench/data/lcb_tests.jsonl 2>&1 | tee -a "$OUT/$t.grade.log"
    fi
    echo "# $(date '+%F %T') ■ $t $(tail -1 "$OUT/$t.log" | cut -c1-200)" | tee -a "$OUT/chain.log"
done
echo "# $(date '+%F %T') 全部跑完" | tee -a "$OUT/chain.log"
