#!/bin/bash
# think_close_probe_spark.sh — 收口位尺(2026-10-20): 对真实"思考不收口"序列整段重打分 top-32, 比几份反修目录下模型给 </think>(128822) 的概率。
#
# 怎么用(spark 仓库根, nohup): speed-bench/think_close_probe_spark.sh <底座 gguf> <序列.ids:提示 token 数:入环位>... -- <反修目录|none>...
#   例: speed-bench/think_close_probe_spark.sh gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf \
#         gguf/v41/night/faith_1016/pelican_1018.ids:8215:38449 gguf/v41/night/faith_1016/pelican_1020.ids:11358:32341 \
#         -- gguf/v41/<基座>-grrb-code_fit_think_n15360-engine gguf/v41/<基座>-grrb-code_fit_n15360-engine
#   FP 档(2026-10-20 下午, 用户批"跑"): 反修目录位置写 fp:<出厂权重目录>:<反修目录> —— 官方参考前向(v41_teacher.py 读 HF 出厂权重)对整条序列
#     逐位出全精度 logits(v41_judge.sh 教师锚, 43k 位 ≈ 23 GB 落 gguf/v41judge), 同趟引擎+反修目录打学生, 两边都过 bugmd_ids_tools.py anchor-faith
#     —— 与 10-17 r316 那趟同口径: 每 4096 生成位 FP 对量化的 Σp(</think>), 入环前后 p(top-1) 分位。这是定"量化压的 vs 模型本来就不给"的唯一直接读数。
# 做什么: 工作台模型子进程空闲就停它 → 每条序列 × 每份反修目录跑一趟 bugmd_gates_spark.sh scoreids(整段 top-32, 不生成) + minp(按产品采样门
#   再算一遍) → 汇总每趟的 "出口 Σp"(入环前/入环后分开)与最高 8 个收口位的 p(</think>) → 退出时按 serve_pick 装回模型(model_restart_spark.sh)。
# 为什么要它: 09-30 的口径(全精度在收口位给 </think> 22~49%, 量化 2~6%)是"反修有没有把这个决策位修回来"的唯一直接读数; 五指标在判决份
#   (没有思考段)上量不到它。同一二进制、同一条序列、只换反修目录, 差异才纯粹来自侧车。
# 判读: 看 minp 的 "出口 Σp 原始" 与 "min_p 后" —— 新侧车 Σp 对旧侧车翻倍以上才算推动了收口位; 两份一样 ⇒ 料没碰到这个决策。
# 出错会怎样: 子进程不是 idle 直接退出不停它; scoreids 自带 2500 MB 看门狗; 任一趟失败照样跑下一趟, 汇总里那一格是"失败"。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
M(){ echo "[close $(date '+%m-%d %H:%M:%S')] $*"; }
MODEL="${1:?底座 gguf}"; shift
SEQS=(); AMPS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do SEQS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
while [ $# -gt 0 ]; do AMPS+=("$1"); shift; done
[ -s "$MODEL" ] && [ ${#SEQS[@]} -gt 0 ] && [ ${#AMPS[@]} -gt 0 ] || { M "★用法见文件头: 底座 + 序列 -- 反修目录★"; exit 2; }
for a in "${AMPS[@]}"; do
    case "$a" in
        none) ;;
        fp:*) IFS=: read -r _ hf amp <<<"$a"; [ -s "$hf/config.json" ] && [ -s "$amp/manifest.txt" ] || { M "★fp 档要 fp:<出厂权重目录>:<反修目录>, 给的是 $a★"; exit 2; };;
        *) [ -s "$a/manifest.txt" ] || { M "★反修目录 $a 没有 manifest.txt★"; exit 2; };;
    esac
done

st=$(curl -s --noproxy '*' --max-time 5 127.0.0.1:8001/metrics | tr ',' '\n' | grep -o '"state":"[a-z]*"' | head -1)
[ -z "$st" ] || [ "$st" = '"state":"idle"' ] || { M "★模型子进程 $st, 不是 idle, 不停, 退出★"; exit 1; }
if pgrep -x ds4-server >/dev/null; then
    M "停模型子进程 $(pgrep -x ds4-server | tr '\n' ' ')"; pkill -x -TERM ds4-server
    for _ in $(seq 1 60); do pgrep -x ds4-server >/dev/null || break; sleep 2; done
    pgrep -x ds4-server >/dev/null && { pkill -x -KILL ds4-server; sleep 5; }
    pgrep -x ds4-server >/dev/null && { M "★ds4-server 杀不掉★"; exit 1; }
    sleep 6
    trap 'rc=$?; M "装回模型子进程"; bash gguf-tools/scripts/model_restart_spark.sh; echo "CLOSE_EXIT $rc"' EXIT
fi
OUT=/tmp/bugmd; mkdir -p "$OUT"; SUM="$OUT/think_close_$(date +%m%d_%H%M).txt"
for s in "${SEQS[@]}"; do
    IFS=: read -r IDS NP ENTRY <<<"$s"; [ -s "$IDS" ] || { M "★没有 $IDS★"; continue; }
    for a in "${AMPS[@]}"; do
        if [[ "$a" == fp:* ]]; then
            # 官方参考前向 + 引擎学生, 产物名照 v41_judge.sh: teacher_<目录>_<ids>_n<N>_<HF 目录名>.bin / stu_…_eng_<gguf>_amp_<反修目录>.bin
            IFS=: read -r _ hf amp <<<"$a"; NT=$(wc -w < "$IDS")
            M "== FP 档 $(basename "$IDS")(提示 $NP, 入环 $ENTRY, $NT 位): v41_judge.sh --hf $(basename "$hf") engine:$(basename "$MODEL"):$(basename "$amp")"
            bash gguf-tools/scripts/v41_judge.sh "$IDS" "$NT" --hf "$hf" "engine:$MODEL:$amp" 2>&1 | grep -E "^\[judge\]|★|PPL|Same top" | tail -12
            jtag="$(basename "$(dirname "$IDS")")_$(basename "$IDS" .ids)_n$NT"
            TB="gguf/v41judge/teacher_${jtag}_$(basename "$hf").bin"; SB="gguf/v41judge/stu_${jtag}_eng_$(basename "$MODEL" .gguf)_amp_$(basename "$amp").bin"
            for side in "FP:$TB" "量化:$SB"; do
                IFS=: read -r nm bin <<<"$side"
                if [ -s "$bin" ]; then
                    tagf="$(basename "$IDS" .ids)_faith_${nm}"; python3 speed-bench/bugmd_ids_tools.py anchor-faith "$bin" "$IDS" "$NP" "$ENTRY" > "$OUT/$tagf.txt" 2>&1
                    { echo "### $tagf ($bin)"; cat "$OUT/$tagf.txt"; echo; } >> "$SUM"; M "-- $nm"; cat "$OUT/$tagf.txt"
                else echo "### $(basename "$IDS" .ids) FP 档 $nm 失败(没有 $bin)" >> "$SUM"; M "★$nm 没出 $bin★"; fi
            done
            continue
        fi
        tag=$(basename "$IDS" .ids)_top32_$([ "$a" = none ] && echo bare || basename "$a" | sed 's/^DeepSeek-V4.1-Flash-//; s/-engine$//')
        M "== $tag(提示 $NP, 入环 $ENTRY)"
        bash speed-bench/bugmd_gates_spark.sh scoreids "$IDS" "$NP" 32 "$ENTRY" "$a" "$MODEL" 2>&1 | grep -E "^\[bugmd\]|★" | tail -3
        if [ -s "$OUT/$tag.bin" ]; then
            python3 speed-bench/bugmd_ids_tools.py minp "$OUT/$tag.bin" "$IDS" "$NP" "$ENTRY" > "$OUT/$tag.minp" 2>&1
            { echo "### $tag"; grep -E "^==|出口 Σp|生成第" "$OUT/$tag.minp"; echo; } >> "$SUM"
            grep -E "^==|出口 Σp" "$OUT/$tag.minp"
        else
            echo "### $tag 失败(没有 $OUT/$tag.bin)" >> "$SUM"; M "★$tag 打分失败★"
        fi
    done
done
M "汇总 → $SUM"; cat "$SUM"
