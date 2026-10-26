#!/bin/bash
# teacher_layer_locate_spark.sh — FP 教师长程失灵的逐层定位(2026-10-25): 引擎(裸量化底座, 与官方 API 逐字同能抄 1500 位) vs FP 教师
# (官方 model.py + v41_patches, 600 位后抄不了)在同一条序列上落每层 MoE 入口 x, 按层 × 位置段看相对误差从哪跳起来。
#
# 判读: 量化噪声在所有位置段同一量级(前 128 位 = 地板); 某层从 [640, …) 段起误差跳一个量级 ⇒ 病在那一层的注意力/索引选择
#   (L2 起跳 = 压缩 KV 注意力本身; L20 起跳 = ratio-1 层索引器选择; 只在第二份 X 跳 = 回看失灵)。
# 用法(spark 仓库根, 包在停模型包装器里; 引擎 ./ds4 须是 10-25 之后的构建: core_v41_score.c 的逐层 dump 上限从 64 位放到一块 2048 位):
#   nohup gguf-tools/scripts/with_model_stopped_spark.sh gguf-tools/scripts/teacher_layer_locate_spark.sh <ids(≤2048 位, 一行一个 id)> [--no-engram] > /tmp/bugmd/teacher_locate_<时间>.log 2>&1 &
#   --no-engram: 两边都摘 engram(引擎 --v41-no-engram / 教师 --no-engram), 把 engram 从对拍里拿掉。
# 产物: gguf/v41judge/locate_<名>[_noeng]/ 下 eng.bin(+ eng.bin.x_Lnn.bin / eng.bin.y_Lnn.bin) / teacher.bin / fp/x_Lnn.bin / rows(anchor_metrics --row-out) / parity.txt
# 出错会怎样: 序列 > 2048 位引擎分两块就不落 dump(parity 报"缺文件"); 引擎或教师 rc≠0 立刻停。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
M(){ echo "[locate $(date '+%m-%d %H:%M:%S')] $*"; }
IDS="${1:?ids 文件}"; shift
NOENG=0; [ "${1:-}" = "--no-engram" ] && NOENG=1
NM=$(basename "$IDS" .ids); N=$(grep -c . "$IDS")
[ "$N" -le 2048 ] || { M "★$N 位 > 2048, 引擎会分块, 不落逐层 dump★"; exit 2; }
GG=$(sed -n 1p gguf/serve_pick.txt); GG="${GG#"$ROOT"/}"; [ -s "$GG" ] || { M "★serve_pick 第 1 行 $GG 不是文件★"; exit 2; }
HF=hf/DeepSeek-V4.1-Flash-1017
D="gguf/v41judge/locate_${NM}$([ $NOENG = 1 ] && echo _noeng)"; mkdir -p "$D/fp"
M "序列 $IDS($N 位) 底座 $GG(裸, 不挂侧车) engram $([ $NOENG = 1 ] && echo 关 || echo 开) → $D"
# ① 引擎: 整条一块(默认块 2048 ≥ N) ⇒ 自动落 eng.x_Lnn.bin
M "① 引擎 --score-ids"
./ds4 --cuda -m "$GG" --mem-budget-mb 110000 --no-dspark --score-ids "$IDS" --score-out "$D/eng.bin" $([ $NOENG = 1 ] && echo --v41-no-engram) > "$D/eng.log" 2>&1; rc=$?
grep -E "PPL|分块|no-engram|错|失败|Error" "$D/eng.log" | tail -3
[ "$rc" = 0 ] && [ -s "$D/eng.bin" ] && [ -s "$D/eng.bin.x_L20.bin" ] || { M "★引擎 rc=$rc 或没落 dump(应有 $D/eng.bin.x_Lnn.bin), 看 $D/eng.log★"; exit 3; }
# ② FP 教师 + --dump-moe
M "② FP 教师 --dump-moe"
"$HOME/v41env/bin/python" gguf-tools/scripts/v41_teacher.py "$HF" --ids "$IDS" --ntok "$N" --out "$D/teacher.bin" --dump-moe "$D/fp" --dump-idx "$D/idx" $([ $NOENG = 1 ] && echo --no-engram) > "$D/teacher.log" 2>&1; rc=$?
grep -E "完成|PPL|★|Error|Traceback" "$D/teacher.log" | tail -3
[ "$rc" = 0 ] && [ -s "$D/teacher.bin" ] && [ -s "$D/fp/x_L20.bin" ] || { M "★教师 rc=$rc, 看 $D/teacher.log★"; exit 4; }
# ③ 出口五指标 + 逐位 rows, 再逐层 × 位置段
gguf-tools/bench/anchor_metrics --ref-raw "$D/teacher.bin" --ids "$IDS" --student "$D/eng.bin" --row-out "$D/rows" | grep -E "KLD|Same top|Σmin|PPL" | tr '\n' ' '; echo
H=$(( (N - 1) / 2 + 1 ))   # BOS + X + X 的第二份起点
"$HOME/v41env/bin/python" gguf-tools/scripts/v41_layer_parity.py "$D/fp" "$D/eng.bin" 40 --buckets "128,640,$H" | tee "$D/parity.txt"   # 引擎 dump 前缀 = --score-out 路径本身(<out>.x_Lnn.bin)
M "完成 → $D"
