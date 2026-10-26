#!/bin/bash
# teacher_bisect_spark.sh — FP 教师长程失灵的版本二分(2026-10-25): 用历史版本的 v41_teacher.py 跑同一条序列, 和今天的教师、引擎学生逐位比。
#
# 背景: 09-22 教师在 24400 位复读区 FP argmax 命中 99.7%(fable5), 今天的教师抄 1500 位的第二份错块、工具 schema 段 NLL 5 对引擎 1.3
#   —— 病是 09-22 之后改教师脚本带进来的。候选三步: 1b95a38c(09-23) → 57b0b316(09-30) → 152c72eb(10-23 --seqs + 整层预读)。
# 用法(spark 仓库根, 包在停模型包装器里):
#   nohup gguf-tools/scripts/with_model_stopped_spark.sh gguf-tools/scripts/teacher_bisect_spark.sh <ids> <ntok> <版本目录名...> > /tmp/bugmd/teacher_bisect_<时间>.log 2>&1 &
#   版本目录 = gguf-tools/scripts/bisect_<名>/(整套 v41_teacher.py + v41_patches.py + v41_cli.py + v41_hf_io.py 从 git 原样抽出, 自带依赖, 不碰现役脚本)。
# 产物: gguf/v41judge/teacher_<ids名>_n<ntok>.<版本>.bin + /tmp/bugmd/<ids名>.<版本>.rows(anchor_metrics --row-out, 与今天教师的 rows 同口径)。
# 判读: 哪个版本第二份 X 的教师 NLL 回到 ≈0(学生水平), 病就在它之后那一步的改动里。
# 出错会怎样: 某版本 rc≠0 打 FAIL 和日志尾 3 行, 继续跑下一个版本; 学生 logits 不在直接退出(先用 v41_judge.sh 打学生)。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
M(){ echo "[bisect $(date '+%m-%d %H:%M:%S')] $*"; }
IDS="${1:?ids 文件}"; NTOK="${2:?ntok}"; shift 2
[ $# -ge 1 ] || { M "★要给至少一个版本目录名★"; exit 2; }
NM=$(basename "$IDS" .ids)
STU=$(ls gguf/v41judge/stu_pelican_${NM}_n${NTOK}_eng_*.bin 2>/dev/null | head -1)
[ -s "$STU" ] || { M "★学生 logits 不在: gguf/v41judge/stu_pelican_${NM}_n${NTOK}_eng_*.bin★"; exit 3; }
for v in "$@"; do
    D="gguf-tools/scripts/bisect_$v"
    [ -s "$D/v41_teacher.py" ] || { M "★$D 没有 v41_teacher.py★"; continue; }
    OUT="gguf/v41judge/teacher_pelican_${NM}_n${NTOK}.$v.bin"; LOG="gguf/v41judge/log_teacher_${NM}.$v.txt"
    M "== 版本 $v: $D/v41_teacher.py → $OUT"
    "$HOME/v41env/bin/python" "$D/v41_teacher.py" hf/DeepSeek-V4.1-Flash-1017 --ids "$IDS" --ntok "$NTOK" --out "$OUT" > "$LOG" 2>&1
    rc=$?
    grep -E "完成|PPL|★|Error|Traceback" "$LOG" | tail -4
    [ "$rc" = 0 ] && [ -s "$OUT" ] || { M "FAIL rc=$rc"; tail -3 "$LOG"; continue; }
    gguf-tools/bench/anchor_metrics --ref-raw "$OUT" --ids "$IDS" --student "$STU" --row-out "/tmp/bugmd/$NM.$v.rows" | grep -E "KLD|Same top|Σmin|PPL" | tr '\n' ' '; echo
done
M "完成"
