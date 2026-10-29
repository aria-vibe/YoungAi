#!/bin/bash
# evo_rejudge_spark.sh — 把几套段 B(现役 + 自学习各轮回退留下的 ckpt)在同一把尺上各判一遍, 判完换回现役(2026-10-28)。
#
# 用法(spark ~/ds4-main, 工作台开着, 自学习一轮都没在跑):
#   bash gguf-tools/scripts/evo_rejudge_spark.sh <ids> <ntok> <名=amp目录>... > /tmp/bugmd/rejudge.log 2>&1
#   例: … gguf/go-onebit/code/judge_doc.ids 15360 a006=gguf/v41/posttrain/restore-api-a006/ckpt_e01 a007=gguf/v41/posttrain/restore-api-a007/ckpt_e01
#   现役(侧车里现在那套)总是第一个判, 名字叫"现役"; 读数存 gguf/go-onebit/evo/rejudge/<ids 名>/<指纹>/(judge.txt + rows), 同一套判过就跳过。
# 为什么要它(10-28 a002~a007 六轮五退): 编程尺 judge.ids 是 120 个 128 位窗首尾相接、无 BOS、相邻窗来自不相干文件, 训练料是连贯的整篇答案;
#   拿同一批语言的整篇文档尺(judge_doc.ids, 每语言 512 位带 BOS)对照, 分清"训练把代码推偏了"还是"尺的格式和料对不上"。
# 出错会怎样: 任何一步失败都先把现役 amp 拷回再退出; 退出前核指纹, 对不上就大字报出来(别再跑自学习, 先手动核侧车)。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
INNER=0; [ "${1:-}" = --_inner ] && { INNER=1; shift; }   # 内部: 停模型包装里回调自己
[ $# -ge 3 ] || { echo "用法: $0 <ids> <ntok> <名=amp目录>..."; exit 2; }
IDS=$1; N=$2; shift 2
HF=hf/DeepSeek-V4.1-Flash-1017; W=gguf-tools/scripts/with_model_stopped_spark.sh
mapfile -t PKS < gguf/serve_pick.txt; GG="${PKS[0]#"$ROOT"/}"; ARM="${PKS[1]#"$ROOT"/}"; GB=$(basename "$GG" .gguf); AB=$(basename "$ARM")
M(){ echo "[rej $(date '+%m-%d %H:%M:%S')] $*"; }
ampmd5(){ md5sum "$1/amp_L00.bin" 2>/dev/null | cut -c1-12; }
OUTD=gguf/go-onebit/evo/rejudge/$(basename "$IDS" .ids)
if [ "$INNER" = 0 ]; then
    pgrep -f "^bash gguf-tools/scripts/restore_B_api_round" >/dev/null && { M "★自学习一轮在跑, 不动侧车★"; exit 1; }
    LIVE=$(ampmd5 "$ARM"); [ -n "$LIVE" ] || { M "★侧车里没有 amp★"; exit 1; }
    SAVE=gguf/go-onebit/evo/rejudge/live_$LIVE; mkdir -p "$OUTD" "$SAVE"
    cp "$ARM"/amp_L*.bin "$SAVE/" && [ "$(ampmd5 "$SAVE")" = "$LIVE" ] || { M "★现役 amp 存不下来★"; exit 1; }
    M "现役 $LIVE; 尺 $IDS $N 位; 待判 $*"
    "$W" bash "$0" --_inner "$IDS" "$N" 现役="$SAVE" "$@"; rc=$?
    [ "$(ampmd5 "$ARM")" = "$LIVE" ] || { rm -f "$ARM"/amp_L*.bin; cp "$SAVE"/amp_L*.bin "$ARM/"; M "★★判完侧车不是现役, 已拷回 $LIVE, 模型要重装(with_model_stopped_spark.sh true)★★"; exit 1; }
    for d in "$OUTD"/*/; do printf '  %-12s %s  %s\n' "$(cat "$d/name" 2>/dev/null)" "$(basename "$d")" "$(grep -E "Mean KLD|Same top|Σmin" "$d/judge.txt" | sed 's/  */ /g' | paste -sd'|')"; done
    exit "$rc"
fi
one(){ local name=$1 dir=$2 fp stu
    fp=$(ampmd5 "$dir"); [ -n "$fp" ] || { M "★$name: $dir 里没有 amp★"; return 1; }
    [ -s "$OUTD/$fp/rows" ] && { M "$name($fp) 判过, 跳过"; return 0; }
    rm -f "$ARM"/amp_L*.bin; cp "$dir"/amp_L*.bin "$ARM/" || return 1
    stu="gguf/v41judge/stu_$(basename "$(dirname "$IDS")")_$(basename "$IDS" .ids)_n${N}_eng_${GB}_amp_${AB}"
    mkdir -p "$OUTD/$fp"; echo "$name" > "$OUTD/$fp/name"; M "$name($fp) 判 …"
    bash gguf-tools/scripts/v41_judge.sh "$IDS" "$N" --hf "$HF" "engine:$GG:$ARM" > "$OUTD/$fp/judge.txt" 2>&1 && cp "$stu.rows" "$OUTD/$fp/rows"; local r=$?
    rm -f "$stu.bin" "$stu".bin.*; return "$r"
}
rc=0; LIVEDIR=${1#*=}
for a in "$@"; do one "${a%%=*}" "${a#*=}" || { rc=1; break; }; done
rm -f "$ARM"/amp_L*.bin; cp "$LIVEDIR"/amp_L*.bin "$ARM/"   # 第一个是现役: 判完一律拷回
exit "$rc"
