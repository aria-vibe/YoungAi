#!/bin/bash
# graphics_sidecar_chain_spark.sh — 图形代码域侧车一条龙(2026-10-24 用户"启动吧", 方案 docs/graphics_sidecar.md §3):
#   C0 等官方 API 采料完成(think_trace_collect.py collect 在后台跑, 日志 /tmp/bugmd/graphics_collect.log 尾行 "collect 完成")
#   C1 提料   think_trace_collect.py code: 每条取最长代码围栏 → gguf-tools/data/corpus/graphics/<子域>.jsonl; ★与三份鹈鹕判决页面共享 16-gram 的整篇剔除★
#   C2 侧车   v41_sidecars_all_spark.sh --gguf <现役 ①> --hf <1017> graphics: idshalf 切 fit/judge(15360 = 6 × 2560) → grrb 解 → 本域五指标 + wt2 守门(它自己停/装回模型)
#   C3 主尺   html_region_judge_spark.sh --zc <新②> 三份鹈鹕页面: 坐标区 KLD 对现役编程 ②(同尺已有读数, 10-24 14:59)
# 用法(spark ~/ds4-main, nohup): gguf-tools/scripts/graphics_sidecar_chain_spark.sh [--from C0|C1|C2|C3]
# 停车规则: 任一段非 0 停在那一段, 尾行 CHAIN_EXIT <rc>。C1 子域不足 6 个或某子域 < 6 篇 = 料太薄, 停(15360 的切法要每子域 20 窗)。
# 出错会怎样: C0 等的是日志尾行, collect 进程死了没写尾行会一直等 —— 同时查进程, 进程没了且无尾行就停并打 ★。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
FROM="C0"; while [ $# -gt 0 ]; do case "$1" in --from) FROM="${2:?}"; shift 2;; *) echo "★不认识的参数 $1★"; exit 2;; esac; done
HF="hf/DeepSeek-V4.1-Flash-1017"; PY="$HOME/v41env/bin/python"; TC="gguf-tools/scripts/think_trace_collect.py"
RAW="gguf-tools/data/posttrain/graphics_raw.jsonl"; CLOG="/tmp/bugmd/graphics_collect.log"; CORP="gguf-tools/data/corpus/graphics"
PEL="gguf/go-onebit/code/pelican"
mapfile -t PK < gguf/serve_pick.txt; GG="${PK[0]#"$ROOT"/}"
M(){ echo "[gchain $(date '+%m-%d %H:%M:%S')] $*"; }
die(){ M "★$*★"; echo "CHAIN_EXIT 1"; exit 1; }
stage_ge(){ local o="C0 C1 C2 C3"; local a=${o%%$1*}; local b=${o%%$FROM*}; [ ${#a} -ge ${#b} ]; }

if stage_ge C0; then
  M "C0 等采料: $CLOG"
  while ! grep -q "^collect 完成" "$CLOG" 2>/dev/null; do
    pgrep -f "think_trace_collect[.]py collect" >/dev/null || die "采料进程不在且日志没有尾行(看 $CLOG)"
    sleep 60
  done
  tail -1 "$CLOG"
fi

if stage_ge C1; then
  M "C1 提料 → $CORP(剔除与 $PEL/*.ids 共享 16-gram 的整篇)"
  "$PY" "$TC" code --raw "$RAW" --out "$CORP" --hf "$HF" --exclude-ids "$PEL"/official_pel1.ids "$PEL"/official_pel2.ids "$PEL"/local_pel_s1.ids || die "C1 提料失败"
  n=$(ls "$CORP"/*.jsonl 2>/dev/null | wc -l); [ "$n" -ge 6 ] || die "C1 子域只有 $n 个(要 6)"
  for f in "$CORP"/*.jsonl; do c=$(grep -c . "$f"); [ "$c" -ge 6 ] || die "C1 $(basename "$f") 只有 $c 篇(< 6, 每子域 20 窗切不出)"; echo "  $(basename "$f") $c 篇"; done
fi

if stage_ge C2; then
  M "C2 侧车: v41_sidecars_all_spark.sh --gguf $GG --hf $HF graphics"
  bash gguf-tools/scripts/v41_sidecars_all_spark.sh --gguf "$GG" --hf "$HF" graphics || die "C2 侧车链失败"
fi

if stage_ge C3; then
  ZC=$(ls -d gguf/v41/"$(basename "$GG" .gguf)"-grrb-graphics_fit_n15360-engine 2>/dev/null | head -1)
  [ -n "$ZC" ] && grep -q "^# 完成" "$ZC/manifest.txt" || die "C3 找不到完成的图形侧车目录(gguf/v41/*-grrb-graphics_fit_n15360-engine)"
  M "C3 主尺: 三份鹈鹕页面 × (① / ①+图形② / ①+图形②+③)"
  bash gguf-tools/scripts/with_model_stopped_spark.sh bash gguf-tools/scripts/html_region_judge_spark.sh --zc "$ZC" "$PEL"/official_pel1.html "$PEL"/official_pel2.html "$PEL"/local_pel_s1.html || die "C3 判决失败"
fi
M "全链收工"; echo "CHAIN_EXIT 0"
