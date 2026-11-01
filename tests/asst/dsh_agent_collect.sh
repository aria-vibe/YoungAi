#!/bin/bash
# dsh_agent_collect.sh — 官方模型在 dsh 里跑一批非图形 agent 任务并抓包, 打包送 spark(2026-10-26 晚, agent 多轮料的 Mac 侧)。
#   为什么在 Mac: dsh 装在 gguf/dsh/(Mac), 官方臂走 API 不要模型; 重采(要 ~/v41env 的 tokenizers 与 hf/ 目录)与渲染(ds4-server --render-anthropic 要 GGUF)在 spark。
# 用法: tests/asst/dsh_agent_collect.sh [--tasks gguf-tools/data/posttrain/mix_q_agent.txt] [--jobs 2] [--out gguf/dsh/captures/agent_<日期>] [--ship]
#   每道任务一趟 tests/asst/dsh_compare.sh official "<任务>" agent_NNN <out>/agent_NNN(抓包目录); --jobs 路并行(各自独立工作目录/代理端口);
#   --ship: 跑完 tar 送 spark:~/ds4-main/gguf/go-onebit/restore/agent_capture/ (接着在 spark 跑 api_agent.py resample / rows)。
# 出错会怎样: 单趟 dsh rc≠0 只记日志不中断批(它的抓包仍有效: 每轮请求体在 dsh 失败前就落盘了); 全批结束打每趟轮数。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT" || exit 1
TASKS=gguf-tools/data/posttrain/mix_q_agent.txt; JOBS=2; OUT="gguf/dsh/captures/agent_$(date +%m%d)"; SHIP=0; N=0
while [ $# -gt 0 ]; do case "$1" in --tasks) TASKS="${2:?}"; shift 2;; --jobs) JOBS="${2:?}"; shift 2;; --out) OUT="${2:?}"; shift 2;; --n) N="${2:?}"; shift 2;; --ship) SHIP=1; shift;; *) echo "★不认识的参数 $1★"; exit 2;; esac; done
[ -s "$TASKS" ] || { echo "★没有任务文件 $TASKS★"; exit 1; }
mkdir -p "$OUT"
grep -v '^#' "$TASKS" | grep -v '^$' | nl -ba -w3 -nrz > "$OUT/tasks.tsv"
if [ "$N" -gt 0 ]; then t=$(wc -l < "$OUT/tasks.tsv"); awk -v n="$N" -v t="$t" 'NR % int(t / n + 0.999) == 1' "$OUT/tasks.tsv" | head -n "$N" > "$OUT/tasks.pick" && mv "$OUT/tasks.pick" "$OUT/tasks.tsv"; fi   # --n: 等距抽 N 道(1 小时版 4 道)
echo "[agent] $(wc -l < "$OUT/tasks.tsv") 道任务, $JOBS 路, 抓包 → $OUT"
run_one() {   # $1 = 序号; 任务正文从 tasks.tsv 按序号取(不经 xargs 传: macOS xargs -I 的替换串上限 255 字节, 任务一行就超, 10-26 实撞 "command line cannot be assembled")
    local n="$1" cap="$OUT/agent_$1" msg
    msg="$(awk -F'\t' -v n="$1" '$1 == n { print $2 }' "$OUT/tasks.tsv")"
    [ -e "$cap" ] && { echo "[agent] agent_$n 已有抓包, 跳过"; return 0; }
    printf '%s\n' "$msg" | sed 's/\\n/\n/g' > "$OUT/task_$n.txt"
    tests/asst/dsh_compare.sh official "$(cat "$OUT/task_$n.txt")" "agent_$n" "$cap" > "$OUT/run_$n.log" 2>&1
    echo "[agent] agent_$n rc=$? 轮 $(ls "$cap"/req_*.json 2>/dev/null | wc -l | tr -d ' ')"
}
export -f run_one; export OUT
cut -f1 "$OUT/tasks.tsv" | xargs -P "$JOBS" -I{} bash -c 'run_one {}'
echo "[agent] 完成: $(ls -d "$OUT"/agent_* 2>/dev/null | wc -l | tr -d ' ') 趟, 共 $(ls "$OUT"/agent_*/req_*.json 2>/dev/null | wc -l | tr -d ' ') 轮请求体"
if [ "$SHIP" = 1 ]; then
    tar czf - -C "$(dirname "$OUT")" "$(basename "$OUT")" | ssh spark "mkdir -p ~/ds4-main/gguf/go-onebit/restore/agent_capture && tar xzf - -C ~/ds4-main/gguf/go-onebit/restore/agent_capture" && echo "[agent] 已送 spark:~/ds4-main/gguf/go-onebit/restore/agent_capture/$(basename "$OUT")"
fi
