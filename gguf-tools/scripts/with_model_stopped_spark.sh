#!/bin/bash
# with_model_stopped_spark.sh — 停下工作台的空闲模型子进程, 跑一条命令, 跑完按 gguf/serve_pick.txt 把模型装回(2026-10-20 固化)。
#
# 怎么用: gguf-tools/scripts/with_model_stopped_spark.sh <命令...>        (在 ~/ds4-main 下, 工作台 ds4-train :8000 开着)
#   例: with_model_stopped_spark.sh bash -c 'gguf-tools/scripts/v41_judge.sh … ; speed-bench/decoder_replay_scan_spark.sh …'
# 为什么要它: 判决尺(v41_judge.sh)、探针、A/B 都要 113 GB 的机器空出来, 10-17 起每个脚本各抄一份"停子进程 → 干活 → 装回"
#   (kernel_ab_spark.sh / model_restart_spark.sh / decoder_replay_scan_spark.sh 三处同一段), 这里收成一处; 被包的脚本再碰到"模型没在跑"
#   就自然跳过自己那段停/装回。停/装回的写法逐字抄 model_restart_spark.sh。
# 判读: 开头打 "停模型子进程 pid …", 末尾打 "装回 … → {ok:true}" 与 "WRAP_EXIT <被包命令的 rc>"。
# 出错会怎样: 子进程不是 idle(有人在聊)直接退出不停它(rc 1); 装回没带口令会 401(口令在 ~/.ds4/assistant/token); 装不回就去模型页点加载。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
M(){ echo "[wrap $(date '+%m-%d %H:%M:%S')] $*"; }
PORT=8000
[ $# -ge 1 ] || { echo "用法: $0 <命令...>"; exit 2; }

restore_model() {
    local pg pz tok body resp
    pg=$(sed -n 1p gguf/serve_pick.txt); pz=$(sed -n 2p gguf/serve_pick.txt)
    pg="${pg#"$ROOT"/}"; pz="${pz#"$ROOT"/}"; [ -n "$pz" ] || pz=none
    tok=$(cat ~/.ds4/assistant/token 2>/dev/null || true)
    body=$(printf '{"gguf":"%s","sidecar":"%s"}' "$pg" "$pz")
    resp=$(curl -s --noproxy '*' --max-time 30 ${tok:+-H "Authorization: Bearer $tok"} -H 'Content-Type: application/json' -d "$body" "http://127.0.0.1:$PORT/api/models/load")
    M "装回 $pg + $pz → $resp"
}

st=$(curl -s --noproxy '*' --max-time 5 127.0.0.1:8001/metrics | tr ',' '\n' | grep -o '"state":"[a-z]*"' | head -1)
[ -z "$st" ] || [ "$st" = '"state":"idle"' ] || { M "★模型子进程 $st, 不是 idle, 不停★"; exit 1; }
stopped=0
if pgrep -x ds4-server >/dev/null; then
    M "停模型子进程 $(pgrep -x ds4-server | tr '\n' ' ')"; pkill -x -TERM ds4-server; stopped=1
    for _ in $(seq 1 60); do pgrep -x ds4-server >/dev/null || break; sleep 2; done
    pgrep -x ds4-server >/dev/null && { pkill -x -KILL ds4-server; sleep 5; }
    pgrep -x ds4-server >/dev/null && { M "★ds4-server 杀不掉★"; exit 1; }
    sleep 6   # 主进程要看到子进程退出、mode 离开 serving(≈6 s), 否则之后 load 回 409
fi
M "available $(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo) GB; 时钟 $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader); 跑: $*"
"$@"
rc=$?
[ "$stopped" = 1 ] && restore_model
echo "WRAP_EXIT $rc"
exit "$rc"
