#!/bin/bash
# model_restart_spark.sh — spark 上换新 ds4-server 二进制: 停空闲的模型子进程 → 按 gguf/serve_pick.txt 装回(2026-10-20 固化)。
#
# 怎么用: gguf-tools/scripts/model_restart_spark.sh        (在 ~/ds4-main 下, 工作台 ds4-train :8000 开着)
# 为什么要它: 工作台主进程不自动重拉退出的子进程, /api/models/load 在同一套正 live 时回 409 "已经在用这一套"——
#   所以换二进制只能"先杀再装"。10-17 起每次重编都手敲这两段(kernel_ab_spark.sh 的 restore_model 同一份写法), 这里固化。
# 判读: 最后打印 "装回 … → {…}" 与子进程 pid; 等到 127.0.0.1:8001/metrics 回 live.state=idle 才算起来了(113 GB 底座 ≈ 85~92 s)。
# 出错会怎样: 子进程不是 idle(有人在聊)直接退出不停它; 装回没带口令会 401(口令在 ~/.ds4/assistant/token); 装不回就去模型页点加载。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
M(){ echo "[restart $(date '+%H:%M:%S')] $*"; }
PORT=8000

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
if pgrep -x ds4-server >/dev/null; then
    M "停模型子进程 $(pgrep -x ds4-server | tr '\n' ' ')(二进制 $(md5sum ds4-server | cut -c1-12))"; pkill -x -TERM ds4-server
    for _ in $(seq 1 60); do pgrep -x ds4-server >/dev/null || break; sleep 2; done
    pgrep -x ds4-server >/dev/null && { pkill -x -KILL ds4-server; sleep 5; }
    pgrep -x ds4-server >/dev/null && { M "★ds4-server 杀不掉★"; exit 1; }
    sleep 6   # 主进程要看到子进程退出、mode 离开 serving(≈6 s), 否则 load 回 409
fi
restore_model
for _ in $(seq 1 90); do
    st=$(curl -s --noproxy '*' --max-time 3 127.0.0.1:8001/metrics 2>/dev/null | tr ',' '\n' | grep -o '"state":"[a-z]*"' | head -1)
    [ "$st" = '"state":"idle"' ] && { M "起来了: pid $(pgrep -x ds4-server | tr '\n' ' ')"; exit 0; }
    sleep 3
done
M "★270 s 内 8001 没回 idle, 去看 gguf/v41/posttrain/ui_logs 最新的 server_*.log★"; exit 1
