#!/bin/bash
# kernel_ab_spark.sh — 引擎核改动的"两次装载 A/B"整链(spark 上跑; 2026-10-20 固化)。
#
# 怎么用: speed-bench/kernel_ab_spark.sh <旧二进制, 如 ds4.base_attn1020> [新二进制=ds4] [ids=/tmp/pr3v/req.ids] [生成 token=512]
# 做什么: 工作台模型子进程空闲就停它(主进程不动) → d1_kv_ring_gate.sh ab 档(1017 底座 + 现役编程侧车, 真实请求 req_20260922,
#   温 0 贪心+投机 512 token, 旧新逐字节 + t/s + 接受直方图) → 退出时按 gguf/serve_pick.txt 把模型装回(POST /api/models/load, 与页面同路)。
# 为什么要它: 10-18/10-19 三次都是手敲"停子进程 → ab → 装回"三段, pr3_verify_spark.sh 头注写着"前置: 生产已停; 跑完由调用方装回",
#   那两段每次都临时拼 —— 临时拼就会漏(装回忘了带口令 = 生产停在那里)。停/装回的写法逐字抄 think_rule_probe_spark.sh。
# 判读: "旧 == 新 逐字节同 ✓" 是硬门, 不同 = 核改了数值, 不许放过; t/s 差在 ±0.5 以内是噪声(同请求两趟 48.16 / 48.20)。
# 出错会怎样: 模型子进程不是 idle(有人在聊)直接退出不停它; ds4-server 没停干净会被 d1 的实例锁挡住; 装不回就自己去模型页点加载。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
M(){ echo "[ab $(date '+%H:%M:%S')] $*"; }
OLD="${1:?用法: $0 <旧二进制> [新二进制=ds4] [ids] [生成 token=512]}"
NEW="${2:-ds4}"; IDS="${3:-/tmp/pr3v/req.ids}"; NGEN="${4:-512}"
MODEL="gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf"
AMP="gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-code_fit_n15360-engine"
PORT=8000
[ -x "./$OLD" ] && [ -x "./$NEW" ] || { M "★没有 ./$OLD 或 ./$NEW★"; exit 1; }
[ -s "$IDS" ] || { M "★没有 $IDS(先 speed-bench/req_render_ids.sh gguf/v41/night/review0923/req_20260922.json $IDS)★"; exit 1; }
M "旧 $(md5sum "$OLD" | cut -c1-12) 新 $(md5sum "$NEW" | cut -c1-12)"
[ "$(md5sum < "$OLD")" != "$(md5sum < "$NEW")" ] || { M "★旧新是同一个二进制, 没东西可比★"; exit 1; }

restore_model() {
    local pg pz tok body resp
    pg=$(sed -n 1p gguf/serve_pick.txt); pz=$(sed -n 2p gguf/serve_pick.txt)
    pg="${pg#"$ROOT"/}"; pz="${pz#"$ROOT"/}"; [ -n "$pz" ] || pz=none
    tok=$(cat ~/.ds4/assistant/token 2>/dev/null || true)
    body=$(printf '{"gguf":"%s","sidecar":"%s"}' "$pg" "$pz")
    resp=$(curl -s --noproxy '*' --max-time 30 ${tok:+-H "Authorization: Bearer $tok"} -H 'Content-Type: application/json' -d "$body" "http://127.0.0.1:$PORT/api/models/load")
    M "装回 $pg + $pz → $resp"
}

# 只在模型空闲时停它: 有人正在聊就不碰(用户的请求比探针优先)
st=$(curl -s --noproxy '*' --max-time 5 127.0.0.1:8001/metrics | tr ',' '\n' | grep -o '"state":"[a-z]*"' | head -1)
[ -z "$st" ] || [ "$st" = '"state":"idle"' ] || { M "★模型子进程 $st, 不是 idle, 不停★"; exit 1; }
stopped=0
if pgrep -x ds4-server >/dev/null; then
    M "停模型子进程"; pkill -x -TERM ds4-server; stopped=1
    for _ in $(seq 1 60); do pgrep -x ds4-server >/dev/null || break; sleep 2; done
    pgrep -x ds4-server >/dev/null && { pkill -x -KILL ds4-server; sleep 5; }
    pgrep -x ds4-server >/dev/null && { M "★ds4-server 杀不掉★"; exit 1; }
    sleep 5
fi
[ "$stopped" = 1 ] && trap 'rc=$?; restore_model; echo "AB_EXIT $rc"' EXIT
M "available $(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo) GB; 时钟 $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader)"
./speed-bench/d1_kv_ring_gate.sh none "$MODEL" "$NGEN" ab "$AMP" "$IDS" "$OLD" "$NEW" 2>&1 | tail -6
rc=${PIPESTATUS[0]}
for b in "$OLD" "$NEW"; do   # 接受直方图 + 一轮分账(ab 档只打 t/s 行, 这两行在 .err 里)
    f=/tmp/d1-kv-ring/ab_$(basename "$IDS" .ids)_$b.err
    [ -s "$f" ] && echo "  $b: $(grep -a -h '接受直方图\|直方图' "$f" | tail -1 | cut -c1-160)"
done
exit "$rc"
