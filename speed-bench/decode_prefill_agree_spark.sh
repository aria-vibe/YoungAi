#!/bin/bash
# decode_prefill_agree_spark.sh — 解码路 vs 预填路 全轨迹逐位一致率(2026-10-20, 用户"倾向引擎里有个魔法数字")。
#
# 问的是: 几万步解码自己写进 KV 的状态, 和把同一串 id 整段预填出来的状态, 是不是一回事。
#   ①用生产服务对一条真实请求发温 0(贪心)一趟, 上限 N: 真吐的每一位 = 解码路(投机验证批 / 纯解码)在该位的 argmax;
#   ②停模型子进程, 把提示 + 真吐整段按 id 预填重打分 top-32(bugmd_gates_spark.sh scoreids, 块 512);
#   ③bugmd_ids_tools.py agree: 每千位 真吐 == 预填 top-1 的比例。两条路都对就只剩浮点近平局, 曲线平着; 某偏移起掉 / 周期性塌 = 魔法数字。
# 用法(spark 仓库根, nohup): speed-bench/decode_prefill_agree_spark.sh <请求 JSON(trace 原样)> <标签> [上限 24576] [反修目录(默认现役编程侧车)]
#   贪心配思考必进长周期环(10-19), 不影响这把尺 —— 环里每一位照样是 argmax。
# 产物: gguf/v41/night/faith_1016/<标签>.ids(提示+真吐), /tmp/bugmd/<标签>_top32_*.bin, 一致率表打在 stdout。退出时装回模型。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
M(){ echo "[agree $(date '+%m-%d %H:%M:%S')] $*"; }
REQ="${1:?请求 JSON}"; TAG="${2:?标签}"; N="${3:-24576}"
AMP="${4:-gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-code_fit_n15360-engine}"
GG="gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf"
st=$(curl -s --noproxy '*' --max-time 5 127.0.0.1:8001/metrics | tr ',' '\n' | grep -o '"state":"[a-z]*"' | head -1)
[ "$st" = '"state":"idle"' ] || { M "★模型子进程 ${st:-不在}, 不是 idle★"; exit 1; }
TR="$(cat gguf/v41/night/trace/current)"; l0=$(wc -l < "$TR")
python3 - "$REQ" "$N" > /tmp/bugmd/agree_req.json <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); d["stream"] = False; d.pop("stream_options", None); d["max_tokens"] = int(sys.argv[2]); d["temperature"] = 0.0
print(json.dumps(d, ensure_ascii=False))
EOF
M "温 0 发车, 上限 $N"
curl -s --noproxy '*' --max-time 3600 -H 'Content-Type: application/json' --data-binary @/tmp/bugmd/agree_req.json 127.0.0.1:8001/v1/chat/completions > "/tmp/bugmd/${TAG}_greedy.json"
for _ in $(seq 1 120); do [ "$(sed -n "$((l0+1)),\$p" "$TR" | grep -c -a '^--- token ids')" -ge 1 ] && break; sleep 1; done; sleep 5
L=$(grep -n -a "^--- token ids" "$TR" | tail -1 | cut -d: -f1)
NP=$(sed -n "${L}p" "$TR" | grep -o 'prompt [0-9]*' | grep -o '[0-9]*')
{ sed -n "$((L+1))p" "$TR" | sed 's/^prompt: *//'; sed -n "$((L+2))p" "$TR" | sed 's/^generated: *//'; } | tr -s ' \n' ' ' | sed 's/ *$/\n/' > "gguf/v41/night/faith_1016/$TAG.ids"
M "$(sed -n "${L}p" "$TR"); ids $(wc -w < "gguf/v41/night/faith_1016/$TAG.ids"); 本趟 think loop $(sed -n "$((l0+1)),\$p" "$TR" | grep -a -c 'think loop') 次"
bash speed-bench/think_close_probe_spark.sh "$GG" "gguf/v41/night/faith_1016/$TAG.ids:$NP:0" -- "$AMP" 2>&1 | grep -E "^\[close\]|★" | tail -6
BIN="/tmp/bugmd/${TAG}_top32_$(basename "$AMP" | sed 's/^DeepSeek-V4.1-Flash-//; s/-engine$//').bin"
[ -s "$BIN" ] || { M "★没有 $BIN★"; exit 1; }
python3 speed-bench/bugmd_ids_tools.py agree "$BIN" "gguf/v41/night/faith_1016/$TAG.ids" "$NP" 1000
