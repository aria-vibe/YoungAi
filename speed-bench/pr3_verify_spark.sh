#!/bin/bash
# pr3_verify_spark.sh — 外部 PR(#3, M5 Metal 提速 + 三处共享路改动 + 服务端思考预算)在 spark 上的真模型门(2026-10-19)。
#
# 怎么跑: 在 PR 分支的目录(~/ds4-pr3, 已 make cuda-generic, gguf 软链到 ~/ds4-main/gguf)下
#   ./speed-bench/pr3_verify_spark.sh <基线二进制绝对路径, 如 ~/ds4-base/ds4> [跑哪几道, 默认 123]
#   第 2 个参数是道号串: "12" = 只跑 A/B 与预填尺(分支里没有思考预算时), "3" = 只跑服务端冒烟(思考预算拆成独立 PR #4 之后)。
#   日志全在 /tmp/pr3v/, 终端只打结论行。
# 前置: 生产 ds4-server 子进程已停(一台机器只许装一份 113 GB 底座, 铁律); 跑完由调用方 POST /api/models/load 把生产装回去。
#
# 三道门, 对着 PR 改了共享路的三处:
#   ① 真实请求 512 token 贪心+投机, 旧新逐字节 + t/s(d1_kv_ring_gate.sh ab 档): PR 的 DSpark 自动暂停就改在这条路上;
#      逐字节不同 = 核或调度改了数值, 不许放过; 同但 t/s 掉 = 自动暂停在 GB10 上是亏的。
#   ② 12k 预填尺块 2048 与 4096(prefill_ttft_ruler.sh, 自带 2500 MB 看门狗): PR 把共享头的默认块改成 4096,
#      core_v41.h 注释里 spark 实测 4096 时 MemAvailable 最低 3029 MB; 这里重量一次, 顺带比 2048 下 PR 对基线的 t/s。
#   ③ 服务端思考预算冒烟(PR 新功能): 8002 端口起 PR 的 ds4-server, 300 token 预算 + ```python 起手, 看 finish_reason、
#      思考段长度、正文是不是以代码块开头、日志里有没有"思考段到预算"那一行。
# 出错会怎样: ① 报"旧 ≠ 新" = 停下查核; ② 某块的 MemAvailable 最低 < 2500 会被尺自己杀掉并报 ★; ③ 正文不以 ```python 开头 = 起手串没强制上。
set -u
cd "$(dirname "$0")/.." || exit 1
BASE="${1:?用法: $0 <基线二进制绝对路径> [道号串, 默认 123]}"
STEPS="${2:-123}"
MODEL=gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf
AMP=gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-code_fit_n15360-engine
REQ=gguf/v41/night/review0923/req_20260922.json
OUT=/tmp/pr3v
mkdir -p "$OUT"
[ -x ./ds4 ] && [ -x ./ds4-server ] || { echo "★当前目录没有 ./ds4 ./ds4-server(先 make cuda-generic)★"; exit 1; }
case "$STEPS" in *[12]*) [ -x "$BASE" ] || { echo "★没有基线二进制 $BASE★"; exit 1; }; cp -f "$BASE" ./ds4.base ;; esac

case "$STEPS" in *1*)
echo "== ① 渲染真实请求 → ids"
./speed-bench/req_render_ids.sh "$REQ" "$OUT/req.ids" 2>&1 | tail -2
echo "== ① 真实请求 512 token 贪心+投机 旧新 A/B"
./speed-bench/d1_kv_ring_gate.sh none "$MODEL" 512 ab "$AMP" "$OUT/req.ids" ds4.base ds4 2>&1 | tail -6
cp -f /tmp/d1-kv-ring/ab_req_*.out /tmp/d1-kv-ring/ab_req_*.err "$OUT"/ 2>/dev/null
;; esac

case "$STEPS" in *2*)
echo "== ② 12k 预填尺 PR 二进制 块 2048 / 4096"
./speed-bench/prefill_ttft_ruler.sh speed-bench/promessi_sposi.txt 40000 "2048 4096" "$MODEL" "$AMP" ced ds4 2>&1 | tail -12
rm -rf "$OUT/ttft_pr"; cp -r /tmp/ttft "$OUT/ttft_pr"
echo "== ② 12k 预填尺 基线 块 2048"
./speed-bench/prefill_ttft_ruler.sh speed-bench/promessi_sposi.txt 40000 "2048" "$MODEL" "$AMP" ced ds4.base 2>&1 | tail -8
rm -rf "$OUT/ttft_base"; cp -r /tmp/ttft "$OUT/ttft_base"
echo "== ② 块 2048 生成文本 PR vs 基线"
if cmp -s "$OUT/ttft_pr/ced_2048.out" "$OUT/ttft_base/ced_2048.out"; then echo "  预填 2048: PR == 基线 逐字节同 ✓"; else echo "  ★预填 2048: PR ≠ 基线★"; fi
;; esac

case "$STEPS" in *3*)
./ds4-server --help 2>&1 | grep -q -- '--think-budget' || { echo "== ③ 这个分支的 ds4-server 没有 --think-budget, 跳过"; echo ALLDONE; exit 0; }
echo "== ③ 思考预算冒烟(ds4-server :8002, --think-budget 300 --budget-answer-prefix '\`\`\`python')"
./ds4-server --cuda -m "$MODEL" --zchain "$AMP" --mem-budget-mb 110000 --host 127.0.0.1 --port 8002 \
    --think-budget 300 --budget-answer-prefix '```python' > "$OUT/srv.log" 2>&1 &
SP=$!
for i in $(seq 1 180); do curl -s --noproxy '*' -m 2 http://127.0.0.1:8002/metrics >/dev/null 2>&1 && break; sleep 5; done
curl -s --noproxy '*' -m 1800 http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-reasoner","temperature":0,"max_tokens":1500,"thinking":{"type":"enabled"},"messages":[{"role":"user","content":"Write a Python function that returns the n-th Fibonacci number iteratively, with a short docstring."}]}' \
  > "$OUT/budget.json"
kill "$SP" 2>/dev/null; wait "$SP" 2>/dev/null
if command -v jq >/dev/null && jq -e . "$OUT/budget.json" >/dev/null 2>&1; then
  echo "  finish_reason=$(jq -r '.choices[0].finish_reason' "$OUT/budget.json") usage=$(jq -c '.usage' "$OUT/budget.json")"
  echo "  思考段字符数=$(jq -r '.choices[0].message.reasoning_content // "" | length' "$OUT/budget.json")"
  echo "  正文前 160 字符: $(jq -r '.choices[0].message.content // "" | .[0:160]' "$OUT/budget.json" | tr '\n' '⏎')"
else
  echo "  ★响应不是 JSON★: $(head -c 300 "$OUT/budget.json")"
fi
grep -a '思考段到预算\|think budget' "$OUT/srv.log" | head -3
;; esac
echo "ALLDONE"
