#!/bin/bash
# render_diff_replay_spark.sh — 把一条真实请求重放给生产模型子进程(只要 1 个 token), 从 trace 取我们渲染出的提示 id,
# 与官方 encoding.py + tokenizer.json 的切法逐 token 对拍(render_diff_spark.py)。2026-10-20: 分词器预切分修复的端到端门。
# 用法(spark 仓库根): speed-bench/render_diff_replay_spark.sh <请求 JSON(trace 原样那行)> [hf 目录]
# 判读: 打印 "★逐 token 全同★" 才算过; 任何分叉都把两边前后 12 个 token 的原文印出来。
# 出错会怎样: 模型子进程不在 → curl 连不上; trace 没开 → 找不到 token ids 块(工作台起的子进程默认开着)。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
REQ="${1:?请求 JSON}"; HF="${2:-hf/DeepSeek-V4.1-Flash-1017}"
TR="$(cat gguf/v41/night/trace/current)"   # current 是个只写着 trace 路径的小文件, 不是软链
[ -s "$TR" ] || { echo "★trace 文件 $TR 不存在, 子进程没开 --trace?★"; exit 1; }
before=$(grep -c -a "^--- token ids" "$TR")
python3 - "$REQ" > /tmp/bugmd/replay_req.json <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); d["stream"] = False; d["max_tokens"] = 1; d.pop("stream_options", None)
print(json.dumps(d, ensure_ascii=False))
EOF
curl -s --noproxy '*' --max-time 600 -H 'Content-Type: application/json' --data-binary @/tmp/bugmd/replay_req.json 127.0.0.1:8001/v1/chat/completions > /tmp/bugmd/replay_resp.json
for _ in $(seq 1 60); do [ "$(grep -c -a "^--- token ids" "$TR")" -gt "$before" ] && break; sleep 1; done
n=$(grep -n -a "^--- token ids" "$TR" | tail -1 | cut -d: -f1)
sed -n "$((n+1))p" "$TR" | sed 's/^prompt: *//' > /tmp/bugmd/replay_prompt.ids
echo "trace 第 $n 行: $(sed -n "${n}p" "$TR"); 我们 $(wc -w < /tmp/bugmd/replay_prompt.ids) 个 id"
~/v41env/bin/python speed-bench/render_diff_spark.py "$HF" "$REQ" /tmp/bugmd/replay_prompt.ids
