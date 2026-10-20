#!/bin/bash
# think_stop_probe.sh — "思考停不下来"定位探针(2026-10-19)。
# 问题: LCB 20 题里 14 题撞长度上限(作者 PR #4 数据), 8k 硬掐+代码起手反而 13/20 ≈ 全精度 15/20,
# 说明解在、"停"的决策坏了。这里不下结论, 只量三件事, 全部打现役服务, 不动生产进程:
#   ① 同一题 low/high/max 三档思考各采一次: 思考长度是否随档位单调(官方模板里档位是提示侧的数字 50/75/100,
#      模型本来按它控长; 三档都撞墙 = 控长回路整个失效, 不是"略啰嗦");
#   ② 收没收口(finish_reason=length 且 content 空 = 撞上限没停);
#   ③ 思考末尾原文 + "已确认/再验一遍"类用语计数, 给人眼分 A(内容上已做完还不停)/B(验证真的一直失败)。
# 用法: think_stop_probe.sh [--temp T] PORT TASK_ID MAX_TOKENS EFFORT [EFFORT...]
#   例: think_stop_probe.sh 8001 HumanEval/129 24576 low high max
#       think_stop_probe.sh --temp 0 8001 HumanEval/129 24576 high max   # 贪心臂: 不传 --temp 就是服务端默认采样(温 1)
# 三档并发发请求(服务端合批, 墙钟≈一趟); 原始回包落 gguf-tools/reports/think_stop_<task>_<effort>[_tT]_<时间>.json。
# ★带上限的趟只能说"到上限没停", 不能说"永不停"(10-19 铁律); 上限是为探针分钟级, 判决要不设上限。
# --temp 是为"贪心是不是不收口的原因"这一问加的(10-19): 温 0 下 top-1 锁死, 一进环就是逐字复读, 只能靠死环检测器强制 </think>;
#   温 1 的同题收不收是概率。两臂同题同档对比, 才分得清"模型不收口"和"贪心把它钉在环里"。
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../.." && pwd)
TEMP=""
if [ "${1:-}" = "--temp" ]; then TEMP=$2; shift 2; fi
[ $# -ge 4 ] || { sed -n 2,16p "$0" >&2; exit 2; }
PORT=$1; TASK=$2; MAXTOK=$3; shift 3
DATA="$ROOT/gguf-tools/bench/data/HumanEval.jsonl"
STAMP=$(date +%Y%m%d_%H%M%S)
RDIR="$ROOT/gguf-tools/reports"; mkdir -p "$RDIR"
SAFE=${TASK//\//_}
export no_proxy='*' NO_PROXY='*'; unset http_proxy https_proxy 2>/dev/null || true

PROMPT=$(python3 - "$DATA" "$TASK" <<'PEOF'
import json, sys
for l in open(sys.argv[1]):
    d = json.loads(l)
    if d["task_id"] == sys.argv[2]:
        print("Complete the following Python function. Return the full function in one ```python block.\n\n" + d["prompt"]); break
else:
    sys.exit("task not found: " + sys.argv[2])
PEOF
) || exit 2

one() {   # one EFFORT → 回包落盘 + 一行摘要
    local eff=$1 t0 t1
    local tag=${TEMP:+_t$TEMP}
    local out="$RDIR/think_stop_${SAFE}_${eff}${tag}_${STAMP}.json"   # 与 eff 分两行: 同一条 local 里先展开后赋值, eff 还没绑定
    t0=$(date +%s)
    curl -s --noproxy '*' -m 7200 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(python3 -c 'import json,sys
b={"model":"ds4","max_tokens":int(sys.argv[2]),"reasoning_effort":sys.argv[3],"messages":[{"role":"user","content":sys.argv[1]}]}
if sys.argv[4]: b["temperature"]=float(sys.argv[4])   # 不传就不带字段: 服务端默认采样, 与 18:42 那趟同口径
print(json.dumps(b))' "$PROMPT" "$MAXTOK" "$eff" "$TEMP")" \
        > "$out"
    t1=$(date +%s)
    python3 - "$out" "$eff" "$((t1-t0))" "$TEMP" <<'PEOF'
import json, sys, re
r = json.load(open(sys.argv[1])); secs = sys.argv[3]
eff = sys.argv[2] + (" 温" + sys.argv[4] if sys.argv[4] else "")
c = r["choices"][0]; m = c["message"]
th = m.get("reasoning_content") or ""; body = m.get("content") or ""
u = r.get("usage", {})
done_w = len(re.findall(r"\b(matches|correct|confirmed|verified|works|final answer)\b", th, re.I))
again_w = len(re.findall(r"\b(wait|double[- ]check|let me (?:re|verify|check)|actually|hmm|but)\b", th, re.I))
closed = c.get("finish_reason") == "stop" and bool(body.strip())
# 逐字复读指纹: 思考末尾 2000 字符里最长的"紧挨着重复"的片段长度(贪心环的直接证据; 温 1 的环通常不逐字)
tail = th[-2000:]; rep = 0
for L in range(8, 400):
    if any(tail[i:i+L] == tail[i+L:i+2*L] for i in range(0, max(1, len(tail)-2*L), 7)): rep = L
print(f"[{eff}] finish={c.get('finish_reason')} 收口={'是' if closed else '否'} completion_tokens={u.get('completion_tokens')} "
      f"思考={len(th)}字符 正文={len(body)}字符 用时={secs}s 已确认类词={done_w} 再验类词={again_w} 末尾逐字复读片段={rep}字符 → {sys.argv[1]}")
print(f"[{eff}] 思考末尾 ↓\n" + th[-700:].replace("\n", "⏎") + "\n")
PEOF
}

for E in "$@"; do one "$E" & done
wait
echo "[think_stop_probe] 完成 task=$TASK maxtok=$MAXTOK → $RDIR/think_stop_${SAFE}_*_${STAMP}.json"
