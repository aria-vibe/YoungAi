#!/bin/bash
# decoder_replay_scan_spark.sh — 预填最后一块"解码器段只回放末尾 K 行"的扫尺(spark 上跑; 2026-10-20, docs/deepinfer_compare.md §3.2)。
#
# 怎么用: speed-bench/decoder_replay_scan_spark.sh [K 列表="0 134 512 1024"] [ids=/tmp/pr3v/req.ids] [生成 token=512] [二进制=ds4]
#   ids 可以给多个, 用逗号分开: "/tmp/pr3v/req.ids,/tmp/pr3v/req1500.ids"(短提示那条看 TTFT 的大头)。
# 做什么: 工作台模型子进程空闲就停它 → 同一二进制、同一真实请求, 每个 K 跑一趟 贪心 + 投机(--decoder-replay K; K=0 = 现状整块跑满)
#   → 一张表: 预填秒/t/s、解码 t/s、投机接受均值与直方图、输出对 K=0 的同前缀字节数 → 退出时按 gguf/serve_pick.txt 装回模型。
# 判读: 这刀换的是"官方线上同款近似"(keep_tokens=140), 不是逐字节同 —— 主判是接受率与输出同前缀长度(首 token 变了前缀就是 0):
#   接受均值在噪声内(±0.1/5)且首 token 不变的最小 K 才可取; 接受掉 0.3 以上就是 09-29 那种隐态变差, 判负。
# 出错会怎样: 模型子进程不是 idle 直接退出; ds4-server 没停干净会被实例锁挡住(看 .err 的"实例锁"行); 装不回就去模型页点加载。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
M(){ echo "[replay $(date '+%H:%M:%S')] $*"; }
KS="${1:-0 134 512 1024}"; IDSL="${2:-/tmp/pr3v/req.ids}"; NGEN="${3:-512}"; BIN="${4:-ds4}"
MODEL="gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf"
AMP="gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-code_fit_n15360-engine"
PORT=8000; OUT=/tmp/replay_scan; mkdir -p "$OUT"
WD_KILL_MB=2500   # 与 d1_kv_ring_gate.sh 同口径的看门狗红线
[ -x "./$BIN" ] || { M "★没有 ./$BIN★"; exit 1; }

restore_model() {
    local pg pz tok body resp
    pg=$(sed -n 1p gguf/serve_pick.txt); pz=$(sed -n 2p gguf/serve_pick.txt)
    pg="${pg#"$ROOT"/}"; pz="${pz#"$ROOT"/}"; [ -n "$pz" ] || pz=none
    tok=$(cat ~/.ds4/assistant/token 2>/dev/null || true)
    body=$(printf '{"gguf":"%s","sidecar":"%s"}' "$pg" "$pz")
    resp=$(curl -s --noproxy '*' --max-time 30 ${tok:+-H "Authorization: Bearer $tok"} -H 'Content-Type: application/json' -d "$body" "http://127.0.0.1:$PORT/api/models/load")
    M "装回 $pg + $pz → $resp"
}
run_one() {   # $1=ids $2=K $3=标签 → $OUT/$3.out/.err; 看门狗与 d1_kv_ring_gate.sh 的 run() 同式
    local ids=$1 k=$2 tag=$3 pid bad=0 a
    local x=(); [ "$k" != 0 ] && x=(--decoder-replay "$k")
    ./"$BIN" -m "$MODEL" --zchain "$AMP" --temp 0 --seed 1 -n "$NGEN" --gen-ids "$ids" ${x[@]+"${x[@]}"} > "$OUT/$tag.out" 2> "$OUT/$tag.err" &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
        if [ "$a" -lt "$WD_KILL_MB" ]; then bad=$((bad+1)); else bad=0; fi
        if [ "$bad" -ge 2 ]; then echo "★看门狗: MemAvailable ${a} MB < ${WD_KILL_MB} 连续两次, 杀 $tag★" | tee -a "$OUT/$tag.err"; kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null; break; fi
        sleep 2
    done
    wait "$pid" 2>/dev/null
}

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
[ "$stopped" = 1 ] && trap 'rc=$?; restore_model; echo "SCAN_EXIT $rc"' EXIT
M "二进制 $(md5sum "$BIN" | cut -c1-12); available $(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo) GB; 时钟 $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader)"

IFS=',' read -r -a IDSA <<< "$IDSL"
for ids in "${IDSA[@]}"; do
    [ -s "$ids" ] || { M "★没有 $ids★"; continue; }
    base=$(basename "$ids" .ids)
    echo "== $ids ($(wc -w < "$ids") token), 生成 $NGEN, 温 0 贪心+投机, 二进制 $BIN =="
    printf '%6s | %-26s | %-10s | %-44s | %s\n' K 预填 解码 "投机接受(均值/直方图)" "对 K=0 同前缀字节 / 总字节"
    for k in $KS; do
        tag="${base}_K${k}"
        run_one "$ids" "$k" "$tag"
        pf=$(grep -a -h -o 'prefill [0-9]* token [0-9.]*s ([0-9.]* t/s)' "$OUT/$tag.err" | tail -1 | sed 's/prefill [0-9]* token //')
        dc=$(grep -a -h -o 'decode [0-9]* token [0-9.]*s ([0-9.]* t/s)' "$OUT/$tag.err" | tail -1 | grep -o '([0-9.]* t/s)' | tr -d '()')
        ac=$(grep -a -h 'DSpark:' "$OUT/$tag.err" | tail -1 | sed 's/.*平均接受 //; s/ 位; 直方图 / /' | cut -c1-44)
        tot=$(wc -c < "$OUT/$tag.out")
        if [ "$k" = 0 ]; then same="$tot(基线)"
        else
            d=$(cmp "$OUT/${base}_K0.out" "$OUT/$tag.out" 2>&1 | head -1)
            # GNU cmp 新版打 "differ: char N", 旧版打 "byte N"; N 是首个不同的位置(1 起) ⇒ 同前缀 = N−1
            if [ -z "$d" ]; then same="$tot(逐字节同)"; else same="$(echo "$d" | grep -oE '(byte|char) [0-9]+' | awk '{print $2-1}') / $tot"; fi
        fi
        grep -a -q '实例锁\|失败\|★' "$OUT/$tag.err" && M "  ★$tag 的 .err 里有报错行: $(grep -a '实例锁\|失败\|★' "$OUT/$tag.err" | head -1 | cut -c1-120)★"
        printf '%6s | %-26s | %-10s | %-44s | %s\n' "$k" "${pf:-?}" "${dc:-?}" "${ac:-?}" "$same"
    done
done
