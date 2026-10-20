#!/bin/bash
# think_loop_locate_spark.sh — 思考段不收口的定位梯子第 3、4 级(2026-10-20, 用户批"3, 4 看看定位一下")。
#
# 第 4 级(先跑, 用着生产模型子进程): 同一条真实请求(trace 原样 JSON), 投机开 / 投机关 各 N 趟, 每趟从 trace 读:
#   思考多少位、有没有被检测器强制收口(周期多少)、结束原因、总生成数、用时、走的是不是纯解码(惩罚路)。
#   投机关 = 请求带 frequency_penalty 1e-30: fp32 下减不动任何 logit, 但引擎按"开了惩罚"走主机采样 + 纯解码(core_v41_api.c 三档注释),
#   与 kv_reuse_gate.sh 同一招 —— 不用换二进制, 顺带把设备采样核与主机采样路也分开了。
# 第 3 级(后跑, 要停模型子进程): 官方 model.py 读我们的量化权重文件(v41_judge.sh file: 档)对同一条序列逐位出 logits,
#   再 bugmd_gates_spark.sh engref 用裸引擎打同一条序列、按位置分桶比 KLD/same-top —— 同权重只差实现, 桶掉下来 = 引擎在那一段走样。
# 用法(spark 仓库根, nohup): speed-bench/think_loop_locate_spark.sh <请求 JSON> <序列.ids> <提示数> <入环位> [每档趟数 3]
#   例: speed-bench/think_loop_locate_spark.sh /tmp/bugmd/r3_req.json gguf/v41/night/faith_1016/pelican_1020.ids 11358 32341 3
# 产物: /tmp/bugmd/locate_<MMDD_HHMM>.txt 汇总; 每趟回答 /tmp/bugmd/locate_run_<档>_<趟>.json; engref 的 .metrics/.buckets 在 /tmp/bugmd。
# 出错会怎样: 子进程不是 idle 直接退; 某趟 curl 超时(3600 s)记"超时"继续下一趟; 第 3 级任一步失败如实写进汇总, 模型照样装回。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
M(){ echo "[locate $(date '+%m-%d %H:%M:%S')] $*"; }
REQ="${1:?请求 JSON}"; IDS="${2:?序列 ids}"; NP="${3:?提示数}"; ENTRY="${4:?入环位}"; N="${5:-3}"
GG="gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf"; QD="gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k"; HF="hf/DeepSeek-V4.1-Flash-1017"
OUT=/tmp/bugmd; mkdir -p "$OUT"; SUM="$OUT/locate_$(date +%m%d_%H%M).txt"
st=$(curl -s --noproxy '*' --max-time 5 127.0.0.1:8001/metrics | tr ',' '\n' | grep -o '"state":"[a-z]*"' | head -1)
[ "$st" = '"state":"idle"' ] || { M "★模型子进程 ${st:-不在}, 不是 idle, 退出★"; exit 1; }
TR="$(cat gguf/v41/night/trace/current)"; [ -s "$TR" ] || { M "★trace $TR 不存在★"; exit 1; }

# ---- 第 4 级: 投机开/关交替 ----
echo "## 第 4 级: 同一请求 投机开(spec) / 关(nospec, frequency_penalty 1e-30) 各 $N 趟" >> "$SUM"
printf '%-8s %-3s %-9s %-10s %-12s %-26s %-8s %s\n' 档 趟 思考位 生成总数 结束 强制收口 用时s 纯解码 >> "$SUM"
for i in $(seq 1 "$N"); do
  for mode in spec nospec; do
    python3 - "$REQ" "$mode" > "$OUT/locate_req.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); d["stream"] = False; d.pop("stream_options", None); d["max_tokens"] = 65536
if sys.argv[2] == "nospec": d["frequency_penalty"] = 1e-30
print(json.dumps(d, ensure_ascii=False))
EOF
    l0=$(wc -l < "$TR"); t0=$(date +%s)
    M "== $mode 第 $i 趟 发车"
    curl -s --noproxy '*' --max-time 3600 -H 'Content-Type: application/json' --data-binary @"$OUT/locate_req.json" 127.0.0.1:8001/v1/chat/completions > "$OUT/locate_run_${mode}_$i.json"
    rc=$?; t1=$(date +%s)
    for _ in $(seq 1 120); do [ "$(sed -n "$((l0+1)),\$p" "$TR" | grep -c -a '^--- token ids')" -ge 1 ] && break; sleep 1; done
    seg="$OUT/locate_seg_${mode}_$i.txt"; sed -n "$((l0+1)),\$p" "$TR" > "$seg"
    gen=$(grep -a -o '^--- token ids: prompt [0-9]*, generated [0-9]*' "$seg" | tail -1 | grep -o '[0-9]*$')
    fin=$(grep -a -o '^finish: [a-z_]*' "$seg" | tail -1 | cut -d' ' -f2)
    forced=$(grep -a -o 'think loop: [^-]*forced </think> at generated token [0-9]*' "$seg" | tail -1 | sed 's/think loop: token-exact repeat //; s/ forced <\/think> at generated token / @/')
    # 思考位 = trace "generated:" 那行里第一个 </think>(128822) 的下标 + 1; 没有就是整段都在思考
    think=$(grep -a '^generated:' "$seg" | tail -1 | sed 's/^generated: *//' | python3 -c 'import sys; a=sys.stdin.read().split(); print(a.index("128822")+1 if "128822" in a else "全段%d" % len(a))' 2>/dev/null)
    # 投机有没有开: /metrics 的 drafts_offered(每块一个值, 逗号连起来; 纯解码那趟应当是 0)
    pure=$(curl -s --noproxy '*' --max-time 5 127.0.0.1:8001/metrics | tr ',' '\n' | grep -o '"drafts_offered":[0-9]*' | cut -d: -f2 | paste -sd/ -)
    printf '%-8s %-3s %-9s %-10s %-12s %-26s %-8s %s\n' "$mode" "$i" "${think:-?}" "${gen:-?}" "${fin:-rc$rc}" "${forced:-无}" "$((t1-t0))" "$pure" >> "$SUM"
    M "$mode $i: 思考 ${think:-?} 生成 ${gen:-?} 结束 ${fin:-rc$rc} 强制 ${forced:-无} 用时 $((t1-t0)) s"
  done
done
cat "$SUM"

# ---- 第 3 级: 官方 model.py 读量化权重 vs 裸引擎, 同一条序列 ----
NT=$(wc -w < "$IDS")
echo "" >> "$SUM"; echo "## 第 3 级: engref $(basename "$IDS") $NT 位: 官方代码读 $(basename "$QD") vs 裸引擎" >> "$SUM"
st=$(curl -s --noproxy '*' --max-time 5 127.0.0.1:8001/metrics | tr ',' '\n' | grep -o '"state":"[a-z]*"' | head -1)
[ "$st" = '"state":"idle"' ] || { M "★子进程 $st 不是 idle, 第 3 级不跑★"; echo "子进程不 idle, 没跑" >> "$SUM"; exit 1; }
M "停模型子进程 $(pgrep -x ds4-server | tr '\n' ' ')"; pkill -x -TERM ds4-server
for _ in $(seq 1 60); do pgrep -x ds4-server >/dev/null || break; sleep 2; done
pgrep -x ds4-server >/dev/null && { pkill -x -KILL ds4-server; sleep 5; }
trap 'rc=$?; M "装回模型子进程"; bash gguf-tools/scripts/model_restart_spark.sh; echo "LOCATE_EXIT $rc"' EXIT
sleep 6
jtag="$(basename "$(dirname "$IDS")")_$(basename "$IDS" .ids)_n$NT"
REFBIN="gguf/v41judge/stu_${jtag}_file_$(basename "$QD").bin"
if [ ! -s "$REFBIN" ]; then
    M "官方代码读量化权重打分 → $REFBIN"
    bash gguf-tools/scripts/v41_judge.sh "$IDS" "$NT" --hf "$HF" "file:$QD" 2>&1 | grep -E "^\[judge\]|★|PPL|Same top" | tail -8 | tee -a "$SUM"
fi
[ -s "$REFBIN" ] || { M "★参考实现学生没出来★"; echo "参考实现失败" >> "$SUM"; exit 1; }
M "engref: 裸引擎 vs 参考实现"
bash speed-bench/bugmd_gates_spark.sh engref "$IDS" "$REFBIN" "$(basename "$IDS" .ids)_1017" "$NP" "$ENTRY" "$GG" 2>&1 | tail -30 | tee -a "$SUM"
M "汇总 → $SUM"
