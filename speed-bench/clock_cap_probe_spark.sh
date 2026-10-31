#!/bin/bash
# clock_cap_probe_spark.sh — 量 GPU 限频(nvidia-smi -lgc)拿多少速度换多少温度(spark 上跑; 2026-10-30)。
#
# 怎么用: 限频/不限频各跑一趟, 标签自己起, 比两趟的汇总行:
#   speed-bench/clock_cap_probe_spark.sh stock          # 默认频率
#   sudo nvidia-smi -lgc 0,2200
#   speed-bench/clock_cap_probe_spark.sh cap2200
#   cmp /tmp/clock-cap/stock.out /tmp/clock-cap/cap2200.out   # 频率不该改数值, 不同 = 有 bug
#   sudo nvidia-smi -rgc                                # 撤销限频
# 做什么: 真实请求 req_20260922(贪心 + 投机, 温 0)跑一段长生成, 每 5 秒采 SM 频率/功耗/GPU 温度/ACPI 最高温; 汇总取后半程(前半程还在升温)。
# 判读: decode 那行 t/s 是速度, 后半程均值是稳态温度。t/s 差 ±0.5 以内是噪声(kernel_ab_spark.sh 同请求两趟 48.16 / 48.20)。
#   满载 SM 低于 2000 MHz 且没限频 = PD 卡死(见 fable5 10-17), 两趟都作废, 先冷断电。
# 出错会怎样: 工作台模型子进程在跑会被实例锁挡住, 所以这里直接退出, 先在模型页卸载; MemAvailable 连续两次低于 2500 MB 杀进程(与 d1_kv_ring_gate.sh 同红线)。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
TAG="${1:?用法: $0 <标签> [生成 token=16384]}"; NGEN="${2:-16384}"
MODEL="gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf"
AMP="gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-code_fit_n15360-engine"
IDS="/tmp/pr3v/req.ids"
OUT="/tmp/clock-cap"; WD_KILL_MB=2500
mkdir -p "$OUT"
[ -s "$IDS" ] || { echo "★没有 $IDS(先 speed-bench/req_render_ids.sh gguf/v41/night/review0923/req_20260922.json $IDS)★"; exit 1; }
pgrep -x ds4-server >/dev/null && { echo "★ds4-server 在跑, 先在模型页卸载★"; exit 1; }
pgrep -x ds4 >/dev/null && { echo "★已有 ds4 在跑★"; exit 1; }

CSV="$OUT/$TAG.csv"
echo "t_s,sm_mhz,power_w,gpu_c,util,reasons,acpi_max_c" > "$CSV"
sample() {   # 单独一个进程采样, 不进引擎进程, 不影响被测路径
    local t0=$SECONDS line acpi
    while :; do
        line=$(nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active --format=csv,noheader,nounits | tr -d ' ')
        acpi=$(cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1)
        echo "$((SECONDS-t0)),$line,$((acpi/1000))" >> "$CSV"
        sleep 5
    done
}
echo "[$TAG] $(date '+%H:%M:%S') 起跑: 空载 $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader); $(nvidia-smi -q -d CLOCK | grep -m1 -A1 'Max Clocks' | tail -1 | tr -s ' ')"
sample & SPID=$!
trap 'kill "$SPID" 2>/dev/null' EXIT
./ds4 -m "$MODEL" --zchain "$AMP" --temp 0 --seed 1 -n "$NGEN" --gen-ids "$IDS" > "$OUT/$TAG.out" 2> "$OUT/$TAG.err" &
PID=$!; bad=0; n=0
while kill -0 "$PID" 2>/dev/null; do
    a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$a" -lt "$WD_KILL_MB" ]; then bad=$((bad+1)); else bad=0; fi
    if [ "$bad" -ge 2 ]; then
        echo "★看门狗: MemAvailable ${a} MB < $WD_KILL_MB 连续两次, 杀 ds4★"
        kill "$PID" 2>/dev/null; sleep 3; kill -9 "$PID" 2>/dev/null; break
    fi
    sleep 5; n=$((n+1))
    [ $((n%12)) -eq 0 ] && echo "[$TAG] $(date '+%H:%M:%S') $(tail -1 "$CSV")"
done
wait "$PID"; rc=$?
kill "$SPID" 2>/dev/null; trap - EXIT
echo "[$TAG] 退出码 $rc"
grep -a -hE 'prefill .* token|decode .* token|DSpark:' "$OUT/$TAG.err" | tail -3 | sed "s/^/[$TAG] /"
# 汇总只算真在干活的采样(util>=50), 取其后半程
awk -F, -v tag="$TAG" 'NR>1 && $5>=50 {n++; s[n]=$2; p[n]=$3; g[n]=$4; a[n]=$7; r[$6]++; if($4>gm)gm=$4; if($3>pm)pm=$3}
END {if(!n){print "["tag"] ★没有满载采样★"; exit}
     h=int(n/2)+1; for(i=h;i<=n;i++){ss+=s[i];sp+=p[i];sg+=g[i];sa+=a[i]} c=n-h+1
     printf "[%s] 满载 %d 采样; 后半程均值 %.0f MHz / %.1f W / GPU %.1f C / ACPI %.1f C; 全程峰值 %d C / %.1f W\n",tag,n,ss/c,sp/c,sg/c,sa/c,gm,pm
     for(k in r) printf "[%s] 限频原因 %s: %d 采样\n",tag,k,r[k]}' "$CSV"
echo "[$TAG] 输出 $(wc -c < "$OUT/$TAG.out") 字节 md5 $(md5sum < "$OUT/$TAG.out" | cut -c1-12); 采样 $CSV"
exit "$rc"
