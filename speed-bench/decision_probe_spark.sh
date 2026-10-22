#!/bin/bash
# decision_probe_spark.sh — 决策位尺(2026-10-22, docs/think_close.md §5 第 0 步): 同一条序列上 FP 原件 vs 引擎学生, 读"停"类 token
# (EOS / </think>)与 '?'(id 33) 的概率 —— 量化在这个底座上把决策位压了几倍、把自问抬了多少。这是后面 ③ "修了多少"的基线。
#
# 用法(spark 仓库根; 停模型/装回交给包装器, 本脚本只干活):
#   nohup gguf-tools/scripts/with_model_stopped_spark.sh speed-bench/decision_probe_spark.sh <底座 gguf> <HF 出厂权重目录> [<放大器目录>|none] [<③ 目录>|none] \
#        ["名|ids|ntok|提示位数|停token|keep/drop" ...] > /tmp/bugmd/decision_<时间>.log 2>&1 &
#   第 4 参数 = ③ 后训练目录(ckpt_eNN, 10-22): 判 "挂 ③ 后决策位修回来多少"; 第 5 起 = 追加的序列行(如另一套留出篇子 docend2), 格式同下表。
# 序列固定三条(全在盘上, 不生成):
#   docend  gguf/go-onebit/docend/docend.ids           12477 位, 提示 0,     停 token = EOS 1(25 篇整篇文档的真结尾, 09-30 料)
#   pelican gguf/v41/night/faith_1016/pelican_1020.ids 43827 位, 提示 11358, 停 token = </think> 128822(1017 自己的轨迹; 行为型: 10-20 FP 也不停)
#   r316    gguf/v41/night/faith_1016/r316.ids         48809 位, 提示 37825, 停 token = </think> 128822(还原型: 10-17 v3 底座 FP 0.968 对量化 0.345)
# 做什么: 每条 v41_judge.sh <ids> <ntok> --hf <HF> engine:<gguf>[:<放大器目录>] —— 教师锚已有就复用, 没有就 FP 参考前向出一份(全 logits 517 KB/位)
#   → bugmd_ids_tools.py tokprobe(停 token / 33)在教师锚与学生锚上各一遍; 思考序列再 anchor-faith(每 4096 生成位 Σp(</think>), 10-17 口径)
#   → 读数全落 /tmp/bugmd/decision_<时间>.txt。学生锚(160 s 可重算, 思考序列一份 22~25 GB)读完即删腾盘, docend 的 6.5 GB 留着; 教师锚全留。
# 判读: 几何均值比 FP/学生 —— 09-30 v3 底座: EOS 8 倍, </think> 4~22 倍。这个底座若 ≈ 1 ⇒ 还原型不存在, think_close 方案停。
# 出错会怎样: 某条 judge rc≠0 照样跑下一条, 汇总里写 ★; 盘余量 < 60 GB 不跑 r316(它一趟教师 + 学生 50 GB); 任何 ★ 行都要人看。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
M(){ echo "[dec $(date '+%m-%d %H:%M:%S')] $*"; }
GG="${1:?底座 gguf}"; HF="${2:?HF 出厂权重目录}"; AMP="${3:-none}"; PTD="${4:-none}"; shift $(( $# < 4 ? $# : 4 ))
[ -s "$GG" ] && [ -s "$HF/config.json" ] || { M "★底座 $GG 或 HF 目录 $HF 不对★"; exit 2; }
[ "$AMP" = none ] || [ -s "$AMP/manifest.txt" ] || { M "★放大器目录 $AMP 没有 manifest.txt★"; exit 2; }
[ "$PTD" = none ] || ls "$PTD"/amp_L*.bin >/dev/null 2>&1 || { M "★③ 目录 $PTD 里没有 amp_Lnn.bin★"; exit 2; }
# v41_judge.sh 引擎档: engine:<gguf>:<②>:<β>:<③>:<二进制>, 空字段留空
ARM="engine:$GG:$([ "$AMP" = none ] || echo "$AMP")::$([ "$PTD" = none ] || echo "$PTD")"
J="$ROOT/gguf/v41judge"; TOOLS="$ROOT/speed-bench/bugmd_ids_tools.py"
HFSFX="_$(basename "$HF")"   # v41_judge.sh: 非默认 HF 时教师锚名带 _<HF 目录名>
TS=$(date +%m%d_%H%M); mkdir -p /tmp/bugmd; OUT="/tmp/bugmd/decision_$TS.txt"
SEQS=(
  "docend|gguf/go-onebit/docend/docend.ids|12477|0|1|keep"
  "pelican|gguf/v41/night/faith_1016/pelican_1020.ids|43827|11358|128822|drop"
  "r316|gguf/v41/night/faith_1016/r316.ids|48809|37825|128822|drop"
)
for extra in "$@"; do SEQS+=("$extra"); done   # 追加的序列行(第 5 参数起)
tag_of(){ echo "$(basename "$(dirname "$1")")_$(basename "$1" .ids)_n$2"; }   # 与 v41_judge.sh 的 corpus_tag + _n<ntok> 同式
probe(){ python3 "$TOOLS" tokprobe "$1" "$2" "$3" "$4" 2>&1 | sed -n '1p;$p'; }   # 首行 = 命中位置数 + 其余位置 p 均值; 末行 = 几何均值 / 中位 / argmax 数
avail_gb(){ df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
{ echo "决策位尺 $TS  底座 $(basename "$GG")  HF $(basename "$HF")  放大器 $AMP  ③ $PTD  可用盘 $(avail_gb) GB"; } | tee "$OUT"
for row in "${SEQS[@]}"; do
    IFS='|' read -r name ids ntok npr tok keep <<<"$row"
    [ -s "$ids" ] || { M "★$name ids $ids 不在★" | tee -a "$OUT"; continue; }
    if [ "$name" = r316 ] && [ "$(avail_gb)" -lt 60 ]; then M "★盘只剩 $(avail_gb) GB, r316 一趟要 50 GB, 跳过★" | tee -a "$OUT"; continue; fi
    TAG=$(tag_of "$ids" "$ntok"); REF="$J/teacher_${TAG}${HFSFX}.bin"; JL="/tmp/bugmd/decision_${TS}_${name}_judge.log"
    M "== $name: v41_judge.sh $ids $ntok --hf $HF $ARM($([ -s "$REF" ] && echo 教师锚已有 || echo 教师锚要算))" | tee -a "$OUT"
    bash gguf-tools/scripts/v41_judge.sh "$ids" "$ntok" --hf "$HF" "$ARM" > "$JL" 2>&1 || M "★$name judge rc≠0, 看 $JL★" | tee -a "$OUT"
    grep -E "五指标|PPL|Σmin|KLD|Same|教师|★" "$JL" | tail -8 | tee -a "$OUT"
    STU=$(ls -t "$J"/stu_"${TAG}"_eng_*.bin 2>/dev/null | head -1)
    [ -s "$REF" ] || { M "★$name 教师锚 $REF 不在★" | tee -a "$OUT"; continue; }
    [ -n "$STU" ] && [ -s "$STU" ] || { M "★$name 学生锚不在★" | tee -a "$OUT"; continue; }
    for t in "$tok" 33; do
        { echo "-- $name 教师 tok=$t"; probe "$REF" "$ids" "$t" "$npr"
          echo "-- $name 学生 tok=$t  $(basename "$STU")"; probe "$STU" "$ids" "$t" "$npr"; } | tee -a "$OUT"
    done
    if [ "$tok" = 128822 ]; then   # 思考序列: </think> 真吐可能一次没出(r316 就是), tokprobe 无命中位, 要 anchor-faith 的全段 Σp
        for side in "$REF" "$STU"; do
            { echo "-- $name anchor-faith $(basename "$side")"; python3 "$TOOLS" anchor-faith "$side" "$ids" "$npr" 0 128822 2>&1 | tail -16; } | tee -a "$OUT"
        done
    fi
    if [ "$keep" = drop ]; then M "删学生锚 $(basename "$STU")($(du -h "$STU" | cut -f1); 160 s 可重算) 腾盘" | tee -a "$OUT"; rm -f "$STU"; fi
done
M "完成 → $OUT  可用盘 $(avail_gb) GB" | tee -a "$OUT"
