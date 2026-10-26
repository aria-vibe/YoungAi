#!/bin/bash
# v41_fincal_chain_spark.sh — 校准量化全链(2026-09-20 深夜金融域首发; 2026-10-24 参数化 + 级 2 探针段, 方案 docs/reindex_ef.md)。
#
# 只串既有入口, 数值全在各自的程序里:
#   S0 取料   v41_amp_run --dump-calib: 裸 ① 上跑【量化份】ids(默认八域 vqhalf_q 8192 行), 40 层同趟落 calib_Lnn.bin(3.4 GB)
#   S1 冒烟   v41_quantize --layers 0:1 --calib --calib-ab: 一层 384 专家各做两遍(校准指派 / 平权指派), 同一把加权尺上
#             校准必须赢平权 —— 机制审计(列权有没有真进指派与训练); 不赢就停, 不许带着假机制跑 66 分钟
#   B1 全量   v41_quantize_spark.sh --calib … --judge-set all: 40 层 + common, 文件态三尺
#   G  转 GGUF  v41_to_gguf(配方 KV 会带 cal=…)
#   C  引擎对拍 v41_engine_parity_spark.sh --gguf <新> wt2 512: 引擎 vs 文件态 Python 学生, KL ≤ 0.07 才算接线对
#   ②  v41_gronly_gate.sh <新>: 解 gr-only + 五指标 + wt2 守门 + 打转尺
#   P  级 2 探针(10-24, 独立于上面的段序): --probe a:b 只重做 [a,b) 层, 其余层与 common 软链 --base-dir(现役量化目录);
#      P0 取料没有就取 → P1 审计(--calib-alpha 0 无 EF 重量化, 逐分片 cmp 现役: 逐字节同 ⇒ 级 2 是唯一变量)
#      → P2 级 2(--calib-ef β 表, 层账 "val 输出误差 级2÷级1 均值" 任一层 ≥ 1.0 停车) → P3 判决(judge.ids 15360 + wt2 512, 两臂同趟 file: 路)
#      → P4 逐位配对(judge_rows_pair.sh)。基线臂 = P1 逐字节同就用现役目录, 否则用 P1 产物(级 1 带 calib 的路有差时, 差不能算进级 2)。
# 停车规则: 任一段非 0 就停在那一段(半成品往下传只出假数); 每段开头打 [chain] 时间戳, 尾行 CHAIN_EXIT <rc>。
# 看门狗: S0/P0 段本脚本自带(available < 8 GB 杀 v41_amp_run); 其余段各脚本自带。
# 用法(spark 本机, 包在 with_model_stopped_spark.sh 里 —— 取料与判决都要 113 GB 的机器空出来):
#   gguf-tools/scripts/with_model_stopped_spark.sh bash gguf-tools/scripts/v41_fincal_chain_spark.sh --probe 10:20 --ef 0.1,1
#   旧链: v41_fincal_chain_spark.sh [--from S0|S1|B1|G|C|GATE] [--ef β1,β2] [--alpha α]
#   参数(10-24, 默认全是现役 1017): --hf <出厂目录> --base <裸 ① GGUF, 取料用> --base-dir <现役量化目录, 探针软链用>
#     --ids <量化份 ids> --cal <取料目录> --tag <产物名> --recipe "<v41_quantize 的位宽/码本参数串>"
#   --ef: 级 2 误差反馈(v41_quantize --calib-ef), 旧链产物名带 fincal2; 取料(S0)与级 1 共用同一份
# 出错会怎样: 缺文件直接 die 不发; P1 cmp 不同不停车只换基线臂并打 ★; P2 层账抓不到 = 量化器没走级 2 路, die。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
FROM="S0"; EFB=""; ALPHA=""; PROBE=""
HF="hf/DeepSeek-V4.1-Flash-1017"
BASE="gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf"        # 取料用的裸 ①(现役)
BASEDIR="gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k"                    # 现役量化目录(探针的软链基底 + 基线臂)
QIDS="gguf/go-onebit/vqhalf/vqhalf_q.ids"                                  # 八域【量化份】8192(a 拟合 / j 判决不动)
CAL="gguf/v41/calib-q8192-1017"
TAG="calq"
# 现役 1017 v3 配方(v41_base_quant_chain_spark.sh B1 同一串, 少了只管 common 的 --mtp-vq): 探针只重做层, 这串必须与现役逐字同, 否则 P1 的 cmp 必不同
RECIPE="--skel q4k --vq-nc 4096 --vq-nc-layers 0:14=8192 --vq-shared-cb --vq-cb-fp8"
while [ $# -gt 0 ]; do case "$1" in
  --from) FROM="${2:?--from 要段名}"; shift 2;;
  --ef) EFB="${2:?--ef 要 β 表}"; shift 2;;
  --alpha) ALPHA="${2:?--alpha 要 [0,1] 的数}"; shift 2;;   # 列权指数(09-21 01:41 实撞: 全额列权 wt2 −2.15 pp), 产物名带 a<α>
  --probe) PROBE="${2:?--probe 要 a:b}"; shift 2;;
  --hf) HF="${2:?}"; shift 2;;
  --base) BASE="${2:?}"; shift 2;;
  --base-dir) BASEDIR="${2:?}"; shift 2;;
  --ids) QIDS="${2:?}"; shift 2;;
  --cal) CAL="${2:?}"; shift 2;;
  --tag) TAG="${2:?}"; shift 2;;
  --recipe) RECIPE="${2:?}"; shift 2;;
  *) echo "★不认识的参数 $1★"; exit 2;;
esac; done
QEXTRA=()
[ -n "$EFB" ] && { [ -n "$PROBE" ] || TAG="${TAG}2"; QEXTRA=(--calib-ef "$EFB"); }
[ -n "$ALPHA" ] && { TAG="$TAG-a${ALPHA//./}"; QEXTRA+=(--calib-alpha "$ALPHA"); }
SMOKE="gguf/v41/smoke-$TAG-L0"
OUTDIR="gguf/v41/$(basename "$BASEDIR")-$TAG"
GG="gguf/v41/$(basename "$BASEDIR")-$TAG-mtpnative.gguf"
WT2="gguf/go-onebit/g7/wt2.ids"
JUDGE_IDS="gguf/go-onebit/code/judge.ids"
M(){ echo "[chain $(date '+%m-%d %H:%M:%S')] $*"; }
die(){ M "★$*★"; echo "CHAIN_EXIT 1"; exit 1; }
# 段序比较: 段 $1 在 --from 指定的段之后(含)才跑
stage_ge(){ local order="S0 S1 B1 G C GATE"; local a=${order%%$1*}; local b=${order%%$FROM*}; [ ${#a} -ge ${#b} ]; }
for f in "$BASE" "$QIDS" "$QIDS.layout" "$WT2" "$HF/config.json"; do [ -e "$f" ] || die "缺 $f"; done

# 取料(S0 / P0 共用): 裸 ① × 量化份 → $CAL, 40 层同趟
dump_calib() {
  M "取料: 裸 $(basename "$BASE") × $QIDS 8192 → $CAL"
  make -C gguf-tools v41_amp_run >/tmp/fincal_make_amp.log 2>&1 || die "v41_amp_run 编译失败(见 /tmp/fincal_make_amp.log)"
  mkdir -p "$CAL"
  ( while sleep 20; do av=$(free -g | awk '/^内存|^Mem/{print $7}'); if [ "${av:-99}" -lt 8 ]; then echo "★看门狗: available ${av}G < 8G, 停车★"; pkill -f "amp/v41_amp_ru[n]"; fi; done ) & WD=$!
  gguf-tools/amp/v41_amp_run "$BASE" "$HF" "$QIDS" 8192 "$CAL" --dump-calib --mem-budget-mb 40000 --weight-cache-mb 88000 2>&1 \
      | tee /tmp/fincal_s0.log | grep --line-buffered -E "校准|引擎|★|error|Error|失败"
  rc=${PIPESTATUS[0]}; kill $WD 2>/dev/null
  [ "$rc" = 0 ] && [ -s "$CAL/calib.txt" ] && [ -s "$CAL/calib_L39.bin" ] || die "取料失败 rc=$rc"
}

# ---------------- P 级 2 探针(独立段序) ----------------
if [ -n "$PROBE" ]; then
  IFS=: read -r PA PB <<<"$PROBE"
  [ -n "$PA" ] && [ -n "$PB" ] && [ "$PA" -lt "$PB" ] 2>/dev/null || die "--probe 要 a:b"
  [ -n "$EFB" ] || die "--probe 要配 --ef β 表(级 2 是探针的变量)"
  [ -d "$BASEDIR" ] && [ -e "$BASEDIR/model-common.safetensors" ] || die "--base-dir $BASEDIR 不是现役量化目录"
  [ -e "$JUDGE_IDS" ] || die "缺 $JUDGE_IDS"
  LV1="gguf/v41/$TAG-probe-$PA-$PB-lv1"; LV2="gguf/v41/$TAG-probe-$PA-$PB-lv2"
  Q="gguf-tools/scripts/v41_quantize_spark.sh"; J="gguf-tools/scripts/v41_judge.sh"; PAIR="gguf-tools/scripts/judge_rows_pair.sh"
  # shellcheck disable=SC2206
  RARGS=($RECIPE)
  if [ -s "$CAL/calib.txt" ] && [ -s "$CAL/calib_L39.bin" ]; then M "P0 取料已有: $CAL($(head -c 200 "$CAL/calib.txt"))"; else dump_calib; fi

  M "P1 审计: L$PA–$((PB-1)) 带 --calib --calib-alpha 0(无 EF)重量化 → $LV1, 逐分片 cmp 现役"
  bash "$Q" --hf "$HF" --out "$LV1" --base "$BASEDIR" --layers "$PA:$PB" --no-common --no-judge --force "${RARGS[@]}" --calib "$CAL" --calib-alpha 0 \
      | grep --line-buffered -E "^\[分片\]|★|失败|基底|量化 →" || die "P1 量化失败"
  same=0; diff_=0
  for ((L=PA; L<PB; L++)); do
    bn=$(printf "model-layer%02d.safetensors" "$L")
    [ -f "$LV1/$bn" ] && [ ! -L "$LV1/$bn" ] || die "P1 $bn 不是真产物(量化器跳过了这层? --force 没传到?)"
    if cmp -s "$LV1/$bn" "$BASEDIR/$bn"; then same=$((same+1)); else diff_=$((diff_+1)); M "  ★$bn 与现役不同: $(cmp "$LV1/$bn" "$BASEDIR/$bn" 2>&1 | head -1)★"; fi
  done
  if [ "$diff_" = 0 ]; then BASEARM="$BASEDIR"; M "P1 ✓ $same/$((PB-PA)) 层逐字节同现役 ⇒ 级 2 是唯一变量, 基线臂 = 现役目录"
  else BASEARM="$LV1"; M "★P1 $diff_/$((PB-PA)) 层与现役不同: 级 1 带 calib 的路 ≠ 零语料路, 基线臂改用 $LV1(级 2 的差不许掺这份)★"; fi

  M "P2 级 2: 同配方 + --calib-ef $EFB → $LV2"
  bash "$Q" --hf "$HF" --out "$LV2" --base "$BASEDIR" --layers "$PA:$PB" --no-common --no-judge --force "${RARGS[@]}" --calib "$CAL" --calib-alpha 0 --calib-ef "$EFB" \
      | grep --line-buffered -E "^\[分片\]|级 2|EF|★|失败|基底|量化 →" || die "P2 量化失败"
  n_acc=$(grep -c "级 2 做" "$LV2/quantize_log.txt"); [ "$n_acc" -ge $((PB-PA)) ] || die "P2 层账只抓到 $n_acc 行(要 $((PB-PA))): 量化器没走级 2 路"
  M "P2 层账(做/跳过/β 选中/val 输出误差 级2÷级1):"; grep "级 2 做" "$LV2/quantize_log.txt" | tail -n $((PB-PA))
  bad=$(grep "级 2 做" "$LV2/quantize_log.txt" | tail -n $((PB-PA)) | sed -E 's/.*均值 ([0-9.]+).*/\1/' | awk '$1>=1.0{c++} END{print c+0}')
  [ "$bad" = 0 ] || die "P2 有 $bad 层 val 输出误差 级2÷级1 ≥ 1.0: 级 2 没降层内误差, 机制有 bug, 不进判决"

  M "P3 判决: $JUDGE_IDS 15360 + wt2 512, 两臂 file:$BASEARM / file:$LV2(教师锚 1017 缓存)"
  bash "$J" "$JUDGE_IDS" 15360 --hf "$HF" "file:$BASEARM" "file:$LV2" || M "★判决份判决有失败项★"
  bash "$J" "$WT2" 512 --hf "$HF" "file:$BASEARM" "file:$LV2" || M "★wt2 判决有失败项★"
  for S in "code_judge_n15360" "g7_wt2_n512"; do
    for A in "$BASEARM" "$LV2"; do f=$(ls -t gguf/v41judge/log_stu_${S}_file_"$(basename "$A")"*.txt 2>/dev/null | head -1); [ -n "$f" ] && { echo "== $S 臂 $(basename "$A")"; grep -E "PPL|Σmin|KLD|Same top" "$f" | tail -4; }; done
  done
  M "P4 逐位配对(判决份): 普遍变好 vs 随机搅动"
  RA=$(ls -t gguf/v41judge/stu_code_judge_n15360_file_"$(basename "$BASEARM")"*.rows 2>/dev/null | head -1)
  RB=$(ls -t gguf/v41judge/stu_code_judge_n15360_file_"$(basename "$LV2")"*.rows 2>/dev/null | head -1)
  [ -s "$RA" ] && [ -s "$RB" ] && bash "$PAIR" "$RA" "$RB" || M "★配对缺 .rows($RA / $RB)★"
  M "探针收工: 判读 f = 级 2 KLD 降幅 ÷ 31.9%(L10–19 整段换回原件, fable5 10-21); f ≥ 1/3 进全量, < 1/6 判负(docs/reindex_ef.md §3)"
  echo "CHAIN_EXIT 0"; exit 0
fi

# ---------------- 旧链 S0 → S1 → B1 → G → C → GATE ----------------
if stage_ge S0; then dump_calib; fi

if stage_ge S1; then
  M "S1 冒烟: L0 校准指派 vs 平权指派(同一把加权尺)"
  make -C gguf-tools v41_quantize >/tmp/fincal_make_q.log 2>&1 || die "v41_quantize 编译失败(见 /tmp/fincal_make_q.log)"
  rm -rf "$SMOKE"; mkdir -p "$SMOKE"
  # shellcheck disable=SC2086
  gguf-tools/quantize/v41_quantize "$HF" "$SMOKE" --layers 0:1 --no-common $RECIPE --calib "$CAL" --calib-ab "${QEXTRA[@]}" --force 2>&1 \
      | tee /tmp/fincal_smoke.log | grep --line-buffered -E "校准|分片|索引|EF|★|失败"
  [ "${PIPESTATUS[0]}" = 0 ] || die "S1 量化器失败"
  grep -q "加权尺上校准赢平权" /tmp/fincal_smoke.log || die "S1 机制审计没过: 加权尺上校准没赢平权(列权没进指派/训练)"
  # 平权 cos 的先验: 09-20 nocal 那趟 L00 是 0.9105; 等于 0.9105 = 列权没生效。校准指派在平权尺上【会明显掉】——
  # 09-21 00:10 实测 0.8861(加权 cos 0.9126 → 0.9322): 把误差挪到激活小的列就是这样, 不是坏; 低于 0.85 才当坏了
  pc=$(grep "^\[分片\] model-layer00" /tmp/fincal_smoke.log | sed -E 's/.*参数 cos ([0-9.]+).*/\1/')
  # 级 2(误差反馈按 H 而不是按平权 MSE 选索引)平权 cos 还会再掉, 下限放到 0.80 —— 真坏的形状是 cos 掉到 0.7 以下(09-11 x 错位那种)
  awk -v c="$pc" 'BEGIN{exit !(c>=0.80 && c<0.9105)}' || die "S1 平权 cos $pc 不在 [0.80, 0.9105)"
  rm -rf "$SMOKE"
fi

if stage_ge B1; then
  M "B1 全量: 40 层 + common, --calib $CAL, 文件态三尺"
  # shellcheck disable=SC2086
  bash gguf-tools/scripts/v41_quantize_spark.sh --hf "$HF" --out "$OUTDIR" $RECIPE --mtp-vq --calib "$CAL" "${QEXTRA[@]}" --judge-set all || die "B1 失败"
fi

if stage_ge G; then
  M "G 转 GGUF → $GG"
  make -C gguf-tools v41_to_gguf >/tmp/fincal_make_g.log 2>&1 || die "v41_to_gguf 编译失败(见 /tmp/fincal_make_g.log)"
  gguf-tools/quantize/v41_to_gguf "$OUTDIR" gguf/v41/consts/engram_consts.bin gguf/v41/consts/tokenizer_consts.bin "$GG" "$HF" 2>&1 | tail -5
  [ "${PIPESTATUS[0]}" = 0 ] && [ -s "$GG" ] || die "G 转换失败"
  sz=$(stat -c %s "$GG"); M "GGUF $sz B = $(awk -v s="$sz" 'BEGIN{printf "%.3f", s/1e9}') GB(只换索引位 ⇒ 预期与现役逐字节同大: $(stat -c %s "$BASE") B)"
  ./ds4 -m "$GG" --cuda --inspect 2>&1 | grep -E "张量|GiB|校验|失败|★" | head -8
fi

if stage_ge C; then
  M "C 引擎对拍 wt2 512: 引擎 vs 文件态 Python 学生"
  STU=$(ls -t gguf/v41judge/stu_g7_wt2_n512_file_"$(basename "$OUTDIR")"*.bin 2>/dev/null | head -1)
  [ -s "$STU" ] || die "C 缺文件态 wt2 学生 logits(B1 的 --judge-set all 该落它)"
  # --engram: 文件态学生带 engram, 引擎也要带, 否则比的是两个不同的模型(09-20 C 段同口径)
  bash gguf-tools/scripts/v41_engine_parity_spark.sh "$WT2" 512 "$STU" --engram --gguf "$GG" 2>&1 | grep -E "KLD|Same|PPL|★" | head -12
fi

if stage_ge GATE; then
  M "② gr-only 三道门(解 + 五指标 + wt2 守门 + 打转尺)"
  bash gguf-tools/scripts/v41_gronly_gate.sh "$GG" || die "三道门失败"
fi
M "全链收工"; echo "CHAIN_EXIT 0"
