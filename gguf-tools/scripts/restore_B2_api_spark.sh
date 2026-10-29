#!/bin/bash
# restore_B2_api_spark.sh — 还原侧车 段 B 第三趟: 官方 API 轨迹当硬靶训 ③, 不用 FP 教师(2026-10-25 晚, 用户令"官方 api 采集, 速进反修 b 段")。
#   为什么不走 posttrain_fp_spark.sh: 那条链的 ② 阶段是 FP 教师打 top-K 表; FP 教师长程失灵(fable5 10-25), 这趟 cfg 不带 teacher=,
#   训练器对 text 行就是 one-hot 交叉熵(core_ptrain_data.c), 靶 = 官方 API 一字不改的轨迹。
# 四步(任一步非 0 停, 尾行 B2API_EXIT <rc>):
#   ① 料: think_trace_collect.py chunks(82 条图形题 raw → ≤1024 窗, 每条 ≤16 窗) + formal2_1022.clean 拼成 formal2_apichunks1024.jsonl; 前 8 窗另存探针料
#   ② 内存探针: restore_B2api_probe.cfg(8 窗 1 轮)在 with_model_stopped 里跑 ./ds4 --ptrain —— 先证明这个窗长不炸(4096 当场 OOM、2048 撞看门狗阈, 见 cfg)
#   ③ 正式: restore_B2api_4096.cfg → gguf/v41/posttrain/restore-B2api-4096/ckpt_eNN + heldout_eNN.tsv
#   ④ restore_sidecar_chain_spark.sh --from C --cfg restore_B2api_4096.cfg: 选轮(留出损失最低)→ amp 并进段 A 侧车 → 段 C 报数
#   鹈鹕带图 3 趟(tests/asst/dsh_compare.sh)另起, 这里不包。
# 用法(spark 仓库根): nohup gguf-tools/scripts/restore_B2_api_spark.sh [--from 1|2|3|4] > /tmp/bugmd/b2api_<时间>.log 2>&1 &
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
FROM=1; TO=4; OCFG=""; OPCFG=""
# --cfg / --pcfg: 换正式/探针配方(10-25 晚: ctxlen 前缀形态用 restore_B2ctx_*.cfg, 料已在盘上就 --from 2 跳过 ①)
TRAW=""   # --teacher-raw raw.jsonl(10-26): ③ 改走 posttrain_fp_spark.sh, 靶 = 官方 API top-20 软靶(api_teacher.py collect 的原料), 不再 one-hot; 保持料由本地 FP 教师补
# --tmo-train / --tmo-judge <分钟>(10-27 自进化): ③ 训练(含教师打表)与 ④ 段 C 的硬超时, 都套在 with_model_stopped 里面; --heldout <目录|none> / --pelseg 0 透传给段 C
TMOT=""; TMOJ=""; HOARG=()
while [ $# -gt 0 ]; do case "$1" in --from) FROM="${2:?}"; shift 2;; --to) TO="${2:?}"; shift 2;; --cfg) OCFG="${2:?}"; shift 2;; --pcfg) OPCFG="${2:?}"; shift 2;; --teacher-raw) TRAW="${2:?}"; shift 2;;
    --tmo-train) TMOT="${2:?}"; shift 2;; --tmo-judge) TMOJ="${2:?}"; shift 2;; --heldout) HOARG+=(--heldout "${2:?}"); shift 2;; --pelseg) HOARG+=(--pelseg "${2:?}"); shift 2;; *) echo "★不认识的参数 $1★"; exit 2;; esac; done
TT=(); [ -n "$TMOT" ] && TT=(timeout --kill-after=60 "${TMOT}m")
# --to N: 只跑到第 N 步就停(先单跑 ② 探针看一窗多少秒, 再决定正式趟料的窗数)
stage(){ [ "$FROM" -le "$1" ] && [ "$TO" -ge "$1" ]; }
PY="$HOME/v41env/bin/python"
W=gguf-tools/scripts/with_model_stopped_spark.sh
HF=hf/DeepSeek-V4.1-Flash-1017
PD=gguf-tools/data/posttrain
CFG=${OCFG:-$PD/restore_B2api_4096.cfg}; PCFG=${OPCFG:-$PD/restore_B2api_probe.cfg}
RAW=$PD/graphics_raw.jsonl
CH=$PD/restore_clean/api_chunks1024.jsonl    # 1024 不是 4096: 训练器反传缓冲按 maxlen 分, 4096 分不出、2048 预热后撞看门狗阈(16:34/16:38 实撞), 见 cfg 注释
MAXTOK=1024; MAXCH=16
DATA=$(grep '^data=' "$CFG" | cut -d= -f2-); PDATA=$(grep '^data=' "$PCFG" | cut -d= -f2-)
OUT=$(grep '^out=' "$CFG" | cut -d= -f2-); POUT=$(grep '^out=' "$PCFG" | cut -d= -f2-)
mapfile -t PK < gguf/serve_pick.txt; GG="${PK[0]#"$ROOT"/}"
ARM="gguf/v41/$(basename "$GG" .gguf)-grrb-restore_fit_n15360-engine"
M(){ echo "[b2api $(date '+%m-%d %H:%M:%S')] $*"; }
die(){ M "★$*★"; echo "B2API_EXIT 1"; exit 1; }
for f in "$GG" "$HF/config.json" "$CFG" "$PCFG" "$RAW" "$PD/restore_clean/formal2_1022.clean.jsonl"; do [ -e "$f" ] || die "缺 $f"; done
grep -q "^# 完成" "$ARM/manifest.txt" 2>/dev/null || die "段 A 侧车没有完成标记: $ARM"
grep -qE '^teacher=' "$CFG" && die "cfg 带了 teacher=, 这趟就是不要 FP 教师"
# 训练子进程的旗照 posttrain_fp_spark.sh / 工作台: 预算 110000 = 机器空着整机给它; --no-dspark 不建草稿器; 挂 ② = 段 A 侧车(目录里此时不该有 amp)
# ★训练前先把侧车里上一趟段 B 的 amp 挪走★(docs/restore_sidecar.md: 再训一次段 B = 先挪走再训; 训练器见 ② 带低秩项直接拒
#   "本版不接按秩拼接训练", 10-25 16:27 实撞)。只挪不删, 目录 amp_prev_<时刻>; 段 C 并包时拷新的进来。部署态在这期间少了段 B 项, 服务重起就是裸 ②。
if [ "$FROM" -le 3 ] && ls "$ARM"/amp_L*.bin >/dev/null 2>&1; then
    old="$ARM/amp_prev_$(date +%m%d%H%M)"; mkdir -p "$old"; mv "$ARM"/amp_L*.bin "$old"/ || die "挪旧 amp 失败"
    M "侧车里上一趟段 B 的 $(ls "$old" | wc -l) 个 amp 挪到 $old(不删)"
fi
DS4=(./ds4 --cuda -m "$GG" --zchain "$ARM" --mem-budget-mb 110000 --no-dspark)

if stage 1; then
    M "① 料: chunks ≤$MAXTOK × ≤$MAXCH 窗/条 ← $RAW"
    "$PY" gguf-tools/scripts/think_trace_collect.py chunks --hf "$HF" --raw "$RAW" --out "$CH" --max-tok "$MAXTOK" --max-chunks "$MAXCH" || die "chunks 失败"
    cat "$PD/restore_clean/formal2_1022.clean.jsonl" "$CH" > "$DATA"
    head -8 "$CH" > "$PDATA"
    M "料: formal2.clean $(wc -l < "$PD/restore_clean/formal2_1022.clean.jsonl") + api_chunks $(wc -l < "$CH") = $(wc -l < "$DATA") 行 → $DATA; 探针 8 窗 → $PDATA"
fi
if stage 2; then
    M "② 内存探针: $PCFG → $POUT"
    mkdir -p "$POUT"
    "$W" bash -c "${DS4[*]} --ptrain $PCFG > $POUT/train.out 2>&1"; rc=$?
    grep -E "ptrain\]|★|错|失败|epoch|held" "$POUT/train.out" | tail -8
    [ "$rc" = 0 ] && [ -s "$POUT/ckpt_e01/amp_L00.bin" ] || die "探针 rc=$rc(看 $POUT/train.out; dmesg 查 OOM)"
    M "② 过: 4096 窗训练器不炸"
fi
if stage 3; then
    M "③ 正式: $CFG → $OUT${TRAW:+ (API 教师软靶 $TRAW)}"
    mkdir -p "$OUT"
    if [ -n "$TRAW" ]; then
        [ -s "$TRAW" ] || die "API 教师原料 $TRAW 不在"
        # posttrain_fp_spark.sh 自己分三阶段(导序列 → API 表 + 本地 FP 教师补缺 → 训), 整个在停模型下跑; 它的 train.out 也落在 $OUT
        "$W" "${TT[@]}" gguf-tools/scripts/posttrain_fp_spark.sh "$CFG" "$HF" "$ARM" "" "" "$TRAW" > "$OUT/fp_chain.log" 2>&1; rc=$?
        grep -E "^\[ptfp|★" "$OUT/fp_chain.log" | tail -14
    else
        "$W" "${TT[@]}" bash -c "${DS4[*]} --ptrain $CFG > $OUT/train.out 2>&1"; rc=$?
    fi
    [ "$rc" = 124 ] && die "③ 训练超时(${TMOT} 分钟; 看 $OUT/train.out 停在哪一步)"
    grep -E "ptrain\]|★|错|失败|epoch|held" "$OUT/train.out" | tail -12
    [ "$rc" = 0 ] && ls "$OUT"/ckpt_e*/amp_L00.bin >/dev/null 2>&1 || die "正式趟 rc=$rc(看 $OUT/train.out)"
    for h in "$OUT"/heldout_e*.tsv; do [ -s "$h" ] && echo "  $(basename "$h") 留出损失 $(awk -F'\t' '{s+=$4; n++} END{if(n) printf "%.4f", s/n}' "$h")"; done
fi
if stage 4; then
    M "④ 选轮 + 并进侧车 + 段 C 报数(restore_sidecar_chain_spark.sh --from C)"
    bash gguf-tools/scripts/restore_sidecar_chain_spark.sh --from C --cfg "$CFG" "${HOARG[@]}" ${TMOJ:+--tmo-judge "$TMOJ"} || die "段 C 失败"
fi
M "完成"; echo "B2API_EXIT 0"
