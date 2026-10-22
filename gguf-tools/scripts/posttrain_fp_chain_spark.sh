#!/bin/bash
# posttrain_fp_chain_spark.sh — ③ 向出厂原件蒸馏的整链一条命令(2026-10-22 晚固化; 当天两趟都是手敲 nohup bash -c 串, 违反"脚本落 repo"):
#   停模型 → posttrain_fp_spark.sh(导序列 / FP 教师分份 + 每份 ptfstat / 训 ③) → 每个 ckpt 过决策位尺 → 编程判决份 + wt2 两把门(每个 ckpt 一臂) → 装回模型。
#
# 用法(spark 仓库根, 工作台 ds4-train :8000 开着, 模型子进程 idle):
#   nohup gguf-tools/scripts/posttrain_fp_chain_spark.sh <ptrain.cfg> <HF 出厂权重目录> <② 侧车目录|none> <每份条数> <标签> [额外要过决策位尺的 ③ 目录, 逗号分隔] \
#        > /tmp/bugmd/chain_<标签>.log 2>&1 &
#   日志: /tmp/bugmd/pt_<标签>.log(教师 + 训练, 每份 [ptfstat] 在里面) / decision_<标签>_<ckpt>.log(+ 汇总 decision_<时间>.txt) / gate_<标签>_{code,wt2}.log
#   额外 ③ 目录: 比如上一趟没打完的 ckpt(10-22 e01 的 r316 因盘余量 < 60 GB 被跳过), 跟新 ckpt 一起过尺。
# 判读: 看本日志末尾各段摘出的行 —— ptfstat 的收口位/EOS 末位、★轮 N 末 的留出与决策位 KL、决策位尺的 "全段 Σp_FP(收口)" 与 "几何均值 p(tok)"、两把门的五指标。
# 出错会怎样: 训练段 rc≠0 整链停(没 ③ 就没尺可打)但模型照样装回; 尺/门某条 rc≠0 打 ★ 继续下一条; 模型子进程不是 idle 时包装器直接退出不停它(rc 1);
#   包装器开始时模型没在跑(上次崩了)就不会装回, 本脚本末尾兜一手 model_restart_spark.sh。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
M(){ echo "[chain $(date '+%m-%d %H:%M:%S')] $*"; }

if [ "${1:-}" = --inner ]; then   # 包装器里跑的整段(模型已停)
    shift
    CFG="$1"; HF="$2"; ZD="$3"; PER="$4"; TAG="$5"; EXTRA="${6:-}"
    OUTD=$(grep '^out=' "$CFG" | head -1 | cut -d= -f2- | tr -d ' ')
    GG=$(sed -n 1p gguf/serve_pick.txt); GG="${GG#"$ROOT"/}"
    M "① 教师 + 训练 → /tmp/bugmd/pt_$TAG.log"
    gguf-tools/scripts/posttrain_fp_spark.sh "$CFG" "$HF" "$ZD" "$PER" > "/tmp/bugmd/pt_$TAG.log" 2>&1; rc=$?
    grep -E "\[ptfstat\]|收口位|EOS 末位|★轮|决策位\(|第 0 步\(|② 完成|完成 →|★" "/tmp/bugmd/pt_$TAG.log" | tail -n 40
    [ "$rc" = 0 ] || { M "★训练段 rc=$rc, 链停(看 /tmp/bugmd/pt_$TAG.log)★"; exit "$rc"; }
    CKS=()
    for c in ${EXTRA//,/ }; do [ -d "$c" ] && CKS+=("$c") || M "★额外 ③ 目录 $c 不在, 跳过★"; done
    for c in $(ls -d "$OUTD"/ckpt_e[0-9][0-9] 2>/dev/null | sort -r); do CKS+=("$c"); done
    for c in "${CKS[@]}"; do
        t=$(echo "$c" | sed 's#gguf/v41/posttrain/##; s#/#_#g')
        M "② 决策位尺 $c → /tmp/bugmd/decision_${TAG}_$t.log"
        speed-bench/decision_probe_spark.sh "$GG" "$HF" "$ZD" "$c" > "/tmp/bugmd/decision_${TAG}_$t.log" 2>&1 || M "★决策位尺 $c rc≠0★"
        grep -E "^\[dec .*== |全段 Σp_FP|几何均值 p\(tok\)|★" "/tmp/bugmd/decision_${TAG}_$t.log" | cut -c1-160 | tail -n 16
    done
    ARMS=()
    for c in $(ls -d "$OUTD"/ckpt_e[0-9][0-9] 2>/dev/null | sort -r); do
        if [ "$ZD" = none ]; then ARMS+=("engine:$GG:::$c"); else ARMS+=("engine:$GG:$ZD::$c"); fi
    done
    M "③ 两把门: ${ARMS[*]}"
    gguf-tools/scripts/v41_judge.sh gguf/go-onebit/code/judge.ids 15360 --hf "$HF" "${ARMS[@]}" > "/tmp/bugmd/gate_${TAG}_code.log" 2>&1 || M "★编程判决份门 rc≠0★"
    grep -E "五指标 引擎|Σmin|KLD|Same top" "/tmp/bugmd/gate_${TAG}_code.log" | cut -c1-160
    gguf-tools/scripts/v41_judge.sh gguf/go-onebit/g7/wt2.ids 512 --hf "$HF" "${ARMS[@]}" > "/tmp/bugmd/gate_${TAG}_wt2.log" 2>&1 || M "★wt2 门 rc≠0★"
    grep -E "五指标 引擎|Σmin|KLD|Same top" "/tmp/bugmd/gate_${TAG}_wt2.log" | cut -c1-160
    exit 0
fi

CFG="${1:?ptrain.cfg}"; HF="${2:?HF 出厂权重目录}"; ZD="${3:?② 侧车目录|none}"; PER="${4:?每份条数}"; TAG="${5:?标签}"; EXTRA="${6:-}"
[ -s "$CFG" ] || { M "★cfg $CFG 不在★"; exit 2; }
mkdir -p /tmp/bugmd
M "发车: cfg $CFG; 侧车 $ZD; 每份 $PER; 标签 $TAG; 额外过尺 ${EXTRA:-无}"
gguf-tools/scripts/with_model_stopped_spark.sh "$0" --inner "$CFG" "$HF" "$ZD" "$PER" "$TAG" "$EXTRA"; rc=$?
# 包装器的装回是发给工作台主进程的 load 请求, 子进程要过几秒才 fork 出来(10-22 22:58 实撞: 紧跟着 pgrep 没看见就又叫了一次 model_restart);
# 等 30 s 还没有才算真没装回(包装器开始时模型就没在跑的情形), 这时才兜一手。
for _ in $(seq 1 15); do pgrep -x ds4-server >/dev/null && break; sleep 2; done
pgrep -x ds4-server >/dev/null || { M "模型没在跑, 装回"; gguf-tools/scripts/model_restart_spark.sh | tail -n 2; }
M "全部结束 rc=$rc"
exit "$rc"
