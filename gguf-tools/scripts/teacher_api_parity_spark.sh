#!/bin/bash
# teacher_api_parity_spark.sh — FP 教师 / file 教师 / 引擎 三方对官方 API 逐位 top-20 logprob 短程对拍(2026-10-25)。
#   为什么: FP 教师(精确 FP4 专家)在前 64 位就有 15~19 nat 尖峰、抄不了长文, 但逐层对引擎只见"VQ 噪声"; 引擎到底对不对,
#   FP 前向到底错不错, 只有官方 API 能裁。协议照 tests/test-vectors: API 贪心续写 N 位(thinking 关, top_logprobs 20),
#   ds4 用同一套聊天模板渲染提示 → ids = 提示 ids + 续写 ids → 三方 teacher-forcing → 逐步比 argmax / API token 的 logprob。
# 用法(spark 仓库根): nohup gguf-tools/scripts/teacher_api_parity_spark.sh <名> "<提示文本>" [续写位数=96] > /tmp/bugmd/apipar_<名>.log 2>&1 &
# 产物: gguf/v41judge/api_parity_<名>/{api.json, prompt_tokens.txt, seq.ids, eng.bin, fp.bin, file.bin, *.log}; 末尾打逐步表与汇总。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
NM="${1:?名}"; PROMPT="${2:?提示文本}"; NT="${3:-96}"
W=gguf-tools/scripts/with_model_stopped_spark.sh
PY="$HOME/v41env/bin/python"
AP=gguf-tools/scripts/v41_api_parity.py
D=gguf/v41judge/api_parity_$NM
QD=gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k
HF=hf/DeepSeek-V4.1-Flash-1017
KEY=gguf-tools/data/posttrain/deepseek_api.key
GG=$(head -1 gguf/serve_pick.txt 2>/dev/null | cut -d' ' -f1); GG="${GG:-gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf}"   # serve_pick 第 1 行 gguf, 第 2 行侧车目录
M(){ echo "[apipar $(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p "$D"

M "① 官方 API 贪心续写 $NT 位(已有 api.json 就复用, 贪心是决定论的)"
[ -s "$D/api.json" ] || "$PY" "$AP" fetch --key "$KEY" --prompt "$PROMPT" --out "$D/api.json" --max-tokens "$NT" || exit 1

run_all() {
    M "② 提示按部署同一拼法渲染(core_chat_frame.c: BOS <｜User｜>正文<｜Assistant｜></think>, 不思考) → ids"
    # `--dump-tokens -p` 走的是"已渲染文本"分词路(cli_main.c: 文本以 BOS 开头才按特殊 token 切), 裸文本进去没有 BOS/模板 ——
    # 第一趟就撞了这个: 17 个裸 token, 与 API 看到的上下文不同, 三方对 API 的差全是模板差。
    R="<｜begin▁of▁sentence｜><｜User｜>${PROMPT}<｜Assistant｜></think>"
    ./ds4 --cuda -m "$GG" --mem-budget-mb 110000 --no-dspark --dump-tokens -p "$R" > "$D/prompt_tokens.txt" 2> "$D/dump_tokens.log" || { M "★dump-tokens 失败★"; tail -3 "$D/dump_tokens.log"; return 1; }
    "$PY" "$AP" ids --json "$D/api.json" --prompt-ids "$D/prompt_tokens.txt" --hf "$HF" --out "$D/seq.ids" | tee "$D/ids.log" || return 1
    NP=$(grep -o "nprompt=[0-9]*" "$D/ids.log" | cut -d= -f2); echo "$NP" > "$D/nprompt.txt"
    N=$(wc -w < "$D/seq.ids")
    M "③ 引擎 --score-ids ($N 位)"
    ./ds4 --cuda -m "$GG" --mem-budget-mb 110000 --no-dspark --score-ids "$D/seq.ids" --score-out "$D/eng.bin" > "$D/eng.log" 2>&1 || { M "★引擎 rc=$?★"; tail -3 "$D/eng.log"; }
    for arm in fp file; do
        wd=$HF; [ "$arm" = file ] && wd=$QD
        M "④ 教师 $arm: $wd"
        "$PY" gguf-tools/scripts/v41_teacher.py "$wd" --ids "$D/seq.ids" --ntok "$N" --out "$D/$arm.bin" > "$D/$arm.log" 2>&1 || { M "★教师 $arm rc=$?★"; tail -3 "$D/$arm.log"; }
        grep -E "PPL" "$D/$arm.log" | tail -1
    done
}
export -f run_all M; export PY AP D HF QD GG PROMPT
"$W" bash -c run_all

M "⑤ 三方对 API"
"$PY" "$AP" compare --json "$D/api.json" --ids "$D/seq.ids" --nprompt "$(cat "$D/nprompt.txt")" --hf "$HF" \
    fp="$D/fp.bin" file="$D/file.bin" eng="$D/eng.bin" 2>&1 | grep -v "transformers\]"
M "完成"
