#!/bin/bash
# teacher_mix_parts_spark.sh — 2×2 之后再细一层(2026-10-25): 量化目录为主, 只把 routed 专家(FP4, dq_fp4)或只把 shared 专家(FP8, dq_fp8)换成 FP 出厂权重。
#   已知: 整个 ffn 换 FP(臂 A)坏 PPL 6.88; 不换(file 档)干净 3.51。A1 = 只换 routed; A2 = 只换 shared。哪个坏, 病就在那种格式的精确权重路里。
# 用法(spark 仓库根): nohup gguf-tools/scripts/teacher_mix_parts_spark.sh > /tmp/bugmd/teacher_mixparts_<时间>.log 2>&1 &
# 产物: gguf/v41judge/locate_tprompt_1k/mixA{1,2}.{bin,log,rows}; 末尾按 250 位打 FP / file / A / A1 / A2 / 引擎 的 NLL。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
W=gguf-tools/scripts/with_model_stopped_spark.sh
PY="$HOME/v41env/bin/python"
D=gguf/v41judge/locate_tprompt_1k
QD=gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k
HF=hf/DeepSeek-V4.1-Flash-1017
IDS=gguf/go-onebit/code/pelican/tprompt_1k.ids
M(){ echo "[mixp $(date '+%m-%d %H:%M:%S')] $*"; }
run_arm() {   # $1 臂名 $2 ffn-part
    local nm="$1" part="$2"
    M "== 臂 $nm: 主 $QD, ffn $part ← $HF"
    "$PY" gguf-tools/scripts/v41_teacher.py "$QD" --ffn-from "$HF" --ffn-part "$part" --ids "$IDS" --ntok 2001 --out "$D/mix$nm.bin" > "$D/mix$nm.log" 2>&1; local rc=$?
    grep -E "诊断|完成|PPL|路由|Error|Traceback" "$D/mix$nm.log" | tail -4
    [ "$rc" = 0 ] && [ -s "$D/mix$nm.bin" ] || { M "★臂 $nm rc=$rc★"; return 1; }
    gguf-tools/bench/anchor_metrics --ref-raw "$D/mix$nm.bin" --ids "$IDS" --student "$D/eng.bin" --row-out "$D/mix$nm.rows" | grep -E "PPL|KLD|Same" | tr '\n' ' '; echo
}
export -f run_arm M; export PY D IDS QD HF
"$W" bash -c "run_arm A1 routed"
"$W" bash -c "run_arm A2 shared"
"$PY" - "$D" <<'EOF'
import numpy as np, os, sys
d = sys.argv[1]; e = np.loadtxt(f"{d}/rows"); fl = np.loadtxt(f"{d}/file_vs_eng.rows")
ld = lambda n: np.loadtxt(f"{d}/{n}.rows") if os.path.exists(f"{d}/{n}.rows") else None
A, A1, A2 = ld("mixA"), ld("mixA1"), ld("mixA2")
print("pos   FP路  file路  A:ffn全FP  A1:只routed FP4  A2:只shared FP8  引擎")
for s in range(0, 2000, 250):
    m = (e[:, 0] >= s) & (e[:, 0] < s + 250); g = lambda r: (r[m, 8].mean() if r is not None else float("nan"))
    print(f"{s:5d} {e[m, 8].mean():6.2f} {fl[m, 8].mean():6.2f} {g(A):6.2f} {g(A1):6.2f} {g(A2):6.2f} {e[m, 9].mean():6.2f}")
EOF
M "完成"
