#!/bin/bash
# teacher_mix_arms_spark.sh — FP 教师长程失灵的 2×2 二分(2026-10-25): 同一份 v41_teacher.py, 骨架与 ffn 分别从 FP 出厂权重 / 量化目录装(--ffn-from)。
#   A = 量化目录为主 + ffn 从 FP(q4_K 骨架 + 精确 FP4 专家);  B = FP 为主 + ffn 从量化目录(精确骨架 + VQ 专家)。
# 已知: FP 全精确 坏(PPL 6.91) / 量化目录 干净(3.51, = 引擎)。哪一臂坏, 病就在那一半精确权重的装载/数值里。
# 用法(spark 仓库根): nohup gguf-tools/scripts/teacher_mix_arms_spark.sh > /tmp/bugmd/teacher_mix2x2_<时间>.log 2>&1 &
# 产物: gguf/v41judge/locate_tprompt_1k/mix{A,B}.{bin,log,rows}; 末尾按 250 位打 FP / file / A / B / 引擎 的 NLL。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
W=gguf-tools/scripts/with_model_stopped_spark.sh
PY="$HOME/v41env/bin/python"
D=gguf/v41judge/locate_tprompt_1k
QD=gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k
HF=hf/DeepSeek-V4.1-Flash-1017
IDS=gguf/go-onebit/code/pelican/tprompt_1k.ids
M(){ echo "[mix $(date '+%m-%d %H:%M:%S')] $*"; }
run_arm() {   # $1 臂名 $2 主目录 $3 ffn 目录
    local nm="$1" main="$2" ffn="$3"
    M "== 臂 $nm: 主 $main, ffn ← $ffn"
    "$PY" gguf-tools/scripts/v41_teacher.py "$main" --ffn-from "$ffn" --ids "$IDS" --ntok 2001 --out "$D/mix$nm.bin" > "$D/mix$nm.log" 2>&1; local rc=$?
    grep -E "诊断|完成|PPL|路由|Error|Traceback" "$D/mix$nm.log" | tail -4
    [ "$rc" = 0 ] && [ -s "$D/mix$nm.bin" ] || { M "★臂 $nm rc=$rc★"; return 1; }
    gguf-tools/bench/anchor_metrics --ref-raw "$D/mix$nm.bin" --ids "$IDS" --student "$D/eng.bin" --row-out "$D/mix$nm.rows" | grep -E "PPL|KLD|Same" | tr '\n' ' '; echo
}
export -f run_arm M; export PY D IDS
"$W" bash -c "run_arm A $QD $HF"
"$W" bash -c "run_arm B $HF $QD"
"$PY" - "$D" <<'EOF'
import numpy as np, os, sys
d = sys.argv[1]; e = np.loadtxt(f"{d}/rows"); fl = np.loadtxt(f"{d}/file_vs_eng.rows")
A = np.loadtxt(f"{d}/mixA.rows") if os.path.exists(f"{d}/mixA.rows") else None
B = np.loadtxt(f"{d}/mixB.rows") if os.path.exists(f"{d}/mixB.rows") else None
print("pos   FP路  file路  A:q4K骨架+FP专家  B:FP骨架+VQ专家  引擎")
for s in range(0, 2000, 250):
    m = (e[:, 0] >= s) & (e[:, 0] < s + 250); g = lambda r: (r[m, 8].mean() if r is not None else float("nan"))
    print(f"{s:5d} {e[m, 8].mean():6.2f} {fl[m, 8].mean():6.2f} {g(A):6.2f} {g(B):6.2f} {e[m, 9].mean():6.2f}")
EOF
M "完成"
