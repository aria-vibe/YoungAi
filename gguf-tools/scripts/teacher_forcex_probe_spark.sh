#!/bin/bash
# teacher_forcex_probe_spark.sh — 把教师每层 MoE 的输入钉成引擎的 x, 单看"同一个 x 进去, 专家算出来的 y 差多少"(2026-10-25)。
#   为什么: FP 教师(dq_fp4 精确专家)对引擎的 y 相对误差到 L20 以后全是 ≥0.95, 但那是 x 已经漂了之后的账, 分不清
#   是专家路算错还是上游漂移累积。--force-x 用引擎 dump 的 x_Lnn 做每层 MoE 入口 ⇒ 层间隔断, y 差 = 当层专家路的差。
#   两臂: FP 模式(精确 FP4 专家)应当只剩"VQ 量化噪声"量级(相对误差 0.3~0.5, 逐层逐位置均匀);
#        file 模式(VQ 专家, 与引擎同权重)是对照, 应当 ≈0。哪一层/哪一位置的 FP 臂冒出 ≫1 或范数比 ≫1.3, 病就在那。
# 用法(spark 仓库根): nohup gguf-tools/scripts/teacher_forcex_probe_spark.sh > /tmp/bugmd/teacher_forcex_<时间>.log 2>&1 &
# 产物: gguf/v41judge/locate_tprompt_1k/{engx64/, fx_fp/, fx_file/, fx_fp.log, fx_file.log}; 末尾打两臂逐层中位相对误差 + 逐位置图。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
W=gguf-tools/scripts/with_model_stopped_spark.sh
PY="$HOME/v41env/bin/python"
D=gguf/v41judge/locate_tprompt_1k
QD=gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k
HF=hf/DeepSeek-V4.1-Flash-1017
IDS=gguf/go-onebit/code/pelican/tprompt_1k.ids
N=64
M(){ echo "[fx $(date '+%m-%d %H:%M:%S')] $*"; }

# 引擎 dump 是 2001 行, --force-x 要求文件行数 == ntok ⇒ 切前 N 行成独立目录
"$PY" - "$D" "$N" <<'EOF'
import numpy as np, sys, os
d, n = sys.argv[1], int(sys.argv[2]); os.makedirs(f"{d}/engx{n}", exist_ok=True)
for L in range(40):
    np.fromfile(f"{d}/eng.bin.x_L{L:02d}.bin", np.float32)[:n*5120].tofile(f"{d}/engx{n}/x_L{L:02d}.bin")
print("engx 切好", n, "行 × 40 层")
EOF

run_arm() {   # $1 臂名 $2 权重目录
    local nm="$1" wd="$2"
    M "== 臂 $nm: $wd, MoE 入口钉引擎 x, ntok $N"
    "$PY" gguf-tools/scripts/v41_teacher.py "$wd" --ids "$IDS" --ntok "$N" --force-x "$D/engx$N" --dump-moe "$D/fx_$nm" --out "$D/fx_$nm.bin" > "$D/fx_$nm.log" 2>&1; local rc=$?
    grep -E "诊断|完成|PPL|Error|Traceback" "$D/fx_$nm.log" | tail -3
    [ "$rc" = 0 ] && [ -s "$D/fx_$nm/y_L39.bin" ] || { M "★臂 $nm rc=$rc★"; return 1; }
}
export -f run_arm M; export PY D IDS N
"$W" bash -c "run_arm fp $HF"
"$W" bash -c "run_arm file $QD"

"$PY" - "$D" "$N" <<'EOF'
import numpy as np, sys, os
d, n = sys.argv[1], int(sys.argv[2])
ld = lambda p: np.fromfile(p, np.float32)[:n*5120].reshape(n, 5120)
print("层 | FP臂 y 相对误差 中位/最大  范数比中位 | file臂 y 相对误差 中位/最大  范数比中位   (同一个引擎 x 进 MoE)")
maps = {"fp": [], "file": []}
for L in range(40):
    ey = ld(f"{d}/eng.bin.y_L{L:02d}.bin"); row = f"L{L:02d} |"
    for arm in ("fp", "file"):
        p = f"{d}/fx_{arm}/y_L{L:02d}.bin"
        if not os.path.exists(p): row += "   (无)   "; continue
        py = ld(p); en = np.maximum(np.linalg.norm(ey, axis=1), 1e-6)
        r = np.linalg.norm(py - ey, axis=1) / en; nr = np.linalg.norm(py, axis=1) / en
        maps[arm].append(r)
        row += f" {np.median(r):6.3f}/{r.max():6.3f} {np.median(nr):5.2f}      |"
    print(row)
for arm in ("fp", "file"):
    if not maps[arm]: continue
    print(f"\n{arm} 臂 y 相对误差图: 行=层, 列=位置 0..{n-1}(×10 取整, ≥0.95 打 #)")
    for L, r in enumerate(maps[arm]):
        print(f"L{L:02d} " + "".join("#" if v >= 0.95 else str(int(v * 10)) for v in r))
EOF
M "完成"
