#!/bin/bash
# teacher_fault_chain_spark.sh — FP 教师长程失灵的三连判决(2026-10-25), 一条命令串起来, 每段各自停/装回模型。序列统一 tprompt_1k(BOS+X1000+X1000 = 2001 位):
#   ① teacher_layer_locate_spark.sh: 引擎(裸) vs FP 教师 40 层 × 4 位置段 + FP 教师索引器选中位(idx/);
#   ② 今天的 v41_teacher.py 改读【量化目录】(file 档: 骨架 q4_K + 专家 VQ 走 C 解码, 代码/配置与 FP 路同一份)跑同一序列 + 选中位
#      → 第二份 X 能抄 = 病只跟精确权重走; 再逐查询比 FP 路 vs file 路 top-512 选择集的重叠(640 位前/后、第二份 X 三段);
#   ③ ① 的 --no-engram 版(两边都摘 engram)。
# 用法(spark 仓库根): nohup gguf-tools/scripts/teacher_fault_chain_spark.sh > /tmp/bugmd/teacher_fault_chain_<时间>.log 2>&1 &
# 产物: gguf/v41judge/locate_tprompt_1k{,_noeng}/{parity.txt,rows,idx/} ; gguf/v41judge/locate_tprompt_1k/file_q4k.{bin,rows,log} + fileidx/
# 出错会怎样: 某段 rc≠0 继续下一段(各段独立判读); 全部跑完模型按 serve_pick 装回。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
W=gguf-tools/scripts/with_model_stopped_spark.sh
PY="$HOME/v41env/bin/python"
M(){ echo "[chain $(date '+%m-%d %H:%M:%S')] $*"; }
QD=gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k
IDS=gguf/go-onebit/code/pelican/tprompt_1k.ids
D=gguf/v41judge/locate_tprompt_1k
M "① locate(engram 开) → $D"
"$W" gguf-tools/scripts/teacher_layer_locate_spark.sh "$IDS"
M "② 教师脚本读量化目录 $QD → $D/file_q4k.bin"
"$W" bash -c "$PY gguf-tools/scripts/v41_teacher.py $QD --ids $IDS --ntok 2001 --out $D/file_q4k.bin --dump-idx $D/fileidx > $D/file_q4k.log 2>&1; rc=\$?; grep -E '完成|PPL|★|Error|Traceback' $D/file_q4k.log | tail -3; [ \$rc = 0 ] && gguf-tools/bench/anchor_metrics --ref-raw $D/teacher.bin --ids $IDS --student $D/file_q4k.bin --row-out $D/file_q4k.rows | grep -E 'KLD|Same top|Σmin|PPL' | tr '\n' ' '; echo; exit \$rc"
if [ -s "$D/file_q4k.rows" ]; then
    M "file 档 vs 引擎 vs FP 教师: 第二份 X 的 NLL(按 250 位) + 选中位重叠"
    "$PY" - "$D" <<'EOF'
import numpy as np, sys, glob, os
d = sys.argv[1]
eng = np.loadtxt(f"{d}/rows"); fil = np.loadtxt(f"{d}/file_q4k.rows")   # 列: pos kld smin same ref_hit stu_hit ref_argmax stu_argmax ref_nll stu_nll
print("pos    FP教师  引擎   file档")
for s in range(0, 2000, 250):
    m = (eng[:, 0] >= s) & (eng[:, 0] < s + 250)
    print(f"{s:5d}  {eng[m, 8].mean():5.2f}  {eng[m, 9].mean():5.2f}  {fil[m, 9].mean():5.2f}")
segs = [(0, 640), (640, 1001), (1001, 2001)]
print("索引源层 top-512 选择集重叠(Jaccard 均值) FP 路 vs file 路:", " / ".join(f"[{a},{b})" for a, b in segs))
for f in sorted(glob.glob(f"{d}/idx/idx_L*.bin")):
    L = os.path.basename(f)[5:7]; g = f"{d}/fileidx/idx_L{L}.bin"
    if not os.path.exists(g): continue
    a = np.fromfile(f, np.int32); b = np.fromfile(g, np.int32)
    k = a.size // 2001; a = a.reshape(2001, k); b = b.reshape(2001, k)
    out = []
    for lo, hi in segs:
        js = []
        for t in range(lo, hi):
            sa = set(a[t][a[t] >= 0].tolist()); sb = set(b[t][b[t] >= 0].tolist())
            if sa or sb: js.append(len(sa & sb) / len(sa | sb))
        out.append(f"{np.mean(js):.3f}" if js else "-")
    print(f"  L{L}: " + " / ".join(out))
EOF
fi
M "③ locate(--no-engram)"
"$W" gguf-tools/scripts/teacher_layer_locate_spark.sh "$IDS" --no-engram
M "完成"
