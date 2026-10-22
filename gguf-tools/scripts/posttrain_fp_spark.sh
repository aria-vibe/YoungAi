#!/bin/bash
# posttrain_fp_spark.sh — ③ 向出厂原件蒸馏的整链(2026-10-22, docs/think_close.md §5): 装料导序列 → FP 参考前向打 top-K → 训 ③。
#
# 用法(spark 仓库根; 本脚本不停模型, 包在停模型包装器里跑):
#   nohup gguf-tools/scripts/with_model_stopped_spark.sh gguf-tools/scripts/posttrain_fp_spark.sh <ptrain.cfg> <HF 出厂权重目录> <② 侧车目录|none> [每趟打分最多几条=96] [附加序列文件] > /tmp/bugmd/pt_fp_<时间>.log 2>&1 &
#   每趟条数的内存账(10-22 实测, 96 条 / 5.6 万位 / 最长 1001): 教师进程整机占 73 GB(CUDA 分配不进 RSS, 看 free 的 used), 可用 51 GB;
#   73 GB 里多少随条数长没拆过, 加条数先在新趟开头看 used 的涨幅, 不盲加。
#   ptrain.cfg 照工作台写法(data= hold= out= epochs= layers= …, 见 model_train_job.c write_cfg), ★不要写 teacher=/dumpids=★: 本脚本按阶段自己加。
#   附加序列文件(可选) = 一行一条 id 串, 并进第一份一起打分: 表里多出来的序列 C 侧不理会, 专给 bugmd_ids_tools.py ptfcheck 对拍 batch 维路 vs 全 logits 锚(第一次用新路必过)。
#   底座取 gguf/serve_pick.txt 第 1 行; ② 侧车显式给(10-22 用户定: 基于 10-18 的原始编程侧车 …-engine.1018), 不读 serve_pick 第 2 行 ——
#   ③ 的起点 = 挂着这份 ② 的模型(base.fnv 关联它), 部署时也要挂同一份; 不改 serve_pick 是为了不顺手改生产装回时的配置。
# 阶段: ① ./ds4 --ptrain <cfg + dumpids=OUT/teacher.ids>: 只装料、把每道题的教师序列去重写出来(core_ptrain_fp.c), 分钟级
#       ② OUT/teacher.ids 按 N 条一份切开, 每份 v41_teacher.py <HF> --seqs 份 --topk K --out 份.ptf(每份一趟流式装原件; 已有 .ptf 的份保留, 只切没打的), cat 成 OUT/teacher.ptf
#       ③ ./ds4 --ptrain <cfg + teacher=fp:OUT/teacher.ptf>: 训 ③, OUT/ckpt_eNN + train.log(多了 decision 行); 选轮由工作台 model_train_pick.c 读 train.log
# 判读: 末尾打 train.log 的 step 0 / epoch / decision 行; 决策位 KL 逐轮比第 0 步低才是在修"停"(docs/think_close.md §4)。
# 出错会怎样: 任一阶段 rc≠0 立刻停(不接着训一个没教师的 ③); 表里缺题 ds4 自己停车; 教师端 MemAvailable 连续两次 < 2500 MB 杀掉 python(与 v41_judge.sh 同一条看门狗线)。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
M(){ echo "[ptfp $(date '+%m-%d %H:%M:%S')] $*"; }
CFG="${1:?ptrain.cfg}"; HF="${2:?HF 出厂权重目录}"; ZD="${3:?② 侧车目录, 不挂写 none}"; PER="${4:-96}"; EXTRA="${5:-}"
[ -s "$CFG" ] && [ -s "$HF/config.json" ] || { M "★cfg $CFG 或 HF 目录 $HF 不对★"; exit 2; }
[ -z "$EXTRA" ] || [ -s "$EXTRA" ] || { M "★附加序列文件 $EXTRA 不在★"; exit 2; }
grep -qE '^(teacher|dumpids)=' "$CFG" && { M "★cfg 里不要写 teacher=/dumpids=, 本脚本按阶段加★"; exit 2; }
OUTD=$(grep '^out=' "$CFG" | head -1 | cut -d= -f2- | tr -d ' '); [ -n "$OUTD" ] || { M "★cfg 缺 out=★"; exit 2; }
K=$(grep '^topk=' "$CFG" | head -1 | cut -d= -f2- | tr -d ' '); K="${K:-64}"
GG=$(sed -n 1p gguf/serve_pick.txt); GG="${GG#"$ROOT"/}"
[ -s "$GG" ] || { M "★serve_pick 第 1 行 $GG 不是文件★"; exit 2; }
ZARG=(); [ "$ZD" = none ] || { ZD="${ZD#"$ROOT"/}"; grep -q "^# 完成" "$ZD/manifest.txt" 2>/dev/null || { M "★侧车 $ZD 没有完成标记的 manifest★"; exit 2; }; ZARG=(--zchain "$ZD"); }
mkdir -p "$OUTD"
M "底座 $GG; 侧车 ${ZD:-none}; 料 $(grep '^data=' "$CFG" | cut -d= -f2-); 出 $OUTD; top-K $K; HF $HF"
# 训练子进程的旗照工作台(model_train_job.c): 预算 110000 = 机器空着整机给它; --no-dspark 不建草稿器
DS4=(./ds4 --cuda -m "$GG" "${ZARG[@]}" --mem-budget-mb 110000 --no-dspark)

# ① 导教师序列
{ cat "$CFG"; echo "dumpids=$OUTD/teacher.ids"; } > "$OUTD/stage1.cfg"
M "① 导教师序列 → $OUTD/teacher.ids"
"${DS4[@]}" --ptrain "$OUTD/stage1.cfg" 2>&1 | grep -E "ptrain\]|★|错|失败" | tail -6
[ -s "$OUTD/teacher.ids" ] || { M "★① 没出 teacher.ids★"; exit 3; }
NSEQ=$(wc -l < "$OUTD/teacher.ids"); M "① 完成: $NSEQ 条序列"

# ② FP 参考前向打 top-K(分份; 看门狗: MemAvailable 连续两次 < 2500 MB 杀 python)
# 分份是续跑式的(10-22 下午改): 已有 .ptf 的份原样保留, 只把 teacher.ids 里还没打过的序列重新切份、编号接着往后排 ——
#   每份耗时与条数几乎无关(实测 96 条 2220~2507 s, 花的是装一遍原件和逐专家来回的固定开销, 不是算), 所以 ① 尾巴那份哪怕
#   只有几条也要跑满一遍, 中途加大 PER 不该重打已有的份; ② 没跑完的半份(.ptf 没落盘)连 id 文件一起丢掉重切。
#   表按序列内容配(core_ptrain_fp.c memcmp 整条 id), 份的顺序/大小对训练无影响。
for f in "$OUTD"/teacher.part.[0-9][0-9][0-9]; do [ -e "$f" ] || continue; [ -s "$f.ptf" ] || rm -f "$f" "$f.log"; done
KEEP=("$OUTD"/teacher.part.[0-9][0-9][0-9]); [ -e "${KEEP[0]}" ] || KEEP=()
if [ "${#KEEP[@]}" -gt 0 ]; then
    awk 'NR==FNR{d[$0]=1; next} !($0 in d)' <(cat "${KEEP[@]}") "$OUTD/teacher.ids" > "$OUTD/teacher.rest"
    M "② 已有 ${#KEEP[@]} 份 .ptf(覆盖 $(cat "${KEEP[@]}" | wc -l) 条), 剩 $(wc -l < "$OUTD/teacher.rest") 条按每份 $PER 重新切"
else
    cp "$OUTD/teacher.ids" "$OUTD/teacher.rest"
fi
split -l "$PER" --numeric-suffixes="${#KEEP[@]}" -a 3 "$OUTD/teacher.rest" "$OUTD/teacher.part."
rm -f "$OUTD/teacher.rest"
PARTS=("$OUTD"/teacher.part.[0-9][0-9][0-9])
if [ -n "$EXTRA" ]; then   # 对拍序列并进第一份【新】份: 它们在表里排在这一份训练序列之后(序号 = 本份训练条数 + 第几行)
    first=""; for p in "${PARTS[@]}"; do [ -s "$p.ptf" ] || { first="$p"; break; }; done
    if [ -n "$first" ]; then
        n0=$(wc -l < "$first"); cat "$EXTRA" >> "$first"
        M "② 附加 $(wc -l < "$EXTRA") 条对拍序列进 $(basename "$first")(表内序号从 $n0 起)"
    else M "② 没有要打的新份, 附加序列 $EXTRA 不打"; fi
fi
M "② FP 教师: ${#PARTS[@]} 份(其中 ${#KEEP[@]} 份已有) × ≤$PER 条, python $HOME/v41env/bin/python"
for p in "${PARTS[@]}"; do
    if [ -s "$p.ptf" ]; then M "  $(basename "$p") 已有 .ptf, 跳过"; continue; fi
    M "  $(basename "$p") $(wc -l < "$p") 条 → $(basename "$p").ptf(MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB)"
    "$HOME/v41env/bin/python" gguf-tools/scripts/v41_teacher.py "$HF" --seqs "$p" --topk "$K" --out "$p.ptf" > "$p.log" 2>&1 &
    PY=$!; low=0
    while kill -0 "$PY" 2>/dev/null; do
        sleep 5
        av=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
        if [ "$av" -lt 2500 ]; then low=$((low + 1)); [ "$low" -ge 2 ] && { M "★看门狗: MemAvailable $av MB 连续两次 < 2500, 杀教师 $PY★"; kill -9 "$PY"; }; else low=0; fi
    done
    wait "$PY"; rc=$?
    grep -E "\[seqs\]|\[完成\]|→|★|Error|Traceback|Killed" "$p.log" | tail -5
    [ "$rc" = 0 ] && [ -s "$p.ptf" ] || { M "★教师 $(basename "$p") rc=$rc, 看 $p.log★"; exit 4; }
    # 每份落盘就看教师在决策位给了多少(10-22 用户: 早点发现问题): 不用 GPU; 教师在 EOS/</think> 位给得很低先怀疑料(尾段切坏/渲染错), 别等 4 小时后训完才知道
    "$HOME/v41env/bin/python" speed-bench/bugmd_ids_tools.py ptfstat "$p.ptf" 2>&1 | tail -12
done
cat "${PARTS[@]/%/.ptf}" > "$OUTD/teacher.ptf"
M "② 完成: $OUTD/teacher.ptf $(du -h "$OUTD/teacher.ptf" | cut -f1)"

# ③ 训
{ cat "$CFG"; echo "teacher=fp:$OUTD/teacher.ptf"; } > "$OUTD/stage3.cfg"
M "③ 训 ③(cfg $OUTD/stage3.cfg) → $OUTD"
"${DS4[@]}" --ptrain "$OUTD/stage3.cfg" > "$OUTD/train.out" 2>&1; rc=$?
grep -E "FP 教师表|决策位|★轮|第 0 步|已落|★|失败" "$OUTD/train.out" | tail -20
[ "$rc" = 0 ] || { M "★③ 训练 rc=$rc, 看 $OUTD/train.out★"; exit 5; }
M "train.log 读数:"; grep -E "^step 0|^epoch|^decision|^probe" "$OUTD/train.log"
M "完成 → $OUTD(ckpt_eNN 各轮; 挂法 --posttrain $OUTD/ckpt_eNN)"
