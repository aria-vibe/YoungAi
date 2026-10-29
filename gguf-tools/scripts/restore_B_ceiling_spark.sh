#!/bin/bash
# restore_B_ceiling_spark.sh — 段 B 的天花板(2026-10-28 用户令"跑段b的天花板"): 拿编程尺自己的 FP 原件读数当靶, 看秩 64 的段 B 最多能把尺推到哪。
#   只是诊断, 产物不进现役: 现役侧车目录全程只读(训练与判决都挂临时侧车目录: 段 A 文件软链过去, 段 B 换成要判的那份)。
#
# 用法(spark ~/ds4-main, 工作台开着): nohup bash gguf-tools/scripts/restore_B_ceiling_spark.sh [--lr 5e-5] [--epochs 12] [--batch 1] [--judge "6 12"] > /tmp/bugmd/ceiling.log 2>&1 &
#   整条只停一次模型(停模型包装回调自己 --_inner), 跑完按 serve_pick.txt 装回。被杀了模型可能没装回: with_model_stopped_spark.sh --load。
# 设计:
#   尺 = BOS + 编程判决份 15360 个 token(gguf/go-onebit/code/judge_bos.ids)。为什么不用原尺 judge.ids: 它开头没有 BOS, 训练器每条都补 BOS(core_ptrain_data.c),
#     靶要和训练看到的序列逐位同一串; 序列由 api_teacher.py simids 出(与训练器同一种切分), FP 锚由 v41_judge.sh 在这把尺上现算(一趟流式装原件)。
#   料 = {prefix: 尺的前 7680 个 token(前 15 种语言), text: 后 7680 个(后 15 种语言)} + 一道尺子开头 ~32 位的短题(训练器预热要, 见 ①): prefix 只当上下文不出梯度 ⇒
#     后半 = 背得下多少(容量), 前半 = 没练过的语言跟着动多少(泛化; 因果注意力下前半的读数不受后半影响)。
#   靶 = FP 锚转的 top-64 表(bugmd_ids_tools.py anchor2ptf); 起点 = 现役段 B; ./ds4 --restore-b, 每遍一个 ckpt, --judge 列出的几遍挂上判。
# 判读: 末尾每个 ckpt 两行配对(对现役, 同一把尺): 后半(练过) / 前半(没练过)的 KLD 与 Same top, 外加 Same top 没中位按 FP 把握分桶。
#   练过的一半都上不了 87~88% ⇒ 秩 64 的段 B 结构上到不了 90; 练过的上去、没练过的不动 ⇒ 有容量, 卡在靶与料。
# --aux <w>(10-29, 逐层靶, core_restoreb_aux.c): 同一份料/靶/超参, 训练损失多一项"每层 xn·B·A 去补 y_fp(x_q) − y_q"(w = 第 1 步相对出口 KL 的梯度范数)。
#   多两步: 当前链把尺整条取料(./ds4 --restore-b rb_dump=, 每层 x / 放大器前的 y) → FP 原件 --force-x 钉同一个 x 逐层算 y_fp(x_q);
#   靶目录 $D/auxtgt(40 层 × x/yq/y 三份 f32, 约 38 GB)已有就复用。产物 $OUT-aux<w>, 判的臂叫 aux<w>_eNN; 对照 = 不带 --aux 那趟(天花板, 10-28)。
#   判读: 没练过的前半比天花板那趟涨得多 ⇒ 逐层靶把段 B 往"权重的账"上钉住了, 泛化范式成立; 一样平 ⇒ 不成立。(10-29 两种逐层靶都判负, fable5)
# --route <lr>(10-29, 带路由的段 B, src/cuda/cuda_amp_route.inc.cu): 同一份料/靶/超参, 段 B 多一张每个专家一组秩系数(lr 单给, 从 0 起 = 现役段 B)。
#   产物 $OUT-route<lr>(ckpt 多 ampr_Lnn.bin), 判的臂叫 route<lr>_eNN, 同样对同遍数的天花板配对。不与 --aux 同开。
# --winmask(10-29, 同语言留出): 料 = 整把尺一条(+ 短题), 按 judge.ids.layout 的窗练一半留一半 —— 每种语言 4 个 128 位窗, 第 1/3 窗练、第 2/4 窗留
#   (窗间在源语料里是跳着取的 ⇒ 留出 = 同语言、没见过的上下文)。训练器只让练的窗出梯度(./ds4 --restore-b rb_mask=)。
#   尺 = 整篇当 text 时训练器看到的序列(judge_bos_full.ids, 与 judge_bos.ids 在前后半分界处分词可能不同 ⇒ FP 锚另算一份), 工作目录 evo/ceilwin, 产物 restore-ceilwin。
#   对照 = 现役; 可与 --route 同开(臂名 route<lr>_eNN)。判读: 留出的窗涨 ⇒ 同语言能迁; 不涨 ⇒ 逐位蒸馏只会背。
# --rank <K> [--freeze <n>](10-29): 段 B 的秩(缺省 64); --freeze n = 起点(现役)的前 n 个秩冻住只训多出来的(rb_freeze=)。
#   例: --winmask --rank 72 --freeze 64 = 现役秩 64 不动, 上面加 8 个秩 —— 量"参数少了是不是就背不下、只能学跨上下文的修正"。臂名 r<K>f<n>_eNN。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
LR=5e-5; EPOCHS=12; BATCH=1; JUDGE="6 12"; INNER=0; AUXW=; RLR=; WMASK=0; RANK=64; FRZ=
while [ $# -gt 0 ]; do case "$1" in
    --lr) LR="${2:?}"; shift 2;; --epochs) EPOCHS="${2:?}"; shift 2;; --batch) BATCH="${2:?}"; shift 2;; --judge) JUDGE="${2:?}"; shift 2;;
    --aux) AUXW="${2:?}"; shift 2;; --route) RLR="${2:?}"; shift 2;; --winmask) WMASK=1; shift;; --rank) RANK="${2:?}"; shift 2;; --freeze) FRZ="${2:?}"; shift 2;;
    --_inner) INNER=1; shift;; *) echo "★不认识的参数 $1★"; exit 2;; esac; done
PY="$HOME/v41env/bin/python"; HF=hf/DeepSeek-V4.1-Flash-1017; W=gguf-tools/scripts/with_model_stopped_spark.sh; BT=speed-bench/bugmd_ids_tools.py
D=gguf/go-onebit/evo/ceiling; SRC=gguf/go-onebit/code/judge.ids; RULER=gguf/go-onebit/code/judge_bos.ids; HALF=7680
OUT=gguf/v41/posttrain/restore-ceiling; TAG=""; AX=$D/auxtgt
if [ "$WMASK" = 1 ]; then [ -z "$AUXW" ] || { echo "★--winmask 不与 --aux 同开(逐层靶是在另一把尺上取的)★"; exit 2; }
    D=gguf/go-onebit/evo/ceilwin; RULER=gguf/go-onebit/code/judge_bos_full.ids; OUT=gguf/v41/posttrain/restore-ceilwin; fi
if [ "$RANK" != 64 ] || [ -n "$FRZ" ]; then OUT="$OUT-r$RANK${FRZ:+f$FRZ}"; TAG="${TAG}r$RANK${FRZ:+f$FRZ}_"; fi
[ -z "$AUXW" ] || { OUT="$OUT-aux$AUXW"; TAG="aux${AUXW}_"; }
[ -z "$RLR" ] || { [ -z "$AUXW" ] || { echo "★--route 不与 --aux 同开★"; exit 2; }; OUT="$OUT-route$RLR"; TAG="route${RLR}_"; }
mapfile -t PKS < gguf/serve_pick.txt; GG="${PKS[0]#"$ROOT"/}"; ARM="${PKS[1]#"$ROOT"/}"; GB=$(basename "$GG" .gguf)
M(){ echo "[ceil $(date '+%m-%d %H:%M:%S')] $*"; }
die(){ M "★$*★"; echo "CEIL_EXIT 1"; exit 1; }
for v in "$EPOCHS" $JUDGE; do [ "$v" -ge 1 ] 2>/dev/null || die "--epochs/--judge 要正整数"; done
for j in $JUDGE; do [ "$j" -le "$EPOCHS" ] || die "--judge $j 超过 --epochs $EPOCHS"; done

if [ "$INNER" = 0 ]; then
    mkdir -p "$D"
    M "① 料与尺: $SRC 前 $HALF 位当上下文、后 $HALF 位当靶"
    # 两行料: 第 1 行是尺子开头 ~32 位的短题, 第 2 行是正料。为什么要短题(10-28 第一趟实撞): 训练器预热先挑一道装得进单包(st.cap_tok)的题走一趟,
    #   正料拆出的 9 段每段都带着上万位前缀、没有一道装得进 ⇒ 空包 ⇒ pt_pack_step(nb = 0) 返回失败, 训练没起来。后训练代码不许动, 就在料上补一道短题:
    #   它的序列是尺子序列的前缀, FP 表按"表序列以本题开头"对得上, 不用另跑原件; 代价是这 ~32 位被练过 ⇒ "没练过的前半"从它后面算起
    if [ "$WMASK" = 1 ]; then
    M "  --winmask: 料 = 整把尺一条 + 短题, 按窗练一半留一半"
    "$PY" - "$HF" "$SRC" "$D/rows.jsonl" <<'EOF' || die "造料失败"
import json, sys
from tokenizers import Tokenizer
hf, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
tok = Tokenizer.from_file(hf + "/tokenizer.json"); ids = [int(x) for x in open(src).read().split()]
full = tok.decode(ids); p = tok.encode(full, add_special_tokens=False).ids
for k in range(32, 200):   # 短题(训练器预热要, 同 ①): 整篇分词结果的前 k 个, 能逐位回得来、结尾不是空白
    t = tok.decode(p[:k])
    if t == t.rstrip(" \r\n") and tok.encode(t, add_special_tokens=False).ids == p[:k]: break
else: sys.exit("★找不到能逐位回得来的短题★")
with open(out, "w", encoding="utf-8") as f:
    f.write(json.dumps({"text": t, "eos": False}, ensure_ascii=False) + "\n")
    f.write(json.dumps({"text": full, "eos": False}, ensure_ascii=False) + "\n")
EOF
    "$PY" gguf-tools/scripts/api_teacher.py simids --hf "$HF" --rows "$D/rows.jsonl" --out "$D/rows.seqs" || die "simids 失败"
    sed -n 2p "$D/rows.seqs" | tr ' ' '\n' | grep -v '^$' > "$RULER"; N=$(wc -l < "$RULER")
    K=$(sed -n 1p "$D/rows.seqs" | wc -w)
    [ "$(sed -n 1p "$D/rows.seqs")" = "$(head -n "$K" "$RULER" | tr '\n' ' ' | sed 's/ $//')" ] || die "短题不是尺子序列的前缀, FP 表对不上"
    # 窗 → 尺上的位置: 原尺第 b 行的窗起点在整篇文本里的字符位 = len(decode(原尺[:b])), 尺上第 p 位(p ≥ 1)按它的字符起点归窗(窗缝处分词合并会让边界差 ±1 位)
    "$PY" - "$HF" "$SRC" "$SRC.layout" "$RULER" "$D/train.lst" "$D/held.lst" "$D/mask.txt" <<'EOF' || die "切窗失败"
import sys, bisect
from tokenizers import Tokenizer
hf, src, lay, ruler, otr, oho, omk = sys.argv[1:8]
tok = Tokenizer.from_file(hf + "/tokenizer.json"); ids = [int(x) for x in open(src).read().split()]
R = [int(x) for x in open(ruler).read().split()]
full = tok.decode(ids); enc = tok.encode(full, add_special_tokens=False)
if enc.ids != R[1:]: sys.exit("★整篇分词与尺(去 BOS)不逐位同, 字符位对不上★")
W, doms = 128, []
for ln in open(lay):
    t = ln.split()
    if not t or t[0].startswith("#"): continue
    if t[0] == "win": W = int(t[1]); continue
    doms.append((t[0], int(t[1]), int(t[2])))
bd = sorted((len(tok.decode(ids[:st + k * W])), k % 2 == 0) for _, st, n in doms for k in range(n // W))
cs = [b[0] for b in bd]
flag = [None] + [bd[max(0, bisect.bisect_right(cs, enc.offsets[p - 1][0]) - 1)][1] for p in range(1, len(R))]
def ranges(want):
    r, s0 = [], None
    for p in range(1, len(R) + 1):
        f = flag[p] if p < len(R) else None
        if f == want and s0 is None: s0 = p
        if f != want and s0 is not None: r.append((s0, p)); s0 = None
    return r
tr, ho = ranges(True), ranges(False)
for path, rg in ((otr, tr), (oho, ho)): open(path, "w").write(f"ruler {len(R)} " + ",".join(f"{a}:{b}" for a, b in rg) + "\n")
open(omk, "w").write("".join(f"{a} {b}\n" for a, b in tr))
print(f"  {len(doms)} 个语言 × {doms[0][2] // W} 窗; 练 {sum(b - a for a, b in tr)} 位({len(tr)} 段), 留 {sum(b - a for a, b in ho)} 位({len(ho)} 段)")
EOF
    M "  尺 $RULER: $N 位; 短题 1:$K(落在第 1 个练窗里)"
    else
    "$PY" - "$HF" "$SRC" "$HALF" "$D/rows.jsonl" <<'EOF' || die "造料失败"
import json, sys
from tokenizers import Tokenizer
hf, src, half, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
tok = Tokenizer.from_file(hf + "/tokenizer.json"); ids = [int(x) for x in open(src).read().split()]
pre = tok.decode(ids[:half]); p = tok.encode(pre, add_special_tokens=False).ids
for k in range(32, 200):   # 短题 = 前缀分词结果的前 k 个: 要解码再分词逐位回得来、结尾不是空白(simids / 训练器会剥尾部空白)
    t = tok.decode(p[:k])
    if t == t.rstrip(" \r\n") and tok.encode(t, add_special_tokens=False).ids == p[:k]: break
else: sys.exit("★找不到能逐位回得来的短题★")
with open(out, "w", encoding="utf-8") as f:
    f.write(json.dumps({"text": t, "eos": False}, ensure_ascii=False) + "\n")
    f.write(json.dumps({"prefix": pre, "text": tok.decode(ids[half:]), "eos": False}, ensure_ascii=False) + "\n")
EOF
    "$PY" gguf-tools/scripts/api_teacher.py simids --hf "$HF" --rows "$D/rows.jsonl" --out "$D/rows.seqs" || die "simids 失败"
    sed -n 2p "$D/rows.seqs" | tr ' ' '\n' | grep -v '^$' > "$RULER"; N=$(wc -l < "$RULER")
    K=$(sed -n 1p "$D/rows.seqs" | wc -w)   # 短题 = BOS + k 位; 它的靶是尺子第 1..k 位
    [ "$(sed -n 1p "$D/rows.seqs")" = "$(head -n "$K" "$RULER" | tr '\n' ' ' | sed 's/ $//')" ] || die "短题不是尺子序列的前缀, FP 表对不上"
    # 前半(prefix)在序列里的止位 = BOS + prefix 自己分词的长度; 训练器同一个分界(pt_add_text 两段各自分词再拼)
    H=$("$PY" -c "import sys; from tokenizers import Tokenizer; import json; t=Tokenizer.from_file(sys.argv[1]+'/tokenizer.json'); r=json.loads(open(sys.argv[2]).readlines()[1]); print(1+len(t.encode(r['prefix'], add_special_tokens=False).ids))" "$HF" "$D/rows.jsonl")
    SAME=$(cmp -s <(tail -n +2 "$RULER") "$SRC" && echo "与原尺逐位同(只多开头 BOS)" || echo "★与原尺不逐位同(解码再分词在窗缝合并了), 以新尺为准★")
    printf 'ruler %s %s:%s\n' "$N" "$K" "$H" > "$D/held.lst"; printf 'ruler %s %s:%s\n' "$N" "$H" "$N" > "$D/train.lst"
    M "  尺 $RULER: $N 位, $SAME; 短题练 1:$K / 没练过 $K:$H / 练过 $H:$N"
    fi
    M "停模型, 余下全在包装里跑(现役侧车只读)"
    "$W" bash "$0" --_inner --lr "$LR" --epochs "$EPOCHS" --batch "$BATCH" --judge "$JUDGE" ${AUXW:+--aux "$AUXW"} ${RLR:+--route "$RLR"} $([ "$WMASK" = 1 ] && echo --winmask) --rank "$RANK" ${FRZ:+--freeze "$FRZ"}; rc=$?
    echo "CEIL_EXIT $rc"; exit "$rc"
fi

# ---- 以下在停模型包装里 ----
N=$(wc -l < "$RULER"); K=$(sed -n 1p "$D/rows.seqs" | wc -w)
side(){   # 临时侧车目录: 段 A 的文件软链到现役目录, 段 B 用 $2 目录的 amp(给 none = 不带段 B, 训练用 —— 训练器见挂着的低秩项直接拒)
    local dir=$1 b=$2 f; rm -rf "$dir"; mkdir -p "$dir"
    for f in "$ROOT/$ARM"/*; do case "$(basename "$f")" in amp_L*.bin|ampr_L*.bin|amp_prev_*|amp_v5_*) ;; *) ln -s "$f" "$dir/";; esac; done
    [ "$b" = none ] || for f in "$ROOT/$b"/amp_L*.bin "$ROOT/$b"/ampr_L*.bin; do [ -e "$f" ] && ln -s "$f" "$dir/"; done
}
judge(){   # $1 = 臂名, $2 = 侧车目录; rows → $D/rows_<臂>/ruler.rows, 学生 logits 判完就删(全词表 f32 7.9 GB)
    local name=$1 dir=$2 stu
    stu="gguf/v41judge/stu_code_$(basename "$RULER" .ids)_n${N}_eng_${GB}_amp_$(basename "$dir")"
    M "判 $name($dir)"
    bash gguf-tools/scripts/v41_judge.sh "$RULER" "$N" --hf "$HF" "engine:$GG:$dir" > "$D/judge_$name.txt" 2>&1 || { tail -5 "$D/judge_$name.txt"; return 1; }
    mkdir -p "$D/rows_$name"; cp "$stu.rows" "$D/rows_$name/ruler.rows"; rm -f "$stu.bin" "$stu".bin.*
    grep -E "Same top|Mean KLD|Σmin" "$D/judge_$name.txt" | head -3 | sed "s/^ */  $name /"
}
M "② 现役(b 段 $(md5sum "$ARM/amp_L00.bin" | cut -c1-12))在新尺上 + FP 锚(没有就现算, 一趟流式装原件)"
[ -s "$D/rows_live/ruler.rows" ] || judge live "$ARM" || die "现役判决失败"
REF="gguf/v41judge/teacher_code_$(basename "$RULER" .ids)_n${N}_$(basename "$HF").bin"; [ -s "$REF" ] || die "FP 锚 $REF 没落下"
M "③ 靶: FP 锚 → top-64 表(整串 + 短题那条前 $K 位: 训练器按整条序列哈希查表, 前缀查不到)"
"$PY" "$BT" anchor2ptf "$REF" "$RULER" "$D/teacher.ptf" 64 "$K" || die "转表失败"
M "④ 训: lr $LR / $EPOCHS 遍 / batch $BATCH / 秩 $RANK${FRZ:+(冻前 $FRZ)}, 起点现役段 B → $OUT"
rm -rf "$D/init" "$OUT"; mkdir -p "$D/init" "$OUT"; cp "$ARM"/amp_L*.bin "$D/init/"
for b in gguf/v41/posttrain/restore-evo-*/base.fnv; do [ -s "$b" ] && { cp "$b" "$D/init/"; break; }; done   # 段 A 指纹(同一轮链)
side "$D/side_noB" none
{ echo "# restore_B_ceiling_spark.sh: 段 B 天花板, 靶 = 编程尺自己的 FP 读数(只诊断, 不进现役)"
  echo "data=$D/rows.jsonl"; echo "out=$OUT"; echo "teacher=fp:$D/teacher.ptf"; echo "rb_init=$D/init"; echo "lr=$LR"; echo "rb_gap=0"; echo "rb_rest=0"
  printf '%s\n' layers=0-39 rank=$RANK epochs=$EPOCHS batch=$BATCH topk=64 maxlen=1024 expand=0 ctxlen=65536 doc_tail=0; } > "$D/${TAG}train.cfg"
if [ -n "$AUXW" ]; then
    if [ -s "$AX/y_L39.bin" ] && cmp -s "$AX/ids.txt" "$RULER"; then M "④a 逐层靶 $AX 已有(序列与尺逐位同), 复用"
    else
        M "④a 取料: 当前链(段 A + 现役段 B)把尺整条前向, 每层 x / 放大器前的 y → $AX"
        rm -rf "$AX"; { sed 's#^out=.*#out='"$D"'/dump_out#' "$D/${TAG}train.cfg"; echo "rb_dump=$AX"; } > "$D/dump.cfg"
        ./ds4 --cuda -m "$GG" --zchain "$D/side_noB" --mem-budget-mb 110000 --no-dspark --restore-b "$D/dump.cfg" > "$D/dump.log" 2>&1 || { tail -5 "$D/dump.log"; die "取料失败, 看 $D/dump.log"; }
        grep -a "取料\] [0-9]* 位 ×" "$D/dump.log"
        cmp -s "$AX/ids.txt" "$RULER" || die "取料的序列与尺 $RULER 不逐位同"
        M "④b FP 原件逐层 y_fp(x_q): 每层 MoE 入口钉成取料的 x(--force-x), 同目录落 y_Lnn.bin"
        "$PY" gguf-tools/scripts/v41_teacher.py "$HF" --ids "$RULER" --ntok "$N" --force-x "$AX" --dump-moe "$AX" > "$D/fx.log" 2>&1 &
        pid=$! low=0
        while kill -0 "$pid" 2>/dev/null; do   # 看门狗与 posttrain_fp_spark.sh ② 同一条线
            sleep 5; av=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
            if [ "$av" -lt 2500 ]; then low=$((low + 1)); [ "$low" -ge 2 ] && { M "★看门狗: MemAvailable $av MB 连续两次 < 2500, 杀 FP $pid★"; kill -9 "$pid"; }; else low=0; fi
        done
        wait "$pid"; rc=$?
        grep -E "反修原料|完成|Error|Traceback|Killed" "$D/fx.log" | tail -3
        [ "$rc" = 0 ] && [ -s "$AX/y_L39.bin" ] || die "FP 逐层 rc=$rc, 看 $D/fx.log"
    fi
    printf 'rb_aux=%s\nrb_auxw=%s\n' "$AX" "$AUXW" >> "$D/${TAG}train.cfg"
fi
[ -z "$RLR" ] || printf 'rb_route=1\nrb_route_lr=%s\n' "$RLR" >> "$D/${TAG}train.cfg"
[ "$WMASK" = 1 ] && echo "rb_mask=$D/mask.txt" >> "$D/${TAG}train.cfg"
[ -z "$FRZ" ] || echo "rb_freeze=$FRZ" >> "$D/${TAG}train.cfg"
./ds4 --cuda -m "$GG" --zchain "$D/side_noB" --mem-budget-mb 110000 --no-dspark --restore-b "$D/${TAG}train.cfg" > "$D/${TAG}train.log" 2>&1; rc=$?
grep -aE "restore-b\]|★" "$D/${TAG}train.log" | grep -v "取料\] [0-9]* / " | tail -$((EPOCHS * 2 + 8))
[ "$rc" = 0 ] || die "训练 rc=$rc, 看 $D/${TAG}train.log"
M "⑤ 判 ckpt: $JUDGE"
for e in $JUDGE; do
    ck="$OUT/ckpt_e$(printf %02d "$e")"; [ -s "$ck/amp_L00.bin" ] || die "$ck 不在"
    a="${TAG}e$(printf %02d "$e")"; side "$D/side_$a" "$ck"; judge "$a" "$D/side_$a" || die "判 $a 失败"
done
M "⑥ 对现役逐位配对: 练过的后半 / 没练过的前半"
for e in $JUDGE; do
    a="${TAG}e$(printf %02d "$e")"; c="e$(printf %02d "$e")"
    for half in train held; do
        if [ "$WMASK" = 1 ]; then hn=$([ $half = train ] && echo "练过的窗" || echo "留出的窗(同语言)"); else hn=$([ $half = train ] && echo "练过的后半" || echo "没练过的前半"); fi
        echo "== $a $hn"
        "$PY" gguf-tools/scripts/judge_rows_tokclass.py --hf "$HF" --paired "$D/$half.lst" "现役=$D/rows_live/{id}.rows" "$a=$D/rows_$a/{id}.rows" | grep -E "KLD|Same"
        # 带 --aux 时再对同遍数的天花板(只有出口 KL)配对: 这一行就是"逐层靶有没有用"的直接读数
        [ -n "$TAG" ] && [ -s "$D/rows_$c/ruler.rows" ] && { echo "   对天花板 $c(只有出口 KL):"
            "$PY" gguf-tools/scripts/judge_rows_tokclass.py --hf "$HF" --paired "$D/$half.lst" "天花板=$D/rows_$c/{id}.rows" "$a=$D/rows_$a/{id}.rows" | grep -E "KLD|Same"; }
    done
    "$PY" gguf-tools/scripts/judge_rows_tokclass.py --hf "$HF" --paired "$D/train.lst" --misses "现役=$D/rows_live/{id}.rows" "$a=$D/rows_$a/{id}.rows" | head -4
done
rm -rf "$D"/side_*
M "完成(现役侧车没动: $(md5sum "$ARM/amp_L00.bin" | cut -c1-12))"
