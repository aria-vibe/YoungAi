#!/bin/bash
# restore_sidecar_chain_spark.sh — 还原侧车两段一条龙(2026-10-24 用户"开始解侧车, 两段: 语料一段, 加上之后决策 think 一段", 方案 docs/restore_sidecar.md):
#   A  段 A 闭式: v41_sidecars_all_spark.sh --gguf <①> --hf <1017> restore
#      = amp_campaign.sh --profile restore sidecar: stage_idsmix 混合拟合份(编程 48 窗 / 图形 12 / 数学 12 / 金法医科各 9 / 通用 12)
#        → grrb 解 40 层 → 编程判决份主裁(裸 / 其他侧车 / 现役 ② 对照) → wt2 守门 → 其余四个业务域的尺(fin41 / law / med / sci; 业务只有五个域, 10-24 用户定)
#      产物 gguf/v41/<①名>-grrb-restore_fit_n15360-engine/(它自己停/装回模型)
#   B  段 B 蒸馏: posttrain_fp_spark.sh gguf-tools/data/posttrain/restore_B_1024.cfg <1017> <段 A 目录>(包在 with_model_stopped 里)
#      → gguf/v41/posttrain/restore-B-1024/ckpt_eNN(低秩 amp_Lnn.bin, base.fnv 绑段 A); 教师表从 10-22 那趟拷来复用
#   C  并包: 段 B 选中轮的 amp_Lnn.bin 拷进侧车目录根(没有决策位单独目录; 引擎挂 --zchain 自己读 amp) → 终判 编程判决份 + wt2 三臂: 裸 / 现役对(编程 ② .1018 + ③ formal2 e02) / 还原侧车
# 用法(spark ~/ds4-main, nohup): gguf-tools/scripts/restore_sidecar_chain_spark.sh [--from A|B|C] [--pick ckpt_eNN] [--cfg <段 B cfg>]
#   --pick: 段 B 选哪轮进终判(默认 = 工作台选轮规则同款: train.log 里留出损失最低的轮; 没有 train.log 就要显式给)
#   --cfg: 段 B 的配方(默认 restore_B_1024.cfg; 10-25 第二趟 restore_B2_4096.cfg = 料加设计推理 + 窗 4096), 产物目录从它的 out= 读
# 停车规则: 任一段非 0 停在那一段, 尾行 CHAIN_EXIT <rc>; 段 A 的 DIE 在 amp_campaign 里, 段 B 的在 posttrain_fp_spark 里, 这里只看 rc。
# 出错会怎样: 段 A 产物目录没有 "# 完成" ⇒ 段 B 不发; 段 B 的 ckpt 目录里没有 amp_L00.bin ⇒ 段 C 不发; 教师表没拷到 ⇒ posttrain_fp 自己重打(慢但对)。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
#   --judge-only(10-27): 只跑段 C 的判决, 不选轮不并包 —— 量"现在侧车目录里这套"的基线(发新一趟训练前, 先看现役在留出题上涨没涨)。
#   --heldout <目录>(10-27, 泛化留出尺): 目录里 list.txt(id 全长 生成段起点) + <id>.ids(think_trace_collect.py ids 产出); 每段过两臂,
#     末尾 judge_rows_tokclass.py --list 合成一张表。训练料里一道都没有的题: 这里涨 = 泛化, 只在鹈鹕涨 = 定向。缺省 gguf/go-onebit/code/svgheld, 没有 list.txt 就跳过。
#   --tmo-judge <分钟>(10-27 自进化一轮一小时): 段 C 判决的硬超时, 套在 with_model_stopped 里面(套外面会连包装层一起杀, 模型装不回来);
#     超时 = rc 124 停链报出来, 不重试 —— 判决超时说明哪一段比预想的长, 要查, 不是再等一遍。
#   --pelseg 0(10-27 自进化): 不判鹈鹕设计推理四段 —— 用户: 鹈鹕只是例子, 要的是整个模型能力的还原; 一轮里的细节尺改用编程判决份按 FP 置信度分桶。
FROM="A"; PICK=""; CFG="gguf-tools/data/posttrain/restore_B_1024.cfg"; JONLY=0; HO="gguf/go-onebit/code/svgheld"; TMOJ=""; PELSEG=1
while [ $# -gt 0 ]; do case "$1" in --from) FROM="${2:?}"; shift 2;; --pick) PICK="${2:?}"; shift 2;; --cfg) CFG="${2:?}"; shift 2;;
    --judge-only) JONLY=1; FROM=C; shift;; --heldout) HO="${2:?}"; shift 2;; --tmo-judge) TMOJ="${2:?}"; shift 2;; --pelseg) PELSEG="${2:?}"; shift 2;; *) echo "★不认识的参数 $1★"; exit 2;; esac; done
HF="hf/DeepSeek-V4.1-Flash-1017"
mapfile -t PK < gguf/serve_pick.txt; GG="${PK[0]#"$ROOT"/}"; DEP="${PK[1]:-}"; DEP="${DEP#"$ROOT"/}"
ARM="gguf/v41/$(basename "$GG" .gguf)-grrb-restore_fit_n15360-engine"
BOUT=$(grep '^out=' "$CFG" | cut -d= -f2- | tr -d ' ')
PREV_T="gguf/v41/posttrain/fp-formal2-A-B-M-L0-39-lr2e-4-e2-1022"      # 10-22 正式趟 2: 同一份料的教师表(teacher.part.*.ptf)在这
DEP_PT="$PREV_T/ckpt_e02"                                               # 现役对的 ③(judge.ids 84.77 那臂)
M(){ echo "[rchain $(date '+%m-%d %H:%M:%S')] $*"; }
die(){ M "★$*★"; echo "CHAIN_EXIT 1"; exit 1; }
stage_ge(){ local o="A B C"; local a=${o%%$1*}; local b=${o%%$FROM*}; [ ${#a} -ge ${#b} ]; }
for f in "$GG" "$HF/config.json" "$CFG"; do [ -e "$f" ] || die "缺 $f"; done

if stage_ge A; then
  M "段 A: v41_sidecars_all_spark.sh --gguf $GG --hf $HF restore"
  bash gguf-tools/scripts/v41_sidecars_all_spark.sh --gguf "$GG" --hf "$HF" restore || die "段 A 失败"
fi
grep -q "^# 完成" "$ARM/manifest.txt" 2>/dev/null || die "段 A 产物没有完成标记: $ARM"

if stage_ge B; then
  M "段 B: 蒸馏 → $BOUT(挂 $ARM)"
  mkdir -p "$BOUT"
  # ★id 文件和 .ptf 必须成对拷★(10-24 晚实撞: 只拷 .ptf, posttrain_fp 见不到 id 文件就把新 teacher.ids 重切成同名的份, 再按文件名"已有 .ptf 跳过"
  #   ⇒ 泄题审计剔了 47 条后序列整体前移, 份 002 起内容全错位, 训练器按内容配只对上 538/628 题停车)。成对拷了, posttrain_fp 才能按内容算出
  #   哪些序列真没打过(teacher.rest), 只补那些; 多出的旧序列训练器按内容配, 用不上就闲着。
  n=0; for t in "$PREV_T"/teacher.part.[0-9][0-9][0-9]; do [ -s "$t" ] && [ -s "$t.ptf" ] && [ ! -e "$BOUT/$(basename "$t")" ] && { cp "$t" "$t.ptf" "$BOUT/"; n=$((n+1)); }; done
  M "教师表从 $PREV_T 拷 $n 份(id+ptf 成对; 同料复用, 内容对不上的序列 posttrain_fp 自己补)"
  bash gguf-tools/scripts/with_model_stopped_spark.sh bash gguf-tools/scripts/posttrain_fp_spark.sh "$CFG" "$HF" "$ARM" || die "段 B 失败"
fi

if stage_ge C && [ "$JONLY" = 1 ]; then
  ls "$ARM"/amp_L*.bin >/dev/null 2>&1 || die "--judge-only: 侧车目录里没有 amp, 判的就不是现役"
  M "--judge-only: 不选轮不并包, 判侧车目录现状($(tail -n 1 "$ARM/manifest.txt" | cut -c1-120))"
elif stage_ge C; then
  if [ -z "$PICK" ]; then
    # 选轮 = 留出损失最低的轮(train.log 的 epoch 行; 格式与工作台 model_train_pick.c 读的同一份), 抓不到就要显式 --pick
    PICK=$(ls -d "$BOUT"/ckpt_e* 2>/dev/null | while read -r d; do e=$(basename "$d"); h=$(ls -t "$BOUT"/heldout_e"${e#ckpt_e}".tsv 2>/dev/null | head -1); [ -s "$h" ] && echo "$(awk -F'\t' '{s+=$4; n++} END{if(n) printf "%.6f", s/n}' "$h") $e"; done | sort -n | head -1 | awk '{print $2}')
    # 给了 eval=(泛化留出)时料不再留出, heldout_eNN.tsv 是空的(10-28 自进化第一轮): 按 train.log 每轮的 gen_kl 取最低; 只有一轮就是它
    [ -z "$PICK" ] && PICK=$(awk '/^epoch /{for(i=1;i<NF;i++) if($i=="gen_kl") printf "%s ckpt_e%02d\n", $(i+1), $2}' "$BOUT/train.log" 2>/dev/null | sort -n | head -1 | awk '{print $2}')
    [ -z "$PICK" ] && [ "$(ls -d "$BOUT"/ckpt_e* 2>/dev/null | wc -l)" = 1 ] && PICK=$(basename "$(ls -d "$BOUT"/ckpt_e*)")
  fi
  [ -n "$PICK" ] && [ -s "$BOUT/$PICK/amp_L00.bin" ] || die "段 C 选不出轮(--pick ckpt_eNN 显式给; 现有: $(ls "$BOUT" 2>/dev/null | tr '\n' ' '))"
  # ★并进侧车★(用户 10-24: 没有决策位单独目录, 两段是同一个产物): 选中轮的 amp_Lnn.bin 拷到侧车目录根, 引擎挂 --zchain 时自己读(core_v41_amp.c:292, 零改动)。
  # 目录里已有 amp = 上一次段 B 的, 先挪走再拷(不许叠加: 训练器不接按秩拼接的 ②, 叠了也是两份低秩互相打架)
  if ls "$ARM"/amp_L*.bin >/dev/null 2>&1; then old="$ARM/amp_prev_$(date +%m%d%H%M)"; mkdir -p "$old"; mv "$ARM"/amp_L*.bin "$old"/; M "侧车里已有 amp, 挪到 $old"; fi
  cp "$BOUT/$PICK"/amp_L*.bin "$ARM"/ || die "拷 amp 进侧车失败"
  printf '# 段 B(向原件蒸馏的决策位低秩项): %s/%s, 料 %s, %s\n' "$BOUT" "$PICK" "$(grep '^data=' "$CFG" | cut -d= -f2-)" "$(date '+%Y-%m-%d %H:%M')" >> "$ARM/manifest.txt"
  M "段 B 选中轮 $PICK 已并进 $ARM($(ls "$ARM"/amp_L*.bin | wc -l) 层 amp); manifest 追加一行"
fi
if stage_ge C; then
  HOL=""; [ -s "$HO/list.txt" ] && HOL="$(awk '{printf "%s:%s ", $1, $2}' "$HO/list.txt")"
  # 两臂(10-25 用户: 架构只有一个侧车, 换下的旧部署不是平行臂): 裸 / 还原侧车(整目录)。
  # 设计推理尺(10-25 上午定): 官方 pel1/pel2 写码前的设计推理接在本地第 2 步提示(6569 位)后, FP 锚已在 gguf/v41judge; 只看生成段(--from 6569)的 Same top / KLD,
  #   基线(段 B 第一趟的侧车)77.1 / 76.3 / 78.2 / —; 这是"path 少一半"的直接尺, 编程判决份与 wt2 守不退。
  M "段 C 终判: 编程判决份 + wt2 + 设计推理四段, 两臂: 裸 / 还原侧车(整目录)"
  J="gguf-tools/scripts/v41_judge.sh"; PEL="gguf/go-onebit/code/pelican"; TC="gguf-tools/scripts/judge_rows_tokclass.py"
  PSEG="think_off1_b2:10326 think_off1_b6:10926 think_off1_b5:16862 think_off2_b3:25475"; [ "$PELSEG" = 0 ] && PSEG=""
  TJ=(); [ -n "$TMOJ" ] && TJ=(timeout --kill-after=60 "${TMOJ}m")
  bash gguf-tools/scripts/with_model_stopped_spark.sh "${TJ[@]}" bash -c "
    bash $J gguf/go-onebit/code/judge.ids 15360 --hf $HF engine:$GG engine:$GG:$ARM &&
    bash $J gguf/go-onebit/g7/wt2.ids 512 --hf $HF engine:$GG engine:$GG:$ARM &&
    for s in $PSEG; do bash $J $PEL/\${s%%:*}.ids \${s##*:} --hf $HF engine:$GG engine:$GG:$ARM || exit 1; done &&
    for s in $HOL; do bash $J $HO/\${s%%:*}.ids \${s##*:} --hf $HF engine:$GG engine:$GG:$ARM || exit 1; done"; rc=$?
  [ "$rc" = 124 ] && die "段 C 判决超时(${TMOJ} 分钟)"; [ "$rc" = 0 ] || die "段 C 判决失败"
  M "段 C 读数(Same top / KLD, 裸 / 侧车):"; grep -h -E "Same top|Mean KLD" gguf/v41judge/log_stu_code_judge_n15360_eng*"$(basename "$GG" .gguf)"*.txt 2>/dev/null | tail -4
  GB="$(basename "$GG" .gguf)"; AB="$(basename "$ARM")"
  for s in $PSEG; do
    n="${s%%:*}"; k="${s##*:}"; TAG="pelican_${n}_n$k"
    M "设计推理 $n 生成段(6569 起):"
    # --conf(10-26 晚): 按 FP 置信度分桶 —— 细节错只看 p>0.9 / 0.5~0.9 两桶的"错 top-1", 总 Same top 被 FP 自己也多选的 p<0.5 桶撑着
    "$HOME/v41env/bin/python" "$TC" --hf "$HF" --ids "$PEL/$n.ids" --from 6569 --conf "裸=gguf/v41judge/stu_${TAG}_eng_${GB}.rows" "侧车=gguf/v41judge/stu_${TAG}_eng_${GB}_amp_${AB}.rows" 2>&1 | grep -E "^== |^  all |^  conf |^  digit "
  done
  if [ -n "$HOL" ]; then   # 泛化留出尺: 多段合计(数字位 = 比例/坐标, 鹈鹕"不像"就差在这)
    H="$(basename "$HO")"; M "泛化留出 $H($(wc -l < "$HO/list.txt") 题, 训练料里没有)生成段合计:"
    "$HOME/v41env/bin/python" "$TC" --hf "$HF" --list "$HO/list.txt" --ids-dir "$HO" \
      "裸=gguf/v41judge/stu_${H}_{id}_n{n}_eng_${GB}.rows" "侧车=gguf/v41judge/stu_${H}_{id}_n{n}_eng_${GB}_amp_${AB}.rows" 2>&1
  fi
fi
M "全链收工"; echo "CHAIN_EXIT 0"
