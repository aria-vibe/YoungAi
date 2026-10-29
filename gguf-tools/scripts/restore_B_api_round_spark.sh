#!/bin/bash
# restore_B_api_round_spark.sh — 段 B 自学习一轮(2026-10-28)。整条链里只有四样东西: 官方 API、量化底座、反修段 A、不断自学习的反修段 B
#   (用户: "所有判断由 DeepSeek 官方api判断, 这样才是自学习"; "里面只应该有官方api、量化基座、反修a段、不断自学习的反修b段")。
#   训练的靶 = 官方 API 逐位 top-20; 没有本地 FP 教师, HF 目录只读分词器(tokenizer.json)把官方文字切成 token, 不装原件权重。
#   门 = 原来的编程尺 + wt2 尺(用户 10-28: "指标还是用之前的编程和wt2指标, 不要再新建指标了"): 锚是 gguf/v41judge 里早就算好的参考前向
#   logits, 判一套侧车只跑引擎学生, 不碰原件。
#
# 用法(spark ~/ds4-main, 工作台 ds4-train :8000 开着):
#   nohup bash gguf-tools/scripts/restore_B_api_round_spark.sh --work gguf/go-onebit/evo/a002 > /tmp/bugmd/api_a002.log 2>&1 &
#   下一轮 --work …/a003: 起点自动是侧车里现在那套(上一轮留下的, 或回退后的), 一轮接一轮在前面的基础上学。
#   飞轮: 每轮 ⑥ 把结果交给官方 API 复盘, 它定下一轮出哪几种语言的题、几道、学习率(evo/plan_next.json); 下一轮 ① 拿它出题、③ 拿它的学习率。
#     没有复盘的那一轮(第一轮 / 删了 plan_next.json)按缺省: 4 道不限语言、lr 2e-5。
#     计划要落在时间预算里(--budget 分钟, 缺省 60 = 用户 10-27 "一小时一轮"): 复盘拿到本轮各段用时(<work>/times.tsv)、训练步数与每步秒数、
#     官方解答总 token, 由它估下一轮总分钟(est_min), 超预算就让它重答(10-28 用户: "训练时间在飞轮建议里面也要考虑进去")。
#   --q N / --lr X 显式给了就压过计划(做对照用); --init none 从零起; --beh 0 不跑评委; --from/--to N 分段(0~7); --tmo 段=分钟
#   --gap Q 缺口分位(缺省 0 = 每位都出梯度): Q > 0 时 ③ 只在第 0 步逐位 KL 落在全料 Q 分位以上的位出全权重, 其余位权重 --rest(./ds4 --restore-b, core_restoreb.c)
#     缺省为什么回到 0(10-28 a011 同料三档): 其余位权重 0 / 0.3 / 1 编程尺 +2.9 / +1.4 / −0.1 SE, 最难 3% 的位 −0.001 / −0.012 / −0.025 —— 不管其余位, 难位反而修得少
#     为什么(10-28 用户: "段 B 本质是量化能力还原, 要找原件能量的还原点, 不是把官方语料再学一遍"): a002~a008 每位都出梯度, 训练料降 9~17%, 编程尺却全位置均匀地退
#   --rest λ 缺口以外的位的权重(缺省 0 = 只练缺口位): 10-28 a010/a011 缺口位修好了, 其余 90% 的位没梯度钉着一起漂坏(编程尺 0~90% 分位每位 +0.002~+0.004), 合起来净退;
#     显式给 = 同料对照, 产物与账本带 -wλ: … --work …/a011 --from 3 --to 5 --rest 0.25
#   --raw 第一层"原始语料"(10-28 用户: "原始模型和后训练专项任务是两回事, 分两块还原, 先原始语料找回原始能力, 再任务能力"; 料源用官方 API 裸续写、只要代码):
#     ① 官方给 30 种语言各 --q 个源文件开头(api_round.py seeds) → beta/completions 当裸文档接着写 --raw-tok 位(collect --raw, 不套对话模板);
#     ② 行 = {prefix: 开头(只当上下文), text: 续写, eos: false}; ⑥ 官方复盘(api_round.py plan --layer raw)定下一轮的语言 / 每种几个开头 / 开头方向 / lr / 理由,
#        写 evo/plan_next_raw.json(与任务层的 plan_next.json 分开); 下一轮 --raw 照它跑(10-28 用户: 每轮的下一步由官方 API 裁决和建议) —— 两把尺本来就都是裸文档
#     为什么: 现役在官方的对话回答上对官方 KL 只有 0.044~0.075, 在裸源文件上对原件 0.249; 对话料练十一轮, 两把裸文档尺一次都没降过 2 SE
#     例: nohup bash … --raw --work gguf/go-onebit/evo/o001 --to 5 > /tmp/bugmd/api_o001.log 2>&1 &
#   --dec 第二层"决策位"(10-29 用户: 第一层裸语料推到编程 Same top 84.97 之后, 接着现役段 B 训第二阶段 —— 同样的编程题域, 料换成带决策位的形态:
#     思考 → </think> → 正文 → EOS; 尺还是编程 + wt2 两把, ★不设门只报数★(用户: "就算回退也不一定是坏事, 因为决策更好了"), 训完挂着, 接 dsh 鹈鹕看效果):
#     ① 官方出 --q 道编程题 → 两件事并行: (a) 官方【开思考】作答(逐位 top-20; </think> / EOS 两位 API 不给分布, 配表时 one-hot)
#        (b) 现役侧车自己想同一批题(服务开着, think_freegen_probe, 上限 32768)→ 每 8192 位切一个分支点, 官方从学生前缀接着想到收口(api_branch.py, 续 ≤32768)
#        为什么要 (b): 收不了口是曝光偏差 —— 学生走到两万位后进了官方没去过的区域, 官方自己的轨迹上没有一位教它回来; 10-26 v4b 加了分支续写,
#        学生自由生成 17/18 收口、鹈鹕 2/2 自收手(fable5 10-27)。学生前缀只当上下文零梯度(prefix 行), 靶全是官方的 ⇒ 不碰 09-30 "禁自吐当靶"
#     ② 料 = 官方整条轨迹一行(头进 prefix, 收口条 eos:true)+ 分支行; ③ 同第一层(--restore-b, ctxlen 65536 前缀形态, 尾窗 1024 反传);
#        train.log 多一行 "decision eNN kl … n … tp …" = 料里目标是 EOS/</think> 的位的 KL(core_restoreb.c rb_dec_log), 第二层动没动决策位看它
#     ⑤ 不回退: 新段 B 留在侧车上, 账本去留列记 keep(dec); 现役在 <work>/init, 要撤就 restore_init。⑥ 复盘本轮不跑(计划分层只有 raw / task)
#     例: nohup bash … --dec --q 4 --beh 0 --work gguf/go-onebit/evo/d001 --why "…" > /tmp/bugmd/api_d001.log 2>&1 &
#   --why "…" 本轮改了什么、为什么会变好(10-28 用户: "每次跑下一轮一定要有理由输出, 现在改了什么为什么会变好"): 发车第一行打出来, 存 <work>/why.txt、账本第 20 列;
#     没给就用任务层计划(plan_in.json)里官方复盘写的 why; 两个都没有不发车
#   训练参数(10-28 用户: "复盘不能仅限于训练语料, 训练参数也很重要, 都要复盘, 脚本里动态获取"): ⑥ 官方复盘连料带参数一起定, 写进计划的 train 对象
#     {lr, epochs 1~3, batch 1~8, maxlen 512/1024/2048, gap, rest, raw_tok(第一层)}; ③ 的训练配置全从这里读, 本轮实际值打在发车第二行、交给下一次复盘当历史。
#     --lr/--epochs/--batch/--maxlen/--gap/--rest/--raw-tok 显式给 = 同料对照, 压过计划(产物与账本另记)。秩、层、clip 不放给复盘(原因见 api_round.py TRAIN_FIXED)
#   --rank N 段 B 的秩(缺省 = 现役的秩)。比现役大 = 扩秩续训, 产物与账本带 -rN; 同一份料比秩: … --work …/a008 --from 3 --to 7 --rank 128
#   --lr X 学习率(缺省 2e-5)。显式给 --lr 时 ③ 的产物目录与账本轮名带 -lrX, 同一份料可以换学习率重训对照:
#     bash … --work gguf/go-onebit/evo/a001 --from 3 --to 6 --beh 0 --lr 2e-4
#     为什么缺省是 2e-5(10-28 a001 同料对照): 起点在官方轨迹上 KLD 只有 0.079(r001/r002 起点 0.26~0.29), 梯度范数只有那两轮的 1/3,
#     Adam 每步走的距离却只看 lr 不看梯度大小 ⇒ lr 2e-4 时 22 步把 amp 推走 9.9%, 训练料自己的 KLD 0.0787 → 0.1017、回退;
#     lr 2e-5 同一份料 0.0787 → 0.0652, 编程 KLD −3.6 SE、wt2 −2.2 SE, 留下。
#
# 一轮怎么走(任一段非 0 停, 尾行 APR_EXIT <rc>):
#   ⓪ 现役侧车在编程 / wt2 两把尺上的读数(按 amp 指纹缓存 gguf/go-onebit/evo/fprows/<指纹>/, 同一套只判一次)
#   ① 官方现出 N 道新编程题(与历来所有题去重, 禁词 evo_ban.txt)+ 官方关思考的解答(带逐位 top-20; 10-28 起, 之前三轮是思考轨迹)
#   ② 料: 收口的轨迹整条一行, 头(系统 effort + 题)进 prefix 只当上下文; 发车前先按训练器同款渲染配 API 表, 配不上的行剔掉
#   ③ 训: 侧车 amp 挪成 <work>/init 当起点; ./ds4 --restore-b(段 B 还原训练, core_restoreb.c; 后训练 ③ 的代码不动): 靶 = 官方 API top-20,
#      只在缺口位(第 0 步逐位 KL ≥ --gap 分位)出梯度; 教师表 = ② 配出来的那份(<work>/teacher.ptf), 配不上就停
#      训练料自己的 KLD 没降就不往下走(优化本身走坏了, 判不判都是回退)
#   ④ 最后一遍的 ckpt_eNN 并进侧车根; 停模型判两把尺(v41_judge.sh 引擎学生, 学生 logits 判完一份删一份)
#   ⑤ 门(新侧车对现役, 同位配对, 2 个标准误):
#        编程提升 = 编程判决份【门半边】Mean KLD 降过 2 SE(256 位一块估 SE); Same top 否决闸(照反修 A 段): 新中旧不中 − 旧中新不中 < −2√不一致行数 就不留;
#        ★门半边 / 报数半边(10-29)★: 判决份每种语言 4 个 128 位窗(judge.ids.layout), 第 1/3 窗进门(code_gate.lst), 第 2/4 窗只报数(code_report.lst)。
#          为什么: 门和成绩用同一把尺时, 同配置两趟就差 ~1.7 SE, 噪声里挑赢家会把尺上的分数往上抬, 报出来的进度里分不清哪些是挑出来的;
#          报数半边从不参与去留(⑥ 按语言拆的表也只用门半边), 它的读数才是这条链真攒下的。账本末尾三列 = 报数半边 ΔKLD / SE / Same 净行数。
#          10-29 回算 o001~o008(门当时是整把尺): dcb56 → b0817 门半边 KLD −7.7 SE、报数半边 −9.2 SE, 两半涨得一样多。
#        通用底线 = wt2 512 位 Same top ≥ 80%(10-28 用户: "wt2 保持在 80 以上就行了, 主要还是把编程推进"); wt2 的 KLD / Same 配对(32 位一块)只报; RMS Δp 报数并进"三项全优"判定; Σmin / PPL 只报; 评委(--beh 1, 缺省开, 只报): 留下的新侧车把 1 道编程留出题自己写一遍, 官方评委打分(回退的不评, 省 20 分钟)。
#      过 = 留(新侧车成为现役); 不过 = 侧车换回 init 重装(新 amp 留在 ckpt 目录)。结果追加 gguf/go-onebit/evo/ledger_api.tsv
#   ⑥ 官方复盘: 编程尺按 30 种语言拆开(同一份 rows, 每种 512 位, 配对 32 位一块), 连同门、训练料 KLD、本轮题目、以前各轮"计划 → 结果"、评委评语
#      交给官方 API(api_round.py plan), 它出下一轮计划 → <work>/plan_out.json, 抄成 evo/plan_next.json; 本轮事实 → <work>/facts.json(以后的复盘当历史读)
#   ⑦ 清理: 本轮训练教师表
# 出错会怎样: 训练/判决半路失败 → 侧车换回 init 重装, 现役不受影响; 超时 = rc 124 停车报出来, 不重试(超时说明哪段比预想的长, 要查)。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
FROM=0; TO=7; WORK=""; Q=""; INIT=cur; BEH=1; LR=""; JMODE=""; BUDGET=60; RANK=""; GAP=""; REST=""; RAWM=0; DECM=0; RAWTOK=""; WHYRUN=""; EPOCHS=""; BATCH=""; MAXLEN=""
# student 30 = 4 题 × 32768 位上限 @45 t/s 两路; branch = 官方续 ≤32768 位 @170 tok/s 六路
declare -A TMO=([q]=5 [collect]=25 [student]=30 [branch]=15 [train]="" [judge]=30 [beh]=20 [review]=10 [plan]=15)
while [ $# -gt 0 ]; do case "$1" in
    --work) WORK="${2:?}"; shift 2;; --budget) BUDGET="${2:?}"; shift 2;; --rank) RANK="${2:?}"; shift 2;; --gap) GAP="${2:?}"; shift 2;; --rest) REST="${2:?}"; shift 2;; --raw) RAWM=1; shift;; --dec) DECM=1; shift;; --why) WHYRUN="${2:?}"; shift 2;; --raw-tok) RAWTOK="${2:?}"; shift 2;; --epochs) EPOCHS="${2:?}"; shift 2;; --batch) BATCH="${2:?}"; shift 2;; --maxlen) MAXLEN="${2:?}"; shift 2;; --q) Q="${2:?}"; shift 2;; --init) INIT="${2:?}"; shift 2;; --beh) BEH="${2:?}"; shift 2;;
    --from) FROM="${2:?}"; shift 2;; --to) TO="${2:?}"; shift 2;; --lr) LR="${2:?}"; shift 2;; --tmo) TMO[${2%%=*}]="${2#*=}"; shift 2;;
    --_judge) JMODE="${2:?}"; shift 2;;   # 内部: 停模型包装里回调自己, 把现在侧车判一遍存到这个目录
    *) echo "★不认识的参数 $1★"; exit 2;; esac; done
[ "$RAWM$DECM" != 11 ] || { echo "★--raw 与 --dec 是两层, 一轮只跑一层★"; exit 2; }
PY="$HOME/v41env/bin/python"; HF=hf/DeepSeek-V4.1-Flash-1017; PD=gguf-tools/data/posttrain; KEY=$PD/deepseek_api.key
EVO=gguf/go-onebit/evo; HOLD=$EVO/hold; FR=$EVO/fprows; W=gguf-tools/scripts/with_model_stopped_spark.sh
TTC=gguf-tools/scripts/think_trace_collect.py; APT=gguf-tools/scripts/api_teacher.py
CODE=gguf/go-onebit/code/judge.ids; WT2=gguf/go-onebit/g7/wt2.ids   # 两把尺, 锚都在 gguf/v41judge
mapfile -t PKS < gguf/serve_pick.txt; GG="${PKS[0]#"$ROOT"/}"; ARM="${PKS[1]#"$ROOT"/}"; GB=$(basename "$GG" .gguf); AB=$(basename "$ARM")
M(){ echo "[apr $(date '+%m-%d %H:%M:%S')] $*"; }
die(){ M "★$*★"; echo "APR_EXIT 1"; exit 1; }
stage(){ [ "$FROM" -le "$1" ] && [ "$TO" -ge "$1" ]; }
T(){ local k=$1; shift; timeout --kill-after=60 "${TMO[$k]}m" "$@"; local rc=$?; [ "$rc" = 124 ] && M "★$k 段超时(${TMO[$k]} 分钟)★"; return "$rc"; }
ampmd5(){ md5sum "$ARM/amp_L00.bin" 2>/dev/null | cut -c1-12; }
# 各段起点记进 <work>/times.tsv(复盘按它算各段分钟); 同一段只记第一次 —— 分段重跑不把两次之间的空档算进上一段
tick(){ [ -n "$WORK" ] && ! grep -q "^$1 " "$WORK/times.tsv" 2>/dev/null && echo "$1 $(date +%s)" >> "$WORK/times.tsv"; }

# ---- 判一套侧车(只在停模型包装里跑): 引擎挂整个侧车目录; rows 与五指标输出存进 $1, 学生 logits 本体判完就删(全词表 f32, 编程一份 7.9 GB) ----
if [ -n "$JMODE" ]; then
    mkdir -p "$JMODE"; rc=0
    one(){ local ids=$1 n=$2 name=$3 stu
        stu="gguf/v41judge/stu_$(basename "$(dirname "$ids")")_$(basename "$ids" .ids)_n${n}_eng_${GB}_amp_${AB}"
        bash gguf-tools/scripts/v41_judge.sh "$ids" "$n" --hf "$HF" "engine:$GG:$ARM" > "$JMODE/$name.judge.txt" 2>&1 && cp "$stu.rows" "$JMODE/$name.rows"; local r=$?
        rm -f "$stu.bin" "$stu".bin.*; grep -E "Same top|Mean KLD|RMS|Σmin|PPL\(student" "$JMODE/$name.judge.txt" | sed "s/^ */  $name /"; return "$r"; }
    one "$CODE" 15360 code || rc=1
    one "$WT2" 512 wt2 || rc=1
    exit "$rc"
fi

[ -n "$WORK" ] || { echo "★要 --work gguf/go-onebit/evo/aNNN★"; exit 2; }
for f in "$KEY" "$HF/tokenizer.json" "$GG" "$CODE" "$WT2" "$HOLD/beh_q.jsonl" "$HOLD/official.review.jsonl"; do [ -s "$f" ] || die "缺 $f"; done
for t in code_judge_n15360 g7_wt2_n512; do [ -s "gguf/v41judge/teacher_${t}_$(basename "$HF").bin" ] || die "缺判决锚 teacher_${t}_$(basename "$HF").bin"; done
grep -q "^# 完成" "$ARM/manifest.txt" 2>/dev/null || die "侧车 $ARM 没有完成标记(段 A)"
mkdir -p "$WORK" "$FR" "$EVO/beh"; RN=$((10#$(basename "$WORK" | tr -dc 0-9)))
TAG=$(basename "$WORK"); [ -n "$LR" ] && TAG="$TAG-lr$LR"; [ -n "$GAP" ] && TAG="$TAG-g$GAP"; [ -n "$REST" ] && TAG="$TAG-w$REST"; [ -n "$EPOCHS$BATCH$MAXLEN" ] && TAG="$TAG-e${EPOCHS:-p}b${BATCH:-p}m${MAXLEN:-p}"   # 显式给 --gap = 同料对照, 产物与账本另记   # 显式换学习率重训同一份料: 产物与账本分开记, 不覆盖按计划跑的那趟
# 秩: 不给就跟现役同秩(amp 头第 2 个 i32); 给了比现役大 = 扩秩续训(训练器前 K0 秩读现役, 多出的 A=0 起步), 产物与账本带 -rN
LIVEK=$(od -An -t d4 -j 4 -N 4 "$ARM/amp_L00.bin" 2>/dev/null | tr -d " "); [ -n "$LIVEK" ] || LIVEK=64
if [ -n "$RANK" ]; then [ "$RANK" -ge "$LIVEK" ] || die "--rank $RANK 比现役的秩 $LIVEK 小(训练器只接扩秩续训)"; [ "$RANK" = "$LIVEK" ] || TAG="$TAG-r$RANK"; else RANK=$LIVEK; fi
# 本轮按哪份计划跑: 第一次进 ① 时把上一轮复盘出的 evo/plan_next.json 抄成 <work>/plan_in.json, 之后分段重跑都读这一份(下一轮复盘改了 plan_next 也不串)
# 两层各一份计划(第一层 = plan_next_raw.json, 任务层 = plan_next.json), 互不串
PNEXT=$EVO/plan_next$([ "$RAWM" = 1 ] && echo _raw; [ "$DECM" = 1 ] && echo _dec).json   # 第二层还没有复盘, plan_next_dec.json 不存在 = 不读任务层的旧计划
PIN=$WORK/plan_in.json; { [ "$FROM" -le 1 ] && [ ! -s "$PIN" ] && [ -s "$PNEXT" ]; } && cp "$PNEXT" "$PIN"
PLANG=(); PBRIEF=""
if [ -s "$PIN" ]; then
    mapfile -t PLANG < <("$PY" -c "import json,sys; print(*json.load(open(sys.argv[1]))['langs'], sep='\n')" "$PIN")
    PBRIEF=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['brief'])" "$PIN")
    [ -n "$Q" ] || Q=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['q'])" "$PIN")
    # 训练参数(10-28 用户: "训练参数也要复盘, 脚本里动态获取"): 计划的 train 对象; 老计划只有顶层 lr。命令行显式给的优先(同料对照)
    read -r P_LR P_EP P_BA P_ML P_GA P_RE P_RT < <("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); t=d.get('train') or {}; t.setdefault('lr', d.get('lr'))
print(*[t.get(k) if t.get(k) is not None else '-' for k in ('lr', 'epochs', 'batch', 'maxlen', 'gap', 'rest', 'raw_tok')])" "$PIN")
    pv(){ [ "$1" = - ] || echo "$1"; }
    [ -n "$LR" ] || LR=$(pv "$P_LR"); [ -n "$EPOCHS" ] || EPOCHS=$(pv "$P_EP"); [ -n "$BATCH" ] || BATCH=$(pv "$P_BA"); [ -n "$MAXLEN" ] || MAXLEN=$(pv "$P_ML")
    [ -n "$GAP" ] || GAP=$(pv "$P_GA"); [ -n "$REST" ] || REST=$(pv "$P_RE"); [ -n "$RAWTOK" ] || RAWTOK=$(pv "$P_RT")
fi
# 本轮的理由: 显式 --why 优先, 否则计划里的 why; 分段重跑沿用第一次记下的 why.txt
[ -n "$WHYRUN" ] || WHYRUN=$(cat "$WORK/why.txt" 2>/dev/null)
[ -n "$WHYRUN" ] || { [ -s "$PIN" ] && WHYRUN=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('why', ''))" "$PIN"); }
[ -n "$WHYRUN" ] || { echo "★要 --why \"本轮改了什么、为什么会变好\"(用户 10-28: 每轮一定要有理由)★"; exit 2; }
[ -s "$WORK/why.txt" ] || printf '%s\n' "$WHYRUN" > "$WORK/why.txt"
M "本轮理由: $WHYRUN"
[ -n "$Q" ] || Q=$([ "$RAWM" = 1 ] && echo 1 || echo 4); [ -n "$LR" ] || LR=2e-5
# 没计划也没显式给 = 历来的值(10-28 前一直写死在 ③ 的配置里)
[ -n "$EPOCHS" ] || EPOCHS=1; [ -n "$BATCH" ] || BATCH=4; [ -n "$MAXLEN" ] || MAXLEN=1024; [ -n "$GAP" ] || GAP=0; [ -n "$REST" ] || REST=0; [ -n "$RAWTOK" ] || RAWTOK=1536
CK=ckpt_e$(printf %02d "$EPOCHS")   # 最后一遍的检查点(③ 每遍落一个)
PARAMSJ="{\"lr\": $LR, \"epochs\": $EPOCHS, \"batch\": $BATCH, \"maxlen\": $MAXLEN, \"gap\": $GAP, \"rest\": $REST$([ "$RAWM" = 1 ] && echo ", \"raw_tok\": $RAWTOK")}"
M "本轮训练参数: $PARAMSJ"
# 训练时长跟着题数走: 4 道 ≈ 22 步 × 26 s + 装载与首末评估 ≈ 16 分钟; 留一倍余量。超时 = 查哪段变慢, 不是加时
[ -n "${TMO[train]}" ] || TMO[train]=$((30 + 8 * Q * (RAWM ? 8 : DECM ? 4 : 1) * EPOCHS))   # 裸续写: 每种语言 Q 个 × 30 种 ≈ 8 道题的料; 决策位: 思考轨迹 + 分支 ≈ 4 倍正文料
OUT=gguf/v41/posttrain/restore-api-$TAG; DATA=$WORK/rows.jsonl; RAW=$WORK/api_raw.jsonl; QJ=$WORK/q.jsonl
STU=$WORK/student.jsonl; BRRAW=$WORK/branch_raw.jsonl; BRROW=$WORK/branch_rows.jsonl; LPRAW=$WORK/lp_raw.jsonl   # --dec: 学生轨迹 / 分支续写原料 / 分支训练行 / 配表用的教师原料(官方轨迹 + 分支)
printf 'code 15360 0:15360\n' > "$FR/code.lst"; printf 'wt2 512 0:512\n' > "$FR/wt2.lst"
# 门半边 / 报数半边(见头注释 ⑤): 按 layout 的窗交替分, 每种语言第 1/3 窗进门、第 2/4 窗只报数
grep -v '^#' gguf/go-onebit/code/judge.ids.layout | awk -v o="$FR" '$1 == "win" {W = $2; next} NF == 3 {for (k = 0; k < int($3 / W); k++) {s = $2 + k * W; r = s ":" s + W; if (k % 2 == 0) g = g (g ? "," : "") r; else h = h (h ? "," : "") r}}
    END {print "code 15360 " g > (o "/code_gate.lst"); print "code 15360 " h > (o "/code_report.lst")}'
[ -s "$FR/code_gate.lst" ] && [ -s "$FR/code_report.lst" ] || die "门/报数半边的窗表没写出来"
BEHQ=$EVO/beh/beh_q_code.jsonl; grep '"evohold_code_00"' "$HOLD/beh_q.jsonl" > "$BEHQ"
mapfile -t BAN < <(grep -v '^#' "$PD/evo_ban.txt" | grep -v '^$')
# 空禁词表 = grep -v -E "" 把料整份剔光、出题也不拦泄题(10-27 zsh 下 mapfile 不在、数组展开成空串实撞过一次)
[ "${#BAN[@]}" -gt 0 ] || die "禁词表 $PD/evo_ban.txt 是空的或不在"
waitidle(){ for _ in $(seq 120); do curl -s --noproxy '*' --max-time 5 http://127.0.0.1:8001/metrics | grep -q '"state":"idle"' && return 0; sleep 5; done; return 1; }
# 就绪 = idle 且最新 serve 日志有"冒烟 过"(10-28 实撞: 子进程一监听就 idle, 抢在冒烟前发请求, 冒烟排队超时、子进程被杀)
waitready(){ local f; waitidle || return 1
    for _ in $(seq 120); do f=$(ls -t gguf/v41/posttrain/ui_logs/serve_*.log 2>/dev/null | head -1)
        grep -q "冒烟 过" "$f" 2>/dev/null && return 0; grep -q "冒烟 ★失败★\|★没起来\|★余量" "$f" 2>/dev/null && { M "★模型子进程没起来: $(grep -h "★" "$f" | tail -1)★"; return 1; }; sleep 5; done; return 1; }
# 段 B 换回 init(现役)并装回模型: 模型在跑 = 停了按新侧车重装; 不在(包着训练/判尺的那层被杀了, 没机会装回) = 只装
restore_init(){ rm -f "$ARM"/amp_L*.bin; cp "$WORK/init"/amp_L*.bin "$ARM/" || return 1
    if pgrep -x ds4-server >/dev/null; then bash "$W" true; else bash "$W" --load; fi > "$WORK/reload.log" 2>&1; }
# 退出保护(10-28: ③ 中途杀脚本两次, 现役 amp 留在 init 里没拷回、模型也没装回, 都是手救的): ③ 把现役挪进 init 起到 ⑤ 定下去留为止挂着;
#   INT/TERM → 杀全部子孙(训练 / 判尺 / 停模型包装) → init 拷回 → 装回 → 核指纹。训练与判尺都是 "后台 + wait" 跑的:
#   前台命令跑着时 bash 要等它退出才处理信号, 保护就等于没有
desc(){ local c; for c in $(pgrep -P "$1"); do desc "$c"; echo "$c"; done; }
guard_on(){ trap 'M "★被打断: 杀子进程, 段 B 换回 init($BFP)★"; kill $(desc $$) 2>/dev/null; wait; restore_init; M "指纹 $(ampmd5)(应为 $BFP)"; echo "APR_EXIT 130"; exit 130' INT TERM; }
# 行 → 训练器同款序列 → 配 API 表; 配上的行号写 $3(api_teacher.py ptf 的覆盖清单按序列内容, 这里换回行号)
match_api(){ local rows=$1 raw=$2 keep=$3
    "$PY" "$APT" simids --hf "$HF" --rows "$rows" --out "$rows.seqs" >/dev/null || return 1
    "$PY" "$APT" ptf --ids "$rows.seqs" --raw "$raw" --hf "$HF" --out "$rows.ptf" --covered "$rows.covered" --topk 20 | sed 's/^/    /' || return 1
    awk 'NR==FNR{c[$0]=1; next} ($0 in c){print FNR}' "$rows.covered" "$rows.seqs" > "$keep"; }
judge_now(){   # 停模型判现在这套侧车 → fprows/<指纹>/; 超时套在包装里面(套外面会连包装层一起杀, 模型装不回来)
    local fp; fp=$(ampmd5); [ -n "$fp" ] || { M "★侧车里没有 amp★"; return 1; }
    M "判侧车 $fp: 编程判决份 15360 位 / wt2 512 位 → $FR/$fp"
    "$W" timeout --kill-after=60 "${TMO[judge]}m" bash "$0" --_judge "$FR/$fp.tmp" > "$WORK/judge_$fp.log" 2>&1 & wait $!; local rc=$?
    grep -E "^  (code|wt2) " "$WORK/judge_$fp.log"
    [ "$rc" = 0 ] || { M "★判决 rc=$rc(124 = 超时), 看 $WORK/judge_$fp.log★"; return 1; }
    rm -rf "${FR:?}/$fp"; mv "$FR/$fp.tmp" "$FR/$fp"; waitready
}
beh_gen(){   # 评委(只报): 现在这套侧车把编程留出题自己写一遍, 官方评委打分; 已写过/评过的跳过(按 id / 正文 md5)
    local o="$EVO/beh/$(ampmd5)"
    T beh "$PY" gguf-tools/scripts/think_freegen_probe.py --questions "$BEHQ" --n 0 --jobs 1 --max-tokens 65536 --timeout 1200 --out "$o.jsonl" || return 1
    T review "$PY" gguf-tools/scripts/api_round.py review --key "$KEY" --raw "$o.jsonl" --out "$o.review.jsonl" --jobs 4 || return 1
}

if stage 0; then   # ⓪ 现役读数
    tick 判现役与评委
    ampmd5 > "$WORK/base.fp"; BFP=$(cat "$WORK/base.fp"); [ -n "$BFP" ] || die "侧车里没有 amp, 没有现役段 B 可比"
    if [ -s "$FR/$BFP/code.rows" ] && [ -s "$FR/$BFP/wt2.rows" ]; then M "⓪ 现役 $BFP 的读数已在 $FR/$BFP, 跳过"
    else M "⓪ 现役 $BFP 判一遍"; judge_now || die "现役判决失败"; fi
    if [ "$BEH" = 1 ]; then waitready || die "模型没就绪"; beh_gen > "$WORK/beh_base.log" 2>&1 || die "现役评委失败(看 $WORK/beh_base.log)"; fi
fi
BFP=$(cat "$WORK/base.fp" 2>/dev/null)
if stage 1; then   # ① 官方现出新编程题 + 官方轨迹
    tick 出题采轨迹
    SEEN=("$PD"/mix_q_*.txt "$PD"/hold_q_*.txt "$HOLD/q.jsonl"); for f in "$EVO"/[rfad][0-9]*/q.jsonl; do [ -s "$f" ] && [ "$f" != "$QJ" ] && SEEN+=("$f"); done
    if [ "$RAWM" = 1 ]; then
        RL=("${PLANG[@]}"); [ "${#RL[@]}" -gt 0 ] || mapfile -t RL < <(grep -v '^#' gguf/go-onebit/code/judge.ids.layout | awk 'NF == 3 {print $1}')   # 计划定的语言; 没计划 = 30 种全上
        if [ -s "$QJ" ]; then M "① 开头已在($(wc -l < "$QJ") 个), 沿用"
        else M "① 官方出 ${#RL[@]} 种语言各 $Q 个源文件开头(裸续写的种子) → $QJ"
            T q "$PY" gguf-tools/scripts/api_round.py seeds --key "$KEY" --langs "${RL[@]}" --n "$Q" --prefix "raw$(printf %03d "$RN")" --out "$QJ" --ban "${BAN[@]}" --brief "$PBRIEF" || die "出开头失败"; fi
        M "① 官方裸续写(beta/completions, 不套对话模板, 每个开头接 $RAWTOK 位, 带逐位 top-20, 8 路) → $RAW"
        T collect "$PY" "$APT" collect --key "$KEY" --questions "$QJ" --out "$RAW" --hf "$HF" --max-tokens "$RAWTOK" --jobs 8 --timeout 1200 --raw || die "采集失败"
    elif [ -s "$QJ" ]; then M "① 题集已在($(wc -l < "$QJ") 道), 沿用"
    else M "① 官方现出 $Q 道新编程题(与历来 ${#SEEN[@]} 份题集去重, 禁词 ${#BAN[@]} 个; $([ -s "$PIN" ] && echo "按计划 $PIN: ${PLANG[*]}" || echo "没有计划, 不限语言")) → $QJ"
        STEER=(); [ "${#PLANG[@]}" -gt 0 ] && STEER=(--langs "${PLANG[@]}" --brief "$PBRIEF")
        T q "$PY" gguf-tools/scripts/api_round.py questions --key "$KEY" --round "$RN" --prefix "apr$(printf %03d "$RN")" --families code --n "$Q" --out "$QJ" --seen "${SEEN[@]}" --ban "${BAN[@]}" "${STEER[@]}" || die "出题失败"; fi
    if [ "$DECM" = 1 ]; then   # 第二层: 官方开思考作答(后台, 只打 API)与学生自己想同一批题(占模型服务)并行, 再从学生前缀分支续写
        M "① 官方解答(开思考, 带逐位 top-20, 8 路, 上限 65536; 后台) → $RAW"
        T collect "$PY" "$APT" collect --key "$KEY" --questions "$QJ" --out "$RAW" --hf "$HF" --max-tokens 65536 --jobs 8 --timeout 1200 > "$WORK/collect.log" 2>&1 & CPID=$!
        waitready || die "模型服务没就绪(学生生成要现役侧车在线)"
        M "① 学生(现役 $BFP)自己想这 $(wc -l < "$QJ") 道题(思考档, 上限 32768, 2 路) → $STU"
        T student "$PY" gguf-tools/scripts/think_freegen_probe.py --questions "$QJ" --n 0 --jobs 2 --max-tokens 32768 --timeout 1200 --out "$STU" || die "学生生成失败"
        M "① 学生 $(wc -l < "$STU") 条, 自己收口 $("$PY" -c "import json,sys; print(sum(1 for l in open(sys.argv[1]) if json.loads(l).get('closed')))" "$STU") 条"
        M "① 分支续写: 学生思考每 8192 位一个分支点(没收口的末尾再加一个), 官方接着想 ≤32768 位, 6 路 → $BRRAW"
        T branch "$PY" gguf-tools/scripts/api_branch.py branch --key "$KEY" --student "$STU" --out "$BRRAW" --hf "$HF" --every 8192 --max-tokens 32768 --jobs 6 --timeout 900 || die "分支续写失败"
        M "① 分支 $("$PY" -c "import json,sys; R=[json.loads(l) for l in open(sys.argv[1])]; R=[r for r in R if r.get('ok')]; print(f'ok {len(R)} 条, 官方续到收口 {sum(1 for r in R if r.get(\"closed\"))} 条')" "$BRRAW")"
        "$PY" gguf-tools/scripts/api_branch.py rows --raw "$BRRAW" --out "$BRROW" || die "分支行失败"
        wait "$CPID" || die "官方开思考采集失败(看 $WORK/collect.log)"; tail -2 "$WORK/collect.log"
    elif [ "$RAWM" = 0 ]; then
    # 关思考(10-28 用户批): 思考模式的料正文只占 11~13%(a001~a003), 其余是思考段的文字, 编程尺量的却是原始代码 ⇒ 让官方直接写正文, 料几乎全是代码
    M "① 官方解答(关思考, 带逐位 top-20, 8 路, 上限 65536) → $RAW"
    T collect "$PY" "$APT" collect --key "$KEY" --questions "$QJ" --out "$RAW" --hf "$HF" --max-tokens 65536 --jobs 8 --timeout 1200 --no-think || die "采集失败"
    fi
fi
if stage 2; then   # ② 料
    tick 拼料
    "$PY" -c "import json,sys; [print(l, end='') for l in open(sys.argv[1]) if json.loads(l).get('ok')]" "$RAW" > "$WORK/api_ok.jsonl"
    if [ "$RAWM" = 1 ]; then   # 裸文档一份一行: 开头只当上下文, 续写是靶; 写到上限停的文档没有结尾, 不补 EOS
        "$PY" -c "import json,sys; [print(json.dumps({'prefix': r['q'], 'text': r['content'], 'eos': False}, ensure_ascii=False)) for r in map(json.loads, open(sys.argv[1]))]" "$WORK/api_ok.jsonl" > "$DATA.all" || die "料失败"
    else "$PY" "$TTC" chunks --hf "$HF" --raw "$WORK/api_ok.jsonl" --out "$DATA.all" --max-tok 1000000 --max-chunks 1 --head-prefix >/dev/null || die "料失败"; fi
    TRAW=$RAW
    if [ "$DECM" = 1 ]; then   # 第二层: 官方整条轨迹行 + 分支行; 教师原料两份拼一份(ptf 按内容找块, 分支的前缀位 one-hot 占位、训练器不算它)
        [ -s "$BRROW" ] || die "分支行 $BRROW 是空的"
        M "② 料: 官方轨迹 $(wc -l < "$DATA.all") 行 + 分支 $(wc -l < "$BRROW") 行"
        cat "$BRROW" >> "$DATA.all"; cat "$RAW" "$BRRAW" > "$LPRAW"; TRAW=$LPRAW
    fi
    BANRE=$(printf '%s\n' "${BAN[@]}" | paste -sd'|')   # 泄题门: 出题时按禁词丢过题, 这里拦轨迹里自己冒出来的 —— 整行剔掉
    grep -v -i -E "$BANRE" "$DATA.all" > "$DATA.ban"
    match_api "$DATA.ban" "$TRAW" "$WORK/keep.txt" || die "料配 API 表失败"
    awk 'NR==FNR{k[$1]=1; next} (FNR in k)' "$WORK/keep.txt" "$DATA.ban" > "$DATA"
    M "② 料 $(wc -l < "$DATA") 行(官方收口 $(wc -l < "$WORK/api_ok.jsonl") / 采 $(wc -l < "$RAW"); 禁词剔 $(( $(wc -l < "$DATA.all") - $(wc -l < "$DATA.ban") )), 配不上 API 表剔 $(( $(wc -l < "$DATA.ban") - $(wc -l < "$DATA") )))"
    mv "$DATA.ban.ptf" "$WORK/teacher.ptf" || die "官方教师表没落下来"   # ③ 直接吃这份(训练器同款渲染配出来的, 配上的行都在里面)
    rm -f "$DATA".all "$DATA".ban "$DATA".ban.*; [ -s "$DATA" ] || die "料是空的"
fi
if stage 3; then   # ③ 训: 靶 = 官方 API top-20, 起点 = 侧车里现在那套
    tick 训练
    [ "$(ampmd5)" = "$BFP" ] || die "侧车 amp 指纹变了(⓪ 记 $BFP, 现在 $(ampmd5)), 有别人动过侧车"
    # 训练器见挂着的侧车带低秩项直接拒(不接按秩拼接, 10-25 实撞) ⇒ 侧车里的 amp 挪成 init(续训起点; 回退也从这里拷回)
    rm -rf "$WORK/init"; mkdir -p "$WORK/init"; mv "$ARM"/amp_L*.bin "$WORK/init/" || die "挪 amp 失败"; guard_on
    for b in gguf/v41/posttrain/restore-evo-*/base.fnv; do [ -s "$b" ] && { cp "$b" "$WORK/init/"; break; }; done   # base.fnv = 段 A 指纹, 段 A 没变
    rm -rf "$OUT"; mkdir -p "$OUT"
    {   # 配方只有这一份, 写在这里(data/ 不进仓库, 配方得跟着脚本走); topk=20 = API 给的就是 20 个
        echo "# restore_B_api_round_spark.sh $TAG: 段 B 还原训练(./ds4 --restore-b, core_restoreb.c): 靶 = 官方 API top-20, 只在缺口位出梯度"
        echo "data=$DATA"; echo "out=$OUT"; echo "teacher=fp:$WORK/teacher.ptf"; [ "$INIT" = none ] || echo "rb_init=$WORK/init"; echo "lr=$LR"; echo "rb_gap=$GAP"; echo "rb_rest=$REST"
        printf '%s\n' layers=0-39 rank=$RANK epochs=$EPOCHS batch=$BATCH topk=20 maxlen=$MAXLEN expand=0 ctxlen=65536 doc_tail=0
    } > "$WORK/train.cfg"
    M "③ 训(段 B 还原, $PARAMSJ, 秩 $RANK): $WORK/train.cfg → $OUT(起点 $([ "$INIT" = none ] && echo "零" || echo "现役 $BFP"); 料 $(wc -l < "$DATA") 行; 停模型)"
    # 训练子进程的旗照 posttrain_fp_spark.sh(预算 110000 = 机器空着整机给它; --no-dspark 不建草稿器)
    "$W" timeout --kill-after=60 "${TMO[train]}m" ./ds4 --cuda -m "$GG" --zchain "$ARM" --mem-budget-mb 110000 --no-dspark --restore-b "$WORK/train.cfg" > "$WORK/train_$TAG.log" 2>&1 & wait $!; rc=$?
    grep -aE "restore-b|★|教师表|^ds4: --restore-b" "$WORK/train_$TAG.log" | tail -12
    if [ "$rc" != 0 ] || [ ! -s "$OUT/$CK/amp_L00.bin" ]; then restore_init; die "训练 rc=$rc(124 = 超时; 看 $WORK/train_$TAG.log), 侧车已换回 init 重装"; fi
    # 训练料自己的 KLD 都没降 = 这趟优化本身走坏了(过拟合只会让它更低), 不用再判; 10-28 a001 就是这样 0.0787 → 0.1017
    read -r K0 K1 < <(awk '/^step 0 /{for(i=1;i<NF;i++) if($i=="train_kl") a=$(i+1)} /^epoch /{for(i=1;i<NF;i++) if($i=="train_kl") b=$(i+1)} END{print a, b}' "$OUT/train.log")
    M "③ 训练料 KLD $K0 → $K1"
    awk -v a="$K0" -v b="$K1" 'BEGIN{exit !(a != "" && b != "" && b < a)}' || { restore_init; die "训练料 KLD 没降($K0 → $K1), 不并进; 侧车已换回 init 重装"; }
fi
if stage 4; then   # ④ 并进 + 判
    [ -s "$WORK/init/amp_L00.bin" ] && guard_on   # 分段从 ④ 起跑也挂保护: 侧车里是待判的新段 B, 现役在 init
    tick 判新侧车
    ls "$ARM"/amp_L*.bin >/dev/null 2>&1 && die "侧车里已经有 amp($(ampmd5)), 不是 ③ 刚训完的状态"
    cp "$OUT/$CK"/amp_L*.bin "$ARM/" || die "拷 amp 进侧车失败"
    printf '# 段 B(自学习, 靶 = 官方 API): %s/%s, 料 %s, %s\n' "$OUT" "$CK" "$DATA" "$(date '+%Y-%m-%d %H:%M')" >> "$ARM/manifest.txt"
    M "④ 新侧车 $(ampmd5) 并进 $ARM"; ampmd5 > "$WORK/new.fp"   # 回退后侧车里就不是它了, ⑥ 复盘按这个指纹找它的 rows
    judge_now || { restore_init; die "新侧车判决失败, 侧车已换回 init 重装"; }
fi
NFP=$(cat "$WORK/new.fp" 2>/dev/null)
# 配对输出先收进变量再打到 stderr。别写 tee /dev/stderr(10-28 实撞): Linux 上那是按路径重开日志文件且截断,
# o001~o008 的 /tmp 日志每过一次 ⑤ 就被清掉前半截, 编程尺的配对行被 wt2 那次覆盖, 只剩监控实时抓到的
pj(){ local l=$1 o; shift; o=$("$PY" gguf-tools/scripts/judge_rows_tokclass.py --hf "$HF" --paired "$l" "$@" "现役=$FR/$BFP/{id}.rows" "新=$FR/$NFP/{id}.rows"); printf '%s\n' "$o" >&2; grep '^PAIRED_JSON' <<<"$o" | cut -d' ' -f2-; }
if stage 5; then   # ⑤ 门
    [ -s "$WORK/init/amp_L00.bin" ] && guard_on
    tick 门与评委
    [ -n "$NFP" ] && [ "$NFP" != "$BFP" ] && [ "$(ampmd5)" = "$NFP" ] || die "侧车指纹 '$(ampmd5)' 不是 ④ 并进的新侧车 '$NFP'(现役 $BFP)"
    M "⑤ 编程(要提升): 编程判决份门半边(每种语言第 1/3 窗)逐位配对"; CJ=$(pj "$FR/code_gate.lst") || die "编程配对失败"
    M "⑤ 编程报数半边(第 2/4 窗, 不参与去留, 这条链真攒下的看它)"; RJ=$(pj "$FR/code_report.lst") || die "报数半边配对失败"
    M "⑤ 通用(不许退): wt2 512 位逐位配对"; WJ=$(pj "$FR/wt2.lst" --block 32) || die "wt2 配对失败"
    M "⑤ 五指标(现役 → 新):"
    for k in code wt2; do for f in "$BFP" "$NFP"; do echo "  $k $f: $(grep -E "Same top|Mean KLD|RMS|Σmin|PPL\(student" "$FR/$f/$k.judge.txt" | tail -5 | sed 's/  */ /g' | paste -sd'|')"; done; done
    # 门照反修 A 段的端到端择优(10-28 用户: "kld sametop 还有 rmsp 都看, 跟反修 a 段一样"; v41_e2e_select.inc.c):
    #   编程: KLD 主裁(降过 2 SE); Same top 否决闸: 新中旧不中 − 旧中新不中 < −2√不一致行数 就不留(防 KLD 降了 top-1 坏了);
#   wt2: 只守 Same top ≥ 80%(用户 10-28 晚改), KLD 与 Same 的配对变化只报;
    #   RMS Δp 与 A 段一样不单独否决(没有逐位配对, 只有两份汇总数), 进"三项全优"的判定一起报。
    rmsv(){ grep -o "RMS Δp(top32窗) = [0-9.]*" "$FR/$1/$2.judge.txt" | tail -1 | awk '{print $NF}'; }
    RMSJ="{\"c0\": $(rmsv "$BFP" code), \"c1\": $(rmsv "$NFP" code), \"w0\": $(rmsv "$BFP" wt2), \"w1\": $(rmsv "$NFP" wt2)}"
    read -r V WHY < <("$PY" - "$CJ" "$WJ" "$RMSJ" <<'EOF'
import json, math, sys
c, w, r = (json.loads(x) for x in sys.argv[1:4]); why = []
if not (c["dkld"] < 0 and -c["dkld"] > 2 * c["dkld_se"]): why.append(f"编程KLD{c['dkld']:+.5f}±{c['dkld_se']:.5f}没降过2SE")
# wt2 只守底线(10-28 用户: "wt2 保持在 80 以上就行了, 主要还是把编程推进"): 新侧车 wt2 Same top ≥ WT2_SAME_FLOOR; 它的 KLD / Same 配对变化只报
WT2_SAME_FLOOR = 80.0
if w["same1"] < WT2_SAME_FLOOR: why.append(f"wt2Same{w['same1']:.2f}%低于{WT2_SAME_FLOOR:.0f}%")
if -c["same_D"] > 2 * math.sqrt(max(1, c["same_d"])): why.append(f"编程Same退{c['same_D']:+d}行超2√{c['same_d']}")
best = lambda x, r0, r1: x["dkld"] <= 0 and r1 <= r0 and x["same_D"] >= -math.sqrt(max(1, x["same_d"]))
tag = "三项全优" if best(c, r["c0"], r["c1"]) and best(w, r["w0"], r["w1"]) else "非全优"
info = f"Same编程{c['same_D']:+d}/√{c['same_d']}·wt2{w['same_D']:+d}/√{w['same_d']}(wt2 KLD{w['dkld']:+.5f}±{w['dkld_se']:.5f} 只报);RMS编程{r['c0']}→{r['c1']}·wt2{r['w0']}→{r['w1']};{tag}"
print("keep" if not why else "rollback", ";".join(why + [info]) if why else "全过;" + info)
EOF
)
    # 第二层不设门(10-29 用户: "先看看指标, 就算回退也不一定是坏事, 因为决策更好了"): 门的读数照算照记, 去留一律留, 鹈鹕(dsh)另判
    [ "$DECM" = 1 ] && { V="keep(dec)"; WHY="第二层不设门,只报:$WHY;决策位 $(grep '^decision' "$OUT/train.log" | awk '{printf "%s kl %s n %s; ", $2, $4, $6}')"; }
    if [ "$V" != rollback ]; then M "⑤ 留: 新侧车 $NFP 成为现役($WHY)"
        if [ "$BEH" = 1 ]; then
            beh_gen > "$WORK/beh_new.log" 2>&1 || die "新侧车评委失败(看 $WORK/beh_new.log)"
            M "⑤ 评委(只报):"; "$PY" gguf-tools/scripts/api_round.py summary "官方=$HOLD/official.review.jsonl" "旧现役=$EVO/beh/$BFP.review.jsonl" "新现役=$EVO/beh/$NFP.review.jsonl" 2>&1 | tail -8
        fi
    else
        rm -f "$ARM"/amp_L*.bin; cp "$WORK/init"/amp_L*.bin "$ARM/" || die "回退拷 amp 失败"
        printf '# 回退(%s 没过门: %s): 段 B 换回 %s/init(%s), 新 amp 在 %s/%s, %s\n' "$TAG" "$WHY" "$WORK" "$BFP" "$OUT" "$CK" "$(date '+%Y-%m-%d %H:%M')" >> "$ARM/manifest.txt"
        bash "$W" true > "$WORK/reload.log" 2>&1; waitready || die "回退后模型没装回"
        [ "$(ampmd5)" = "$BFP" ] || die "回退后指纹不对"; M "⑤ 回退: $WHY; 侧车换回 $BFP 并已重装"
    fi
    trap - INT TERM   # 去留已定(留 = 新段 B 就是现役; 回退 = 已换回), 之后打断不用再换
    # ledger 列: 轮 现役 新 训练料KLD起 训练料KLD末 编程KLD现役 编程KLD新 编程ΔKLD SE 编程Same现役 编程Same新 wt2KLD现役 wt2KLD新 wt2ΔKLD SE wt2Same现役 wt2Same新 去留 理由 why
    #   + 报数半边 ΔKLD SE Same净行数(10-29 起; 编程那几列 10-29 起是门半边, 之前是整把尺)
    read -r K0 K1 < <(awk '/^step 0 /{for(i=1;i<NF;i++) if($i=="train_kl") a=$(i+1)} /^epoch /{for(i=1;i<NF;i++) if($i=="train_kl") b=$(i+1)} END{print a, b}' "$OUT/train.log")
    "$PY" - "$CJ" "$WJ" "$RJ" <<EOF >> "$EVO/ledger_api.tsv"
import json, sys
c, w, h = (json.loads(x) for x in sys.argv[1:4]); r = lambda x, k=5: round(x, k)
print("\t".join(map(str, ["$TAG", "$BFP", "$NFP", "$K0", "$K1", r(c["kld0"]), r(c["kld1"]), r(c["dkld"]), r(c["dkld_se"]), r(c["same0"], 2), r(c["same1"], 2),
      r(w["kld0"]), r(w["kld1"]), r(w["dkld"]), r(w["dkld_se"]), r(w["same0"], 2), r(w["same1"], 2), "$V", "$WHY", open("$WORK/why.txt").read().strip().replace("\t", " "),
      r(h["dkld"]), r(h["dkld_se"]), h["same_D"]])))
EOF
    M "ledger: $(tail -1 "$EVO/ledger_api.tsv")"
fi
if stage 6 && [ "$DECM" = 1 ]; then M "⑥ 第二层本轮不复盘(计划分层只有 raw / task; 10-29 用户: 先看指标和鹈鹕)"
elif stage 6; then   # ⑥ 官方复盘 → 下一轮计划
    tick 复盘
    # 本轮的去留以账本为准(⑤ 写的那一行: 第 3 列新侧车指纹, 第 18/19 列去留与理由); 门的读数从两套 rows 重算(同一工具, 逐位确定)
    IFS=$'\t' read -r NFP V WHY < <(awk -F'\t' -v t="$TAG" '$1 == t {n = $3; v = $18; w = $19} END {print n "\t" v "\t" w}' "$EVO/ledger_api.tsv")
    [ -n "$NFP" ] && [ -s "$FR/$NFP/code.rows" ] || die "账本里没有 $TAG 这一轮(或它的 rows 不在 $FR/$NFP), 复盘没有事实可摆"
    CJ=$(pj "$FR/code_gate.lst" 2>/dev/null) && WJ=$(pj "$FR/wt2.lst" --block 32 2>/dev/null) || die "门的读数重算失败"
    # 编程尺按语言拆开: 同一份 rows、同一个配对工具, 每种语言只取门半边的两个窗(第 1/3 窗, 32 位一块); 报数半边不进复盘, 否则计划会照着它挑语言
    : > "$WORK/lang.tsv"
    while read -r lang s n; do
        J=$(echo "code 15360 $s:$((s + 128)),$((s + 256)):$((s + 384))" > "$WORK/lang.lst"; pj "$WORK/lang.lst" --block 32 2>/dev/null) || die "分语言配对失败($lang)"
        "$PY" -c "import json,sys; d=json.loads(sys.argv[2]); print(sys.argv[1], d['kld0'], d['kld1'], d['dkld'], d['dkld_se'])" "$lang" "$J" >> "$WORK/lang.tsv"
    done < <(grep -v '^#' gguf/go-onebit/code/judge.ids.layout | awk 'NF == 3')
    rm -f "$WORK/lang.lst"
    LIVE=$(ampmd5)
    if [ "$RAWM" = 1 ]; then LAYER=(--layer raw --raw-tok "$RAWTOK"); HIST=("$EVO"/o[0-9]*/facts.json)
        PNOTE="第一层历来各轮(o001~)都是这套流程; 任务层(a001~a012, 出题 → 对话作答)十一轮编程尺一次都没降过 2 SE, 第一层 o001~o003 三轮全过(10-28)。"
    else LAYER=(--layer task); HIST=("$EVO"/a[0-9]*/facts.json)
        PNOTE="下一轮你关思考作答(料 = 正文本身, 几乎全是代码; 10-28 实测一题 4185 token / 15.5 秒), 本轮及以前若是思考轨迹, 料的成分与时长都不能直接照搬。"; fi
    read -r K0 K1 < <(awk '/^step 0 /{for(i=1;i<NF;i++) if($i=="train_kl") a=$(i+1)} /^epoch /{for(i=1;i<NF;i++) if($i=="train_kl") b=$(i+1)} END{print a, b}' "$OUT/train.log")
    # 缺口按 token 类拆: 尺(现役的编程尺 rows)与本轮料(--restore-b 落的第 0 步逐位 KL)各一张表 —— 料暴露的缺口和尺上的缺口是不是同一类
    "$PY" gguf-tools/scripts/api_round.py tokclass --hf "$HF" --ruler-ids "$CODE" --ruler-rows "$FR/$LIVE/code.rows" --mat "$OUT/kl0.tsv" --out "$WORK/classes.txt" | sed "s/^/  /" || die "缺口按类拆失败"
    M "⑥ 官方复盘: 编程尺 $(wc -l < "$WORK/lang.tsv") 种语言的现役读数与本轮变化 + 门($V) + 训练料 KLD $K0 → $K1 + 历来各轮 → 下一轮计划"
    T plan "$PY" gguf-tools/scripts/api_round.py plan --key "$KEY" --langtab "$WORK/lang.tsv" --code-json "$CJ" --wt2-json "$WJ" \
        --q "$QJ" --train "$K0 $K1" --lr "$LR" --tag "$TAG" --verdict "$V" --why "$WHY" --review "$EVO/beh/$LIVE.review.jsonl" --plan-in "$PIN" \
        --times "$WORK/times.tsv" --train-log "$OUT/train.log" --raw "$RAW" --budget "$BUDGET" --note "$PNOTE" --classes "$WORK/classes.txt" \
        --params "$PARAMSJ" --history "${HIST[@]}" --facts "$WORK/facts.json" --out "$WORK/plan_out.json" "${LAYER[@]}" | sed 's/^/  /' || die "官方复盘失败"
    cp "$WORK/plan_out.json" "$PNEXT" || die "计划没写进 $PNEXT"
fi
if stage 7; then   # ⑦ 清理(10-27 用户: "每次跑完记得清理临时文件"): 本轮训练教师表(料是一次性的)
    before=$(df -B1 --output=avail . | tail -1)
    rm -f "$OUT"/teacher.part.* "$OUT"/teacher.ptf "$OUT"/teacher.rest "$DATA".seqs "$DATA".ptf "$DATA".covered
    M "⑦ 清理: 腾出 $(( ($(df -B1 --output=avail . | tail -1) - before) / 1000000000 )) GB; 盘余 $(df -h --output=avail . | tail -1 | tr -d ' ')"
fi
M "完成"; echo "APR_EXIT 0"
