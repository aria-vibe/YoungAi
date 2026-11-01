#!/bin/bash
# html_region_judge_spark.sh — 一份 HTML(部署态: BOS + 全文)过 FP 锚与三臂(① 裸 / ①② 侧车 / ①②③ 现役), 逐位置读数按内容区分桶(2026-10-24, docs/graphics_sidecar.md §2 第 1 步)。
#
# 干什么: 官方原件写的鹈鹕页面是"原件在同一句请求下真写出来的东西"; 把它 teacher-forced 过三臂, 看量化在 path 坐标 / 标签属性 / CSS / JS / 文字哪一区掉得最多,
#   回答用户 10-24 的判断"编程里的数学/美术成分是不是被量化吃掉了"。只串既有入口: 造料 judge_rows_tokclass.py --html, 判决 v41_judge.sh(教师锚 + engine 三臂 + .rows), 分桶 judge_rows_tokclass.py --regions。
# 用法(spark ~/ds4-main, 要机器空出来 —— 包在 with_model_stopped_spark.sh 里; ★dsh 对照跑着的时候不能发, 它用的就是这台模型★):
#   gguf-tools/scripts/with_model_stopped_spark.sh bash gguf-tools/scripts/html_region_judge_spark.sh gguf/go-onebit/code/pelican/official_pel1.html [更多 html...]
# 臂(与 spark 现役部署对一致, 读 gguf/serve_pick.txt 的 ① 与 ②; ③ 固定现役选中轮):
#   ① engine:<gguf>   ①② engine:<gguf>:<②>   ①②③ engine:<gguf>:<②>::<③>
# 产物: <html 同名>.ids / .ids.regions(gguf/go-onebit/code/pelican/), 教师锚与学生 .rows 在 gguf/v41judge/, 分桶表打在 stdout(调用方 tee 进日志)。
# 出错会怎样: 造料失败直接停; 判决某臂失败 v41_judge 打 ★ 并跳过, 分桶只在三臂 .rows 都在时跑(缺臂就打 ★ 列出缺哪份, 不拿两臂冒充三臂)。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
HF="hf/DeepSeek-V4.1-Flash-1017"
PY="$HOME/v41env/bin/python"; TC="gguf-tools/scripts/judge_rows_tokclass.py"; J="gguf-tools/scripts/v41_judge.sh"
PT="gguf/v41/posttrain/fp-formal2-A-B-M-L0-39-lr2e-4-e2-1022/ckpt_e02"   # 现役对的 ③(10-22 选中轮, judge.ids 84.77 那臂)
DEP="gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-code_fit_n15360-engine.1018"   # 现役对的 ②(编程侧车)
mapfile -t PK < gguf/serve_pick.txt; GG="${PK[0]}"; ZC="${PK[1]}"
GG="${GG#"$ROOT"/}"; ZC="${ZC#"$ROOT"/}"
# 三臂(10-25 改): ① 裸 / ①+现役对(编程 ② .1018 + ③ e02, 两目录) / ①+serve_pick 的侧车(整目录; 10-25 起是还原侧车, ② 与段 B 的 amp 一个目录, 引擎自己读 amp)。
#   10-24 的 ①/①②/①②③ 三臂在"②③ 并进一个目录"后不成立: 给还原侧车再叠一个 ③ 是两份低秩互相打架。
# --zc <②目录>: 换第三臂的侧车目录(图形域侧车 vs 还原侧车同尺对比)。
while [ $# -gt 0 ]; do case "$1" in --zc) ZC="${2:?--zc 要目录}"; ZC="${ZC#"$ROOT"/}"; shift 2;; *) break;; esac; done
[ $# -ge 1 ] || { echo "用法: $0 [--zc <侧车目录>] <html> [html...]"; exit 2; }
for f in "$GG" "$ZC/manifest.txt" "$DEP/manifest.txt" "$PT/amp_L00.bin" "$HF/config.json"; do [ -e "$f" ] || { echo "★缺 $f★"; exit 2; }; done
M(){ echo "[region $(date '+%m-%d %H:%M:%S')] $*"; }
# 决策位尺(10-25): 词表里全部 "</" 开头的 token 当"收尾"集, anchor_metrics --set 给每个位置多两列(FP / 学生落在这组上的质量)
CS="gguf/go-onebit/code/pelican/close_tokens.ids"
[ -s "$CS" ] || "$PY" "$TC" --hf "$HF" --close-set "$CS" || { echo "★造收尾集失败★"; exit 2; }
rc=0
for H in "$@"; do
  [ -s "$H" ] || { M "★$H 不存在★"; rc=1; continue; }
  IDS="${H%.html}.ids"
  M "造料 $H → $IDS(+ .regions / .decisions)"
  "$PY" "$TC" --hf "$HF" --html "$H" --out "$IDS" || { M "★造料失败★"; rc=1; continue; }
  N=$(wc -l < "$IDS")
  M "判决 $IDS n=$N: FP 锚 + 三臂(裸 / 现役对 / $(basename "$ZC"))"
  bash "$J" "$IDS" "$N" --hf "$HF" --set "$CS" "engine:$GG" "engine:$GG:$DEP::$PT" "engine:$GG:$ZC" || M "★判决有失败项(看 gguf/v41judge/log_stu_*)★"
  # 学生 .rows 的命名照 v41_judge.sh: stu_<语料tag>_eng_<gguf名>[_amp_<②名>][_pt_<③路径 / → _>].rows; 语料 tag = corpus_tag(ids) = 目录名_文件名_n<N>
  TAG="$(basename "$(dirname "$IDS")")_$(basename "$IDS" .ids)_n$N"; GB="$(basename "$GG" .gguf)"; ZB="$(basename "$ZC")"; DB="$(basename "$DEP")"; PTN="${PT//\//_}"
  R1="gguf/v41judge/stu_${TAG}_eng_${GB}.rows"; R2="gguf/v41judge/stu_${TAG}_eng_${GB}_amp_${DB}_pt_${PTN}.rows"; R3="gguf/v41judge/stu_${TAG}_eng_${GB}_amp_${ZB}.rows"
  miss=0; for r in "$R1" "$R2" "$R3"; do [ -s "$r" ] || { M "★缺 $r★"; miss=1; }; done
  if [ "$miss" = 0 ]; then
    M "按内容区 + 决策位分桶 $H"
    "$PY" "$TC" --hf "$HF" --ids "$IDS" --regions "$IDS.regions" --decisions "$IDS.decisions" "①=$R1" "现役对=$R2" "$(basename "$ZC" | sed 's/.*-grrb-//')=$R3" || rc=1
  else
    # 兜不住就报名字: 判决脚本的命名若改了, 这里 glob 一下给人看, 不猜
    ls -t gguf/v41judge/stu_${TAG}_eng_*.rows 2>/dev/null | head -5; rc=1
  fi
done
M "收工 rc=$rc"; exit $rc
