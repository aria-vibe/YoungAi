#!/bin/bash
# dsh_compare.sh — 同一句请求放进 DeepSeek Harness(dsh, npm 包装在 gguf/dsh/, 10-23 用户同意)跑, 两臂只换模型来源, 用来分"效果差是模型还是 harness":
#   official = 官方 API 的 deepseek-flash(= DeepSeek-V4.1-Flash 原件, 全精度)
#   local    = spark 上正在服务的 ds4-server(1017 量化底座 + 侧车), 经 ssh 隧道直连模型子进程 127.0.0.1:8001
# 两臂同一个 dsh、同一个适配器(dsh-llm-deepseek, Anthropic Messages 协议 —— ds4-server 本来就讲 /v1/messages)、同一份 patch, 只有 baseURL 不同。
# patch 里定死: maxTokens 65536(= 官方思考模式默认上限, 也是我们助理的上限; dsh 默认 256000, 本地进环会跑两小时) /
#   models 只列 deepseek-flash 且不带 systemPromptUpdate / toolUpdate(官方默认条目带, ds4-server 不认中途 system 与 tool_addition 块; 两臂都去掉才可比) /
#   审批 never(headless 没人点"允许"; 沙盒仍是 dsh 默认的 workspace-write: 只能写本次的工作目录); 权限插件要一个对得上的预设, 否则它不启动(10-23 实撞 "match no preset")。
# 用法: tests/asst/dsh_compare.sh official|local "请求" [标签(默认时间戳)]
# 产物: gguf/dsh/runs/<臂>-<标签>/ws/ = dsh 的工作目录(跑时在 /tmp/project.XXXXXX, 跑完挪回; 模型写的文件都在这); 上一层 events.jsonl(--json 事件流) + stderr.log + patch.yml + request.txt; 结束打印用时与 ws/ 文件清单。
# key: 官方臂读 gguf-tools/data/posttrain/deepseek_api.key(.gitignore 的 *.key), 只经子进程环境交给 dsh, 不打印、不落盘到产物目录。
# 出错会怎样: 隧道起不来 → 本地臂直接退(不回退到官方); dsh 报 MISSING_CREDENTIAL = key 文件没读到; 报 404 /v1/messages = 隧道连到了别的端口。
set -uo pipefail
ARM="${1:?official|local}"; MSG="${2:?请求}"; TAG="${3:-$(date +%m%d-%H%M%S)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DSH="$ROOT/gguf/dsh/node_modules/.bin/dsh"
[ -x "$DSH" ] || { echo "dsh 没装: cd gguf/dsh && npm install @deepseek-ai/dsh@0.2.1-rc.2"; exit 1; }
RUN="$ROOT/gguf/dsh/runs/$ARM-$TAG"; mkdir -p "$RUN"
[ -e "$RUN/ws" ] && { echo "$RUN/ws 已存在, 换个标签"; exit 1; }
# 模型的工作目录 = /tmp 下随机名的空目录, 跑完整个挪回 $RUN/ws。不能直接在 $RUN/ws 里跑:
#   dsh 的 workspace-write 沙盒只管写不管读, 10-26 pel-f2b 一收口就 `ls ..`, 从目录名 local-pel-f2b 猜出是评测,
#   翻到隔壁 official-pel1/ws/index.html(官方答案)和各趟截图 —— 照抄的结果没法判模型。路径里也不能带仓库、臂名、标签。
# 10-23 第一趟模型一 ls 就看见 patch.yml(baseURL 暴露本地还是官方), 所以日志与 patch 一直放在工作目录外。
# dsh 从工作目录往上找 .git 当项目根, 一路读 AGENTS.md / CLAUDE.md 进系统提示: 给工作目录单独 git init, 它自己就是项目根, 两臂都是一个干净的空项目 —— 和用户平时在一个空目录里用 dsh 一样
WS="$(mktemp -d /tmp/project.XXXXXX)" || { echo "建不了工作目录"; exit 1; }
git -C "$WS" init -q
TUNNEL_PORT=18431
case "$ARM" in
  official)
    BASE="https://api.deepseek.com/anthropic"
    KEY="$(cat "$ROOT/gguf-tools/data/posttrain/deepseek_api.key")" || { echo "读不到官方 key"; exit 1; } ;;
  local)
    if ! nc -z 127.0.0.1 $TUNNEL_PORT 2>/dev/null; then
      ssh -f -N -o ExitOnForwardFailure=yes -L $TUNNEL_PORT:127.0.0.1:8001 spark || { echo "到 spark:8001 的隧道起不来"; exit 1; }
    fi
    BASE="http://127.0.0.1:$TUNNEL_PORT"; KEY="local" ;;
  *) echo "臂只有 official / local"; exit 2 ;;
esac
cat > "$RUN/patch.yml" <<EOF
- id: llm-deepseek
  config:
    baseURL: $BASE
    maxTokens: 65536
    models:
      - id: deepseek-flash
        # 显式写 models 条目时 inputModalities 默认只剩 text(dsh-llm-deepseek 的 schema 默认 ["text"], 目录默认条目是 text+image):
        # 10-23~10-25 的 8 趟 read_image 全报 "does not declare image input", 模型瞎画、自己拿 sips 转字符画看图。两臂都补回看图(10-25 用户令)。
        inputModalities: [text, image]
- id: approval
  config:
    policy: never
- id: permission
  config:
    presets:
      workspace-auto:
        sandbox: workspace-write
        approval: never
    defaultPreset: workspace-auto
EOF
printf '%s\n' "$MSG" > "$RUN/request.txt"
T0=$(date +%s)
( cd "$WS" && DEEPSEEK_API_KEY="$KEY" NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost" \
    "$DSH" --profile headless --patch "$RUN/patch.yml" --json "$MSG" > "$RUN/events.jsonl" 2> "$RUN/stderr.log" )
RC=$?
mv "$WS" "$RUN/ws" || echo "工作目录没挪回来, 还在 $WS"
WS="$RUN/ws"
# 用时落盘给记分卡(tests/asst/dsh_scorecard.py)读: 事件流里没有时间戳, 不写这一行, 10-23 那几趟的用时就只剩 fable5 里的手抄数
echo "$(( $(date +%s) - T0 ))" > "$RUN/timing.txt"
echo "arm=$ARM tag=$TAG rc=$RC 用时 $(cat "$RUN/timing.txt") s  目录 $RUN"
( cd "$WS" && find . -type f -not -path './.*' | head -40 | while read -r f; do printf '  %s  %s B\n' "$f" "$(wc -c < "$f" | tr -d ' ')"; done )
[ $RC -eq 0 ] || tail -n 20 "$RUN/stderr.log"
exit $RC
