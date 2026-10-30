#!/bin/bash
# 把 docs/img/arch.html 渲染成 README 首图: arch.png(中文) 与 arch-en.png(英文)。
# 只在 Mac 上跑, 用本机 Google Chrome 无头模式, 不装任何依赖。
# 两遍: 先 --dump-dom 读页面实际高度(页面里的脚本写进 body 的 data-h), 再按这个高度截图;
# 窗口高写死的话, 改了图的内容就会被截掉底边或多出一片空白。
set -euo pipefail
cd "$(dirname "$0")"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
[ -x "$CHROME" ] || { echo "找不到 $CHROME" >&2; exit 1; }
# $1 = 页面名(默认 arch; quant_compare 等同目录的其他图也走这条), 宽度从页面 body 的 width 读。
NAME="${1:-arch}"
SRC="file://$PWD/$NAME.html"
W=$(grep -o 'width: *[0-9]*px' "$NAME.html" | head -1 | grep -o '[0-9]*')
for lang in zh en; do
    q=""; out=$NAME.png
    [ "$lang" = en ] && { q="?en"; out=$NAME-en.png; }
    h=$("$CHROME" --headless --disable-gpu --window-size=$W,1200 --virtual-time-budget=3000 --dump-dom "$SRC$q" 2>/dev/null \
        | grep -o 'data-h="[0-9]*"' | grep -o '[0-9]*')
    [ -n "$h" ] || { echo "$lang: 没读到页面高度" >&2; exit 1; }
    # 2 倍像素: GitHub 按容器宽缩小显示, 1 倍在高分屏上字是糊的
    "$CHROME" --headless --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
        --window-size=$W,"$h" --screenshot="$PWD/$out" "$SRC$q" 2>/dev/null
    echo "$out  ${W}x${h} css px"
done
