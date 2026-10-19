#!/bin/bash
# release_upload.sh — 把发布包传到 GitHub release(已建好的 tag, 草稿也行), 断了自己清残留重传(2026-10-19)。
#
# 怎么用: gguf-tools/scripts/release_upload.sh [--via <ssh 主机>] <tag> <文件>...
#   例: release_upload.sh --via spark v0.5.0-rc dist/youngai-darwin-arm64-20261019.tar.gz dist/youngai-linux-aarch64-20261019.tar.gz
#   每个文件最多传 3 次; 远端已有同名、状态 uploaded 且大小一致的直接跳过(重跑即续传)。走 gh 已登录的令牌, 不读任何新环境变量。
#   --via: 上传连接经那台机器出去(ssh -L 转发到 uploads.github.com:443, curl --connect-to 指过去; TLS 仍是本机对 GitHub,
#   令牌不落那台机器)。10-19 实测 Mac 的出口(代理和"直连"都一样, 像是全局接管)对单个连接只放 30~44 MB 就断,
#   限速到 1 MB/s 照样在 31 MB 断 —— 是字节上限不是速度; 经 spark 出去不受这个限制。
#
# 为什么不直接 gh release upload:
#   ① 上传中途断了(本机代理/线路掐长连接, 报 "unexpected EOF"), GitHub 会留下一个同名、状态 starter 的半截附件。
#      gh release view 不列它, gh --clobber 也删不掉它, 之后再传同名文件服务端一开头就拒、马上关连接 ——
#      客户端看到的是"传了 30 多 MB 就 Broken pipe / HTTP2 framing error", 很像网络问题, 换代理换直连都一样。
#      10-18 v0.4.0-rc 和 10-19 v0.5.0-rc 两次都撞上。这里每次失败后按名字把 starter 残留删掉再传。
#   ② gh 失败的退出码常被包在 $(...) 后面读错; 这里用 curl 的 HTTP 码判成败, 201 才算传上。
#   ③ 走 HTTP/1.1: 本机代理下 HTTP/2 的长上传流更容易被掐。
set -uo pipefail
VIA=""
[ "${1:-}" = --via ] && { VIA="${2:?--via 要一个 ssh 主机名}"; shift 2; }
TAG="${1:?用法: $0 [--via <ssh 主机>] <tag> <文件>...}"; shift
[ $# -gt 0 ] || { echo "★没给要传的文件★"; exit 2; }
REPO="fodelf/YoungAi"
RID=$(gh api "repos/$REPO/releases" --jq ".[] | select(.tag_name == \"$TAG\") | .id" | head -1)
[ -n "$RID" ] || { echo "★找不到 tag $TAG 的 release(草稿也算), 先 gh release create --draft★"; exit 1; }
TOKEN=$(gh auth token) || { echo "★gh 没登录★"; exit 1; }
fsize() { stat -f %z "$1" 2>/dev/null || stat -c %s "$1"; }   # BSD(Mac) / GNU
NET=()   # curl 的出口参数: 空 = 本机默认出口(含本机代理设置)
if [ -n "$VIA" ]; then
    PORT=18443
    ssh -f -N -o ExitOnForwardFailure=yes -L "$PORT:uploads.github.com:443" "$VIA" || { echo "★开不了经 $VIA 的转发(端口 $PORT 被占?)★"; exit 1; }
    FWD=$(pgrep -f "ssh -f -N -o ExitOnForwardFailure=yes -L $PORT:uploads.github.com:443 $VIA" | head -1)
    trap '[ -n "${FWD:-}" ] && kill "$FWD" 2>/dev/null' EXIT
    NET=(--noproxy '*' --connect-to "uploads.github.com:443:127.0.0.1:$PORT")
    echo "上传经 $VIA 出去(本机 127.0.0.1:$PORT → $VIA → uploads.github.com:443)"
fi
assets() { gh api "repos/$REPO/releases/$RID/assets" --paginate --jq '.[] | "\(.id) \(.state) \(.size) \(.name)"'; }
rc_all=0
for f in "$@"; do
    [ -s "$f" ] || { echo "★没有 $f★"; rc_all=1; continue; }
    name=$(basename "$f"); size=$(fsize "$f"); ok=0
    for k in 1 2 3; do
        line=$(assets | awk -v n="$name" '$4 == n')
        if [ -n "$line" ]; then
            read -r aid state asz _ <<<"$line"
            if [ "$state" = uploaded ] && [ "$asz" = "$size" ]; then echo "$name: 远端已有且大小一致, 跳过"; ok=1; break; fi
            echo "$name: 远端有残留(状态 $state, $asz 字节), 先删"; gh api -X DELETE "repos/$REPO/releases/assets/$aid" || true
        fi
        t0=$(date +%s)
        out=$(curl -sS --http1.1 ${NET[@]+"${NET[@]}"} -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/octet-stream" \
              -H "Accept: application/vnd.github+json" --data-binary @"$f" \
              "https://uploads.github.com/repos/$REPO/releases/$RID/assets?name=$name" \
              -o /dev/null -w "%{http_code} %{size_upload}" 2>&1 | tail -1)
        code=${out%% *}
        echo "$name: 第 $k 次 HTTP $code, 发出 ${out##* } / $size 字节, $(( $(date +%s) - t0 )) s"
        [ "$code" = 201 ] && { ok=1; break; }
    done
    [ "$ok" = 1 ] || { echo "★$name 三次都没传上★"; rc_all=1; }
done
echo "== 远端附件(id 状态 字节 名字)"; assets
exit $rc_all
