#!/bin/bash
# pr6_verify_spark.sh — 外部 PR #6(Metal 正确性 + 请求缓冲生命周期, 改了 core_v41_api/kvreuse/state/forward 共享路)在 spark 上的真模型门(2026-10-27)。
#
# 怎么跑: ~/ds4-main/speed-bench/pr6_verify_spark.sh <客户端目录> <臂目录>...
#   例: pr6_verify_spark.sh ~/ds4-pr6 ~/ds4-pr6-base ~/ds4-pr6 ~/ds4-pr6-nogate
#   客户端目录 = PR 源码(用它的 tests/repro/metal_api.py + 夹具 + src/eval/eval_cases_a.c 出 q1–q4); 每个臂目录里要有已编好的 ds4-server / ds4-eval。
#   第一个臂当基线, 其余逐条和它比。日志与产物在 /tmp/pr6v/, 终端只打进度和结论行。
# 做什么: 工作台模型子进程空闲就停它(主进程不动), 退出时按 gguf/serve_pick.txt 装回(停/装回逐字抄 kernel_ab_spark.sh);
#   每个臂: :8011 起 ds4-server(生产同参: 现役还原侧车 + --mem-budget-mb 110000) → metal_api.py 三套:
#     quick     = 预热 + 4 个 8K 夹具各两遍(温 0, 两遍必须同 = 决定论)
#     questions = q1–q4(温 0, seed 1, 思考档, 2048 上限)
#     lifetime  = 7.3 万 token 冷启 + 12 条续接(必须复用 ≥6 万前缀 = 缓存命中路) + 13.9 万 token 预填中途断开 + 断开后再发一条
#   → 停 server → ds4-eval q1–q4(裸底座, 温 0 seed 1, 2048; 它没有 --zchain)。
# 判读: 每条用例的 all_sha256(思考 + 正文)与基线逐条比, "全同 ✓" 是硬门; 预填/解码 t/s 只报数(单趟)。
# 出错会怎样: 模型子进程不是 idle 直接退出不停它; 某臂 server 起不来或某套报错 → 记 ★ 跳到下一臂, 不影响装回;
#   metal_api.py 的 lifetime 自带断言(续接没复用 / 断开没被看到)会让该套失败, 日志里有原文。
set -uo pipefail
ROOT="$HOME/ds4-main"; cd "$ROOT" || exit 1
M(){ echo "[pr6v $(date '+%H:%M:%S')] $*"; }
CLIENT="${1:?用法: $0 <客户端目录> <臂目录>...}"; shift
[ $# -ge 2 ] || { M "★至少两个臂(基线 + 候选)★"; exit 1; }
[ -f "$CLIENT/tests/repro/metal_api.py" ] || { M "★$CLIENT 里没有 tests/repro/metal_api.py★"; exit 1; }
MODEL="$ROOT/gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf"
AMP="$ROOT/gguf/v41/DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-restore_fit_n15360-engine"
PORT=8000; SPORT=8011; OUT=/tmp/pr6v
for d in "$@"; do [ -x "$d/ds4-server" ] && [ -x "$d/ds4-eval" ] || { M "★$d 缺 ds4-server 或 ds4-eval★"; exit 1; }; done
[ -e "$OUT" ] && { M "★$OUT 已存在, 先挪走(不覆盖上一趟)★"; exit 1; }
mkdir -p "$OUT"

restore_model() {
    local pg pz tok body resp
    pg=$(sed -n 1p gguf/serve_pick.txt); pz=$(sed -n 2p gguf/serve_pick.txt)
    pg="${pg#"$ROOT"/}"; pz="${pz#"$ROOT"/}"; [ -n "$pz" ] || pz=none
    tok=$(cat ~/.ds4/assistant/token 2>/dev/null || true)
    body=$(printf '{"gguf":"%s","sidecar":"%s"}' "$pg" "$pz")
    resp=$(curl -s --noproxy '*' --max-time 30 ${tok:+-H "Authorization: Bearer $tok"} -H 'Content-Type: application/json' -d "$body" "http://127.0.0.1:$PORT/api/models/load")
    M "装回 $pg + $pz → $resp"
}
SP=""
cleanup() { rc=$?; [ -n "$SP" ] && { kill "$SP" 2>/dev/null; wait "$SP" 2>/dev/null; }; [ "$stopped" = 1 ] && restore_model; echo "PR6V_EXIT $rc"; }

st=$(curl -s --noproxy '*' --max-time 5 127.0.0.1:8001/metrics | tr ',' '\n' | grep -o '"state":"[a-z]*"' | head -1)
[ -z "$st" ] || [ "$st" = '"state":"idle"' ] || { M "★模型子进程 $st, 不是 idle, 不停★"; exit 1; }
stopped=0
if pgrep -x ds4-server >/dev/null; then
    M "停模型子进程"; pkill -x -TERM ds4-server; stopped=1
    for _ in $(seq 1 60); do pgrep -x ds4-server >/dev/null || break; sleep 2; done
    pgrep -x ds4-server >/dev/null && { pkill -x -KILL ds4-server; sleep 5; }
    pgrep -x ds4-server >/dev/null && { M "★ds4-server 杀不掉★"; exit 1; }
    sleep 5
fi
trap cleanup EXIT

for d in "$@"; do
    arm=$(basename "$d"); A="$OUT/$arm"; mkdir -p "$A"
    M "== 臂 $arm($(md5sum < "$d/ds4-server" | cut -c1-12)) available $(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo) GB 时钟 $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader)"
    (cd "$d" && exec ./ds4-server --cuda -m "$MODEL" --zchain "$AMP" --mem-budget-mb 110000 --host 127.0.0.1 --port "$SPORT") > "$A/server.log" 2>&1 &
    SP=$!
    up=0; for _ in $(seq 1 180); do kill -0 "$SP" 2>/dev/null || break; curl -s --noproxy '*' -m 2 "127.0.0.1:$SPORT/metrics" >/dev/null 2>&1 && { up=1; break; }; sleep 5; done
    if [ "$up" = 1 ]; then
        for s in quick questions lifetime; do
            M "  $arm / $s"
            python3 -I "$CLIENT/tests/repro/metal_api.py" --url "http://127.0.0.1:$SPORT" --out "$A/$s" --suite "$s" > "$A/$s.log" 2>&1
            rc=$?; [ $rc -eq 0 ] && M "  $arm / $s 完成" || M "  ★$arm / $s rc=$rc★ $(tail -1 "$A/$s.log" | cut -c1-200)"
        done
    else
        M "  ★$arm server 没起来★ $(tail -2 "$A/server.log" | tr '\n' ' ' | cut -c1-200)"
    fi
    kill "$SP" 2>/dev/null; wait "$SP" 2>/dev/null; SP=""; sleep 5
    M "  $arm / ds4-eval q1–q4"
    (cd "$d" && exec ./ds4-eval --cuda -m "$MODEL" --mem-budget-mb 110000 --plain --questions 4 --tokens 2048 --temp 0 --seed 1 --trace "$A/eval.trace") > "$A/eval.log" 2>&1
    rc=$?; M "  $arm / ds4-eval rc=$rc $(grep -a -iE 'score|correct|passed|得分' "$A/eval.log" | tail -1 | cut -c1-160)"
done

M "== 汇总(第一个臂是基线)"
python3 -I - "$OUT" "$@" <<'PY'
import hashlib, json, sys
from pathlib import Path
out = Path(sys.argv[1]); arms = [Path(a).name for a in sys.argv[2:]]
def runs(arm):
    r = {}
    for s in ['quick', 'questions', 'lifetime']:
        f = out / arm / s / 'summary.json'
        if f.exists():
            for x in json.loads(f.read_text()): r[f'{s}/{x["name"]}'] = x
        else:
            for rep in sorted((out / arm / s).glob('*/report.json')) if (out / arm / s).exists() else []:
                r[f'{s}/{rep.parent.name}'] = json.loads(rep.read_text())
    return r
R = {a: runs(a) for a in arms}
base = arms[0]
for a in arms[1:]:
    same = diff = miss = 0
    for k, b in R[base].items():
        c = R[a].get(k)
        if c is None: miss += 1; print(f'  {a}: 缺 {k}'); continue
        if c.get('all_sha256') == b.get('all_sha256'): same += 1
        else: diff += 1; print(f'  ★{a}: {k} 与基线不同★')
    print(f'{a} 对 {base}: 同 {same} / 不同 {diff} / 缺 {miss}' + ('  全同 ✓' if diff == 0 and miss == 0 and same else ''))
    ea, eb = out / a / 'eval.log', out / base / 'eval.log'
    def body(p):
        t = p.read_bytes() if p.exists() else b''
        return hashlib.sha256(b'\n'.join(l for l in t.split(b'\n') if b' t/s' not in l and b'sec' not in l)).hexdigest()[:12]
    print(f'  ds4-eval 日志(去掉速度行)哈希 {body(eb)} vs {body(ea)}')
print('用例                           ' + '  '.join(f'{a[:22]:>22}' for a in arms) + '   (首字 s / 解码 t/s / 复用)')
for k in R[base]:
    cells = []
    for a in arms:
        m = R[a].get(k, {}).get('metrics', {}) or {}
        cells.append(f'{m.get("prompt_ms", 0)/1000:7.2f}/{m.get("decode_tok_s") or 0:5.1f}/{m.get("reused") or 0:6}' if m else f'{"-":>22}')
    print(f'{k:30} ' + '  '.join(f'{c:>22}' for c in cells))
for a in arms:
    f = out / a / 'lifetime' / 'cancel.json'
    if f.exists(): print(f'{a}: 断开后回 idle {json.loads(f.read_text())["seconds_to_idle"]:.2f} s')
PY
echo ALLDONE
