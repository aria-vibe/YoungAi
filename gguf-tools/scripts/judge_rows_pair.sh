#!/bin/bash
# judge_rows_pair.sh — 两臂在同一份判决份上的逐位配对(2026-10-23): 一臂改动了什么、改在哪, 为什么平均数不动。
#
# 用法: gguf-tools/scripts/judge_rows_pair.sh <A.rows> <B.rows>      (A = 对照臂, B = 改动臂; v41_judge.sh 落的 stu_*.rows)
#   .rows 每行 = 位置 kld smin same ref命中 stu命中 ref_argmax stu_argmax(anchor_metrics --row-out)
# 出什么:
#   ① ΔKL = KL_B − KL_A 的分布: 几乎没动(|Δ| < 1e-3)/ 变好 / 变坏各多少位、各自 Δ 之和 —— 看是"没改"还是"有好有坏抵掉"
#   ② 按两臂 KL 均值分十档(每档同位数; 不能只按 A 分: A 偶然偏高的位, B 看起来天然变好 —— 回归均值的假象): 各档 KL_A 均值、ΔKL 均值、档内变好占比 —— 看改进落在低 KL 的大多数位还是高 KL 的尾巴
#   ③ top-1 翻转: same 0→1 / 1→0 各多少位, 翻转位上 KL_A 的均值
#   ④ ΔKL 的集中度: |Δ| 最大的 1% / 10% 位占全部 |Δ| 的份额
# 为什么用 awk 不用 Python: 只做读数统计, 不进数值链; 两份 1.5 万行一次过完。
# 出错会怎样: 两份行数或位置对不上直接退出 2(拿错文件配对就是假账 —— v41_judge.sh 注释里记过同名 ckpt 后一臂盖掉前一臂的 .rows)。
set -euo pipefail
A="${1:?A.rows(对照臂)}"; B="${2:?B.rows(改动臂)}"
[ -s "$A" ] && [ -s "$B" ] || { echo "★.rows 不在或为空★"; exit 2; }
[ "$(wc -l < "$A")" = "$(wc -l < "$B")" ] || { echo "★两份行数不同: $(wc -l < "$A") vs $(wc -l < "$B")★"; exit 2; }
paste -d' ' "$A" "$B" | awk '$1 != $9 { print "★第 " NR " 行位置对不上: " $1 " vs " $9 "★"; bad = 1; exit } END { exit bad }' || exit 2
TMPF=$(mktemp /tmp/judge_rows_pair.XXXXXX)
paste -d' ' "$A" "$B" | awk '{ print ($2 + $10) / 2, $2, $10, $4, $12 }' | sort -g -k1,1 | awk '
{ ka[NR] = $2; kb[NR] = $3; sa[NR] = $4; sb[NR] = $5 }
END {
    n = NR
    for (i = 1; i <= n; i++) {
        d = kb[i] - ka[i]; dd[i] = d; ad[i] = d < 0 ? -d : d; tot += ad[i]; SA += ka[i]; SB += kb[i]
        if (ad[i] < 1e-3) { n0++ } else if (d < 0) { ng++; sg += d } else { nw++; sw += d }
        if (sa[i] == 0 && sb[i] == 1) { f01++; k01 += ka[i] }
        if (sa[i] == 1 && sb[i] == 0) { f10++; k10 += ka[i] }
    }
    printf "位数 %d | 平均 KL A %.5f → B %.5f (Δ %+.5f)\n", n, SA / n, SB / n, (SB - SA) / n
    printf "① 几乎没动(|Δ|<1e-3) %d 位 (%.1f%%) | 变好 %d 位 ΣΔ %+.2f | 变坏 %d 位 ΣΔ %+.2f | 净 %+.2f\n", n0, 100 * n0 / n, ng, sg, nw, sw, sg + sw
    printf "② 按两臂 KL 均值分十档(低 → 高):\n"
    for (q = 0; q < 10; q++) {
        lo = int(q * n / 10) + 1; hi = int((q + 1) * n / 10); s = 0; sd = 0; g = 0; m = hi - lo + 1
        for (i = lo; i <= hi; i++) { s += ka[i]; sd += dd[i]; if (dd[i] < -1e-3) g++ }
        printf "   档 %d  KL_A 均 %.4f  ΔKL 均 %+.5f  档内 ΣΔ %+.2f  变好占比 %.0f%%\n", q, s / m, sd / m, sd, 100 * g / m
    }
    printf "③ top-1 翻转: 0→1 %d 位 (这些位 KL_A 均 %.3f) / 1→0 %d 位 (KL_A 均 %.3f) / 净 %+d\n", f01, f01 ? k01 / f01 : 0, f10, f10 ? k10 / f10 : 0, f01 - f10
    for (i = 1; i <= n; i++) print ad[i] > TMP
}' TMP="$TMPF"
sort -gr "$TMPF" | awk '{ a[++n] = $1; t += $1 } END {
    for (i = 1; i <= int(n / 100); i++) s1 += a[i]
    for (i = 1; i <= int(n / 10); i++) s10 += a[i]
    printf "④ |ΔKL| 集中度: 最大 1%% 的位占 %.1f%%, 最大 10%% 占 %.1f%%\n", 100 * s1 / t, 100 * s10 / t }'
rm -f "$TMPF"
