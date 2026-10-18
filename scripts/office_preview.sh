#!/bin/sh
# scripts/office_preview.sh <文件.pptx|.docx|.pdf>... — 把产出文件渲染成每页一张 JPG, 肉眼看版式用。
# 用法: ./ds4_office_test --keep 打印 kept: /tmp/ds4_office_pdf_NNN, 然后 scripts/office_preview.sh /tmp/ds4_office_pdf_NNN/slides-*.pptx
# 输出: 每个文件旁边一个 <名>.preview/ 目录, 里面 p-1.jpg p-2.jpg …(50 dpi, 看布局够用; 要看细节改 DPI)。
# 依赖: LibreOffice(soffice)与 poppler(pdftoppm); Mac 用 brew 装, spark 上有 poppler 没 soffice 时只能转 pdf 输入。
# 没装 soffice 会报 "soffice: not found" 然后退出 1, 不是文件坏了。
set -e
DPI=${DPI:-50}
for f in "$@"; do
  d="${f%.*}.preview"; rm -rf "$d"; mkdir -p "$d"
  case "$f" in
    *.pdf) pdf="$f" ;;
    *) soffice --headless --convert-to pdf --outdir "$d" "$f" >/dev/null 2>&1; pdf="$d/$(basename "${f%.*}").pdf" ;;
  esac
  pdftoppm -jpeg -r "$DPI" "$pdf" "$d/p"
  echo "$d: $(ls "$d"/p-*.jpg | wc -l | tr -d ' ') pages"
done
