/* office_pptx_chart.c — 表头第一格是图型名的表 → 原生 PowerPoint 图表(ppt/charts/chartN.xml + 本页 graphicFrame)。
 * 规格(模型在 Markdown 里写一张普通表, 技能 skills/pptx/SKILL.md 教):
 *   表头: 第一格 bar / line / pie(也认 柱 / 折线 / 饼), 后面每格一个系列名
 *   数据行: 第一格类别, 后面数字(千分位逗号、% 和单位都会被剥掉只留数)
 *   最多 12 个类别 × 4 个系列; 饼图只画第一个系列。第一格不是图型名的表照常排成表格。
 * 为什么用表不用围栏代码块: 技能文件不许出现围栏(技能不带脚本, 检查器整份拒), 示例稿就写不进去; 表格模型本来就会写, 千分位逗号也不和分隔符打架。
 * 为什么原生不画图: 原生图表在 PowerPoint / WPS 里能改数、改色、跟主题走; 画成图片就死了。数据用 strLit/numLit 内嵌(不带 xlsx 工作簿, 三家都能开)。
 * 配色 = 主题 accent / c2 / c3 / c4; 轴与标签字色 = 正文色, 网格线 = 次要色 30%; 底透明(深色主题就是深底)。 */
#include "office_internal.h"
#include <stdio.h>

#define C_NS "xmlns:c=\"http://schemas.openxmlformats.org/drawingml/2006/chart\""
#define MAXC 12
#define MAXS 4
typedef struct { int kind, nc, ns; char cat[MAXC][64], ser[MAXS][64]; double v[MAXS][MAXC]; } chart_t;   /* kind 0 bar 1 line 2 pie */

static void cell_text(const office_node *c, char *out, size_t n) {   /* 单元格纯文本, 去首尾空格 */
    char *t = c ? office_plain(c->child) : NULL; const char *p = t ? t : ""; while (*p == ' ') p++;
    size_t k = strlen(p); while (k && p[k - 1] == ' ') k--; if (k >= n) k = n - 1; memcpy(out, p, k); out[k] = 0; free(t);
}
static double num(const char *s) { char t[64]; size_t k = 0; for (; *s && k + 1 < sizeof t; s++) if ((*s >= '0' && *s <= '9') || *s == '.' || *s == '-') t[k++] = *s; t[k] = 0; return k ? atof(t) : 0; }
static int kind_of(const char *tok) {
    if (!strncmp(tok, "bar", 3) || !strncmp(tok, "column", 6) || !strncmp(tok, "\xE6\x9F\xB1", 3)) return 0;
    if (!strncmp(tok, "line", 4) || !strncmp(tok, "\xE6\x8A\x98", 3)) return 1;
    if (!strncmp(tok, "pie", 3) || !strncmp(tok, "\xE9\xA5\xBC", 3)) return 2;
    return -1;
}
static bool parse(const office_node *t, chart_t *c) {
    memset(c, 0, sizeof *c); if (!t || t->kind != OFF_TABLE || !t->child || !t->child->header) return false;
    const office_node *hr = t->child; char tok[64]; cell_text(hr->child, tok, sizeof tok);
    if ((c->kind = kind_of(tok)) < 0) return false;
    for (const office_node *h = hr->child ? hr->child->next : NULL; h && c->ns < MAXS; h = h->next) { cell_text(h, c->ser[c->ns], sizeof c->ser[0]); if (c->ser[c->ns][0]) c->ns++; }
    for (const office_node *r = hr->next; r && c->nc < MAXC; r = r->next) {
        cell_text(r->child, c->cat[c->nc], sizeof c->cat[0]); if (!c->cat[c->nc][0]) continue;
        const office_node *d = r->child ? r->child->next : NULL;
        for (int s = 0; d && s < c->ns; s++, d = d->next) { cell_text(d, tok, sizeof tok); c->v[s][c->nc] = num(tok); }
        c->nc++;
    }
    return c->ns > 0 && c->nc > 0;
}
bool pptx_chart_ok(const office_node *tbl) { chart_t c; return parse(tbl, &c); }

static void txpr(ds4_buf *b, const char *color, int sz) {
    ds4_buf_printf(b, "<c:txPr><a:bodyPr/><a:lstStyle/><a:p><a:pPr><a:defRPr sz=\"%d\"><a:solidFill><a:srgbClr val=\"%s\"/></a:solidFill><a:latin typeface=\"+mn-lt\"/><a:ea typeface=\"+mn-ea\"/></a:defRPr></a:pPr><a:endParaRPr lang=\"zh-CN\"/></a:p></c:txPr>", sz, color);
}
static void dlbls(ds4_buf *b, const pptx_theme_t *T, const char *pos) {
    ds4_buf_puts(b, "<c:dLbls><c:numFmt formatCode=\"General\" sourceLinked=\"0\"/><c:spPr><a:noFill/><a:ln><a:noFill/></a:ln></c:spPr>"); txpr(b, T->text, 1100);
    ds4_buf_printf(b, "<c:dLblPos val=\"%s\"/><c:showLegendKey val=\"0\"/><c:showVal val=\"1\"/><c:showCatName val=\"0\"/><c:showSerName val=\"0\"/><c:showPercent val=\"0\"/><c:showBubbleSize val=\"0\"/></c:dLbls>", pos);
}
static void data(ds4_buf *b, const chart_t *c, int s) {
    ds4_buf_printf(b, "<c:cat><c:strLit><c:ptCount val=\"%d\"/>", c->nc);
    for (int i = 0; i < c->nc; i++) { ds4_buf_printf(b, "<c:pt idx=\"%d\"><c:v>", i); office_xml_text(b, c->cat[i]); ds4_buf_puts(b, "</c:v></c:pt>"); }
    ds4_buf_printf(b, "</c:strLit></c:cat><c:val><c:numLit><c:formatCode>General</c:formatCode><c:ptCount val=\"%d\"/>", c->nc);
    for (int i = 0; i < c->nc; i++) ds4_buf_printf(b, "<c:pt idx=\"%d\"><c:v>%g</c:v></c:pt>", i, c->v[s][i]);
    ds4_buf_puts(b, "</c:numLit></c:val>");
}
static void axes(ds4_buf *b, const pptx_theme_t *T) {
    ds4_buf_printf(b, "<c:catAx><c:axId val=\"1\"/><c:scaling><c:orientation val=\"minMax\"/></c:scaling><c:delete val=\"0\"/><c:axPos val=\"b\"/><c:numFmt formatCode=\"General\" sourceLinked=\"1\"/><c:majorTickMark val=\"none\"/><c:minorTickMark val=\"none\"/><c:tickLblPos val=\"nextTo\"/>"
                      "<c:spPr><a:ln w=\"9525\"><a:solidFill><a:srgbClr val=\"%s\"><a:alpha val=\"50000\"/></a:srgbClr></a:solidFill></a:ln></c:spPr>", T->muted);
    txpr(b, T->text, 1200);
    ds4_buf_puts(b, "<c:crossAx val=\"2\"/><c:crosses val=\"autoZero\"/><c:auto val=\"1\"/><c:lblAlgn val=\"ctr\"/><c:lblOffset val=\"100\"/><c:noMultiLvlLbl val=\"0\"/></c:catAx>");
    ds4_buf_printf(b, "<c:valAx><c:axId val=\"2\"/><c:scaling><c:orientation val=\"minMax\"/></c:scaling><c:delete val=\"0\"/><c:axPos val=\"l\"/><c:majorGridlines><c:spPr><a:ln w=\"6350\"><a:solidFill><a:srgbClr val=\"%s\"><a:alpha val=\"30000\"/></a:srgbClr></a:solidFill></a:ln></c:spPr></c:majorGridlines>"
                      "<c:numFmt formatCode=\"General\" sourceLinked=\"1\"/><c:majorTickMark val=\"none\"/><c:minorTickMark val=\"none\"/><c:tickLblPos val=\"nextTo\"/><c:spPr><a:ln><a:noFill/></a:ln></c:spPr>", T->muted);
    txpr(b, T->muted, 1100);
    ds4_buf_puts(b, "<c:crossAx val=\"1\"/><c:crosses val=\"autoZero\"/><c:crossBetween val=\"between\"/></c:valAx>");
}
void pptx_chart(pptx_sw *w, const office_node *tbl, long x, long y, long cx, long cy) {
    const pptx_theme_t *T = w->t; chart_t c; if (!parse(tbl, &c)) return;
    const char *colors[MAXS] = { T->accent, T->c2, T->c3, T->c4 };
    ds4_buf b = {0};
    ds4_buf_puts(&b, XML_HEAD "<c:chartSpace " C_NS " " A_NS_DECL " " R_NS_DECL "><c:roundedCorners val=\"0\"/><c:chart><c:autoTitleDeleted val=\"1\"/><c:plotArea><c:layout/>");
    if (c.kind == 2) {
        ds4_buf_puts(&b, "<c:pieChart><c:varyColors val=\"1\"/><c:ser><c:idx val=\"0\"/><c:order val=\"0\"/><c:tx><c:v>"); office_xml_text(&b, c.ser[0]); ds4_buf_puts(&b, "</c:v></c:tx>");
        for (int i = 0; i < c.nc; i++) ds4_buf_printf(&b, "<c:dPt><c:idx val=\"%d\"/><c:bubble3D val=\"0\"/><c:spPr><a:solidFill><a:srgbClr val=\"%s\"/></a:solidFill><a:ln w=\"19050\"><a:solidFill><a:srgbClr val=\"%s\"/></a:solidFill></a:ln></c:spPr></c:dPt>", i, colors[i % MAXS], T->bg);
        dlbls(&b, T, "bestFit"); data(&b, &c, 0);
        ds4_buf_puts(&b, "</c:ser><c:firstSliceAng val=\"0\"/></c:pieChart>");
    } else {
        ds4_buf_puts(&b, c.kind ? "<c:lineChart><c:grouping val=\"standard\"/><c:varyColors val=\"0\"/>" : "<c:barChart><c:barDir val=\"col\"/><c:grouping val=\"clustered\"/><c:varyColors val=\"0\"/>");
        for (int s = 0; s < c.ns; s++) {
            ds4_buf_printf(&b, "<c:ser><c:idx val=\"%d\"/><c:order val=\"%d\"/><c:tx><c:v>", s, s); office_xml_text(&b, c.ser[s]); ds4_buf_puts(&b, "</c:v></c:tx>");
            if (c.kind) ds4_buf_printf(&b, "<c:spPr><a:ln w=\"28575\" cap=\"rnd\"><a:solidFill><a:srgbClr val=\"%s\"/></a:solidFill><a:round/></a:ln></c:spPr><c:marker><c:symbol val=\"circle\"/><c:size val=\"7\"/><c:spPr><a:solidFill><a:srgbClr val=\"%s\"/></a:solidFill><a:ln><a:solidFill><a:srgbClr val=\"%s\"/></a:solidFill></a:ln></c:spPr></c:marker>", colors[s], colors[s], T->bg);
            else ds4_buf_printf(&b, "<c:spPr><a:solidFill><a:srgbClr val=\"%s\"/></a:solidFill></c:spPr><c:invertIfNegative val=\"0\"/>", colors[s]);
            dlbls(&b, T, c.kind ? "t" : "outEnd"); data(&b, &c, s);
            if (c.kind) ds4_buf_puts(&b, "<c:smooth val=\"0\"/>");
            ds4_buf_puts(&b, "</c:ser>");
        }
        if (!c.kind) ds4_buf_puts(&b, "<c:gapWidth val=\"80\"/><c:overlap val=\"-10\"/>"); else ds4_buf_puts(&b, "<c:marker val=\"1\"/>");
        ds4_buf_puts(&b, "<c:axId val=\"1\"/><c:axId val=\"2\"/>"); ds4_buf_puts(&b, c.kind ? "</c:lineChart>" : "</c:barChart>");
        axes(&b, T);
    }
    ds4_buf_puts(&b, "<c:spPr><a:noFill/><a:ln><a:noFill/></a:ln></c:spPr></c:plotArea>");
    if (c.ns > 1 || c.kind == 2) { ds4_buf_puts(&b, "<c:legend><c:legendPos val=\"b\"/><c:overlay val=\"0\"/>"); txpr(&b, T->text, 1200); ds4_buf_puts(&b, "</c:legend>"); }
    ds4_buf_puts(&b, "<c:plotVisOnly val=\"1\"/><c:dispBlanksAs val=\"gap\"/></c:chart><c:spPr><a:noFill/><a:ln><a:noFill/></a:ln></c:spPr>"); txpr(&b, T->text, 1200); ds4_buf_puts(&b, "</c:chartSpace>");
    w->nchart++; w->rid++;
    char part[80]; snprintf(part, sizeof part, "ppt/charts/chart%d.xml", w->nchart); office_zip_add(w->zip, part, b.ptr, b.len); ds4_buf_free(&b);
    ds4_buf_printf(&w->rels, "<Relationship Id=\"rId%d\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart\" Target=\"../charts/chart%d.xml\"/>", w->rid, w->nchart);
    ds4_buf_printf(&w->xml, "<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id=\"%d\" name=\"Chart %d\"/><p:cNvGraphicFramePr><a:graphicFrameLocks noGrp=\"1\"/></p:cNvGraphicFramePr><p:nvPr/></p:nvGraphicFramePr>"
                            "<p:xfrm><a:off x=\"%ld\" y=\"%ld\"/><a:ext cx=\"%ld\" cy=\"%ld\"/></p:xfrm><a:graphic><a:graphicData uri=\"http://schemas.openxmlformats.org/drawingml/2006/chart\"><c:chart " C_NS " " R_NS_DECL " r:id=\"rId%d\"/></a:graphicData></a:graphic></p:graphicFrame>",
                   w->nid, w->nchart, x, y, cx, cy, w->rid);
    w->nid++;
}
