/* office_pptx.c — 文档树 → 幻灯片: 按 --- 切片, 每片按内容判版式(绘图原语在 office_pptx_draw.c, 包骨架在 office_pptx_pkg.c)。
 * 版式只看内容, 不认任何标记(技能 skills/pptx/SKILL.md 教模型怎么写出每种版式):
 *   封面   第一页, 一个标题 + 至多一段副标题          大字居中, arcs 主题画圆弧; 带一张图 → 图铺满右 45%(裁满), 字靠左
 *   章节   只有一个标题                               左上大序号 + 标题, 和封面同款圆弧; 带图同封面
 *   数字卡 标题 + 一张"表头 + 一行数据"的表(2~5 列)    一列一张卡片: 大数字 + 小标签。要点或表格都讲不好"三个关键数"
 *   卡片格 标题 + 2~6 条每条以 **加粗** 开头的要点    每条一张卡片(序号球 + 加粗引题 + 说明), 一行或两行排开; 加粗前面放 ![](图标.png) → 淡色圆里的图标代替序号球
 *   双栏   标题 + 两个 ### 小标题                     左右两张面板, 各自装小标题下面的内容(对比 / 前后 / 利弊)
 *   金句   标题可有可无 + 只有一段 > 引用             大引号 + 大字斜体
 *   内容   其它                                       标题在上; 有图 → 字左图右; 有表 → 表占正文区(有字时字在上)
 * 10-18 改版依据(用户: 样式简单/布局粗糙/配色不行; 对照 Anthropic pptx 技能的 Design Ideas): 正文 20→24 磅; 去掉标题下划线与左侧竖条(那是"AI 生成"的标志性元素);
 * 加粗走强调色; 表格去竖线、隔行浅底; 每种版式都给一块面板色, 没有图也不是白纸黑字。
 * 10-18 二期(用户: "动画、渐变等等样式体验需要优化"): 每页 bg→bg2 渐变底; 封面/章节页光晕; 卡片阴影(深色主题描边); 每页淡入切换; 卡片/数字卡/双栏逐个淡入(office_pptx_draw.c pptx_timing);
 * 表头第一格 bar/line/pie 的表 → 原生图表(office_pptx_chart.c), 排法同图片(字左图右 / 整页)。
 * <!-- --> 注释 = 讲稿备注(notesSlide); 图片只认 png/jpeg, 读不到的图在正文里写一行并记进 report。 */
#include "office_internal.h"
#include <stdio.h>

#define A_NS "xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\""
#define R_NS "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\""
#define P_NS "xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\""
#define GAP 300000   /* 卡片 / 栏之间 0.33 英寸, 全篇一个值 */

/* 封面/章节页: 右下两道同心圆弧(出血到页外), 全篇唯一的装饰母题 */
static void arcs(pptx_sw *w) {
    if (!w->t->arcs) return;
    pptx_glow(w, PPTX_W - 4200000, PPTX_H - 4200000, 7000000, w->t->accent, w->t->dark ? 26 : 14);
    pptx_shape(w, "ellipse", PPTX_W - 3600000, PPTX_H - 3600000, 5400000, 5400000, NULL, 100, w->t->accent, 50800, 30, NULL);
    pptx_shape(w, "ellipse", PPTX_W - 2200000, PPTX_H - 2200000, 3200000, 3200000, NULL, 100, w->t->accent, 25400, 55, NULL);
}
static void page_number(pptx_sw *w) {
    char num[16]; snprintf(num, sizeof num, "%d", w->slide); ds4_buf p = {0};
    pptx_para_str(&p, w->t, num, 1000, false, w->t->muted, "r");
    pptx_text_box(w, PPTX_W - PPTX_MARGIN - 900000, PPTX_H - 420000, 900000, 300000, "b", &p, NULL);
    ds4_buf_free(&p);
}
static void title_box(pptx_sw *w, const office_node *title) {
    ds4_buf p = {0}; pptx_para(&p, w->t, title->child, 3600, true, false, w->t->title, 0, NULL, NULL);
    pptx_text_box(w, PPTX_MARGIN, PPTX_TITLE_Y, PPTX_BODY_W, PPTX_TITLE_H, "ctr", &p, NULL); ds4_buf_free(&p);
}
/* 数字卡: 表头一行 = 标签, 数据一行 = 数字; lead = 卡片上面的一句话(可无) */
static void stat_cards(pptx_sw *w, const office_node *tbl, const office_node *lead) {
    const pptx_theme_t *T = w->t; ds4_buf p = {0}; long y = PPTX_BODY_Y;
    if (lead) { pptx_para(&p, T, lead->child, 2000, false, false, NULL, 0, NULL, NULL); pptx_text_box(w, PPTX_MARGIN, y, PPTX_BODY_W, 700000, "t", &p, NULL); p.len = 0; y += 900000; }
    const office_node *hr = tbl->child, *dr = hr ? hr->next : NULL; const int n = tbl->ncols;
    const long cw = (PPTX_BODY_W - GAP * (n - 1)) / n, ch = PPTX_BODY_Y + PPTX_BODY_H - y, mid = y + ch * 52 / 100;   /* 卡片撑满正文区, 数字压在卡片中线上 */
    const office_node *hc = hr->child, *dc = dr->child;
    for (int i = 0; i < n && hc && dc; i++, hc = hc->next, dc = dc->next) {
        const long x = PPTX_MARGIN + i * (cw + GAP);
        pptx_anim(w, pptx_card(w, x, y, cw, ch), true);
        pptx_para(&p, T, dc->child, n > 4 ? 3600 : 4400, true, false, T->accent, 0, NULL, "ctr"); pptx_anim(w, pptx_text_box(w, x + 100000, mid - 1500000, cw - 200000, 1500000, "b", &p, NULL), false); p.len = 0;
        pptx_para(&p, T, hc->child, 1800, false, false, T->muted, 0, NULL, "ctr"); pptx_anim(w, pptx_text_box(w, x + 100000, mid + 100000, cw - 200000, 800000, "t", &p, NULL), false); p.len = 0;
    }
    ds4_buf_free(&p);
}
static const office_node *card_head(const office_node *it, const office_node **icon);
/* 卡片格: 每条要点一张卡; 引题 = 第一段加粗文字, 其余行内节点是说明 */
static void card_grid(pptx_sw *w, const office_node *ls, const office_node *lead) {
    const pptx_theme_t *T = w->t; ds4_buf p = {0}; long y = PPTX_BODY_Y, h = PPTX_BODY_H;
    if (lead) { pptx_para(&p, T, lead->child, 2000, false, false, NULL, 0, NULL, NULL); pptx_text_box(w, PPTX_MARGIN, y, PPTX_BODY_W, 700000, "t", &p, NULL); p.len = 0; y += 900000; h -= 900000; }
    const int n = office_count(ls, OFF_ITEM), cols = n <= 3 ? n : n == 4 ? 2 : 3, rows = (n + cols - 1) / cols;
    const long cw = (PPTX_BODY_W - GAP * (cols - 1)) / cols, ch = (h - GAP * (rows - 1)) / rows, pad = 220000, ball = 420000;
    int k = 0; char num[16];
    for (const office_node *it = ls->child; it; it = it->next, k++) {
        const long x = PPTX_MARGIN + (k % cols) * (cw + GAP), cy = y + (k / cols) * (ch + GAP);
        const office_node *icon, *first = card_head(it, &icon);
        pptx_anim(w, pptx_card(w, x, cy, cw, ch), true);
        if (icon) {   /* 淡色圆 + 图标(fetch_image 拿的单色图标, 颜色由模型按主题给): "图标在彩色圆里"是 Design Ideas 点名的母题 */
            const long ring = ball + 140000; pptx_anim(w, pptx_shape(w, "ellipse", x + pad, cy + pad, ring, ring, T->accent, w->t->dark ? 22 : 14, NULL, 0, 100, NULL), false);
            const int pid = pptx_picture(w, icon, x + pad + ring / 5, cy + pad + ring / 5, ring * 3 / 5, ring * 3 / 5, false); if (pid) pptx_anim(w, pid, false);
        } else {
            snprintf(num, sizeof num, "%d", k + 1); pptx_para_str(&p, T, num, 1600, true, T->bg, "ctr");
            pptx_anim(w, pptx_shape(w, "ellipse", x + pad, cy + pad, ball, ball, T->accent, 100, NULL, 0, 100, &p), false); p.len = 0;
        }
        const long top = ball + (icon ? 140000 : 0);
        office_node head = *first; head.next = NULL;
        pptx_para(&p, T, &head, rows > 1 ? 2000 : 2200, true, false, T->title, 0, NULL, NULL);
        if (first->next) pptx_para(&p, T, first->next, rows > 1 ? 1600 : 1800, false, false, T->text, 0, NULL, NULL);
        pptx_anim(w, pptx_text_box(w, x + pad - 91440, cy + pad + top + 60000, cw - 2 * pad + 182880, ch - pad - top - 120000, "t", &p, NULL), false); p.len = 0;
    }
    ds4_buf_free(&p);
}
/* 双栏: h3a..h3b 是左栏, h3b..末尾是右栏; 每栏一块面板, 小标题用强调色 */
static void two_columns(pptx_sw *w, const office_node *h3a, const office_node *h3b, const office_node *title) {
    const pptx_theme_t *T = w->t; ds4_buf p = {0}; const long cw = (PPTX_BODY_W - GAP) / 2, pad = 250000;
    for (int i = 0; i < 2; i++) {
        const office_node *h = i ? h3b : h3a, *stop = i ? NULL : h3b; const long x = PPTX_MARGIN + i * (cw + GAP);
        pptx_anim(w, pptx_card(w, x, PPTX_BODY_Y, cw, PPTX_BODY_H), true);
        pptx_para(&p, T, h->child, 2400, true, false, T->accent, 0, NULL, NULL);
        pptx_body_paras(&p, T, h->next, stop, 2000, title);
        pptx_anim(w, pptx_text_box(w, x + pad - 91440, PPTX_BODY_Y + pad, cw - 2 * pad + 182880, PPTX_BODY_H - 2 * pad, "t", &p, NULL), false); p.len = 0;
    }
    ds4_buf_free(&p);
}
static void quote_page(pptx_sw *w, const office_node *q, bool has_title) {
    const pptx_theme_t *T = w->t; ds4_buf p = {0}; const long y = has_title ? PPTX_BODY_Y : PPTX_TITLE_Y + 600000, h = PPTX_H - y - 900000;
    pptx_para_str(&p, T, "\xE2\x80\x9C", 12000, true, T->accent, NULL);   /* 大引号当唯一的装饰 */
    pptx_text_box(w, PPTX_MARGIN - 100000, y - 300000, 1400000, 1800000, "t", &p, NULL); p.len = 0;
    pptx_para(&p, T, q->child, 2800, false, true, T->title, 0, NULL, NULL);
    pptx_text_box(w, PPTX_MARGIN + 900000, y + 300000, PPTX_BODY_W - 1800000, h - 300000, "ctr", &p, NULL);
    ds4_buf_free(&p);
}
/* 分类: 一页里有什么; 版式判定在 slide() */
typedef struct { const office_node *title, *img, *tbl, *quote, *list, *para, *h3a, *h3b, *chart; int others, h3, paras, lists; } parts;
static parts classify(const office_node *blocks) {
    parts c = {0};
    for (const office_node *x = blocks; x; x = x->next) {
        if (x->kind == OFF_HEADING && !c.title && x->level <= 2) c.title = x;
        else if (x->kind == OFF_HEADING && x->level == 3) { if (!c.h3a) c.h3a = x; else if (!c.h3b) c.h3b = x; c.h3++; c.others++; }
        else if (x->kind == OFF_IMAGE && !c.img) c.img = x;
        else if (x->kind == OFF_TABLE && !c.chart && !c.img && !c.tbl && pptx_chart_ok(x)) c.chart = x;   /* 表头第一格 bar/line/pie = 图表; 和图片同一个位子, 一页只认一个 */
        else if (x->kind == OFF_TABLE && !c.tbl) c.tbl = x;
        else if (x->kind == OFF_NOTES || x->kind == OFF_HR) continue;
        else { c.others++; if (x->kind == OFF_QUOTE) c.quote = x; else if (x->kind == OFF_PARA) { c.paras++; if (!c.para) c.para = x; } else if (x->kind == OFF_LIST) { c.lists++; if (!c.list) c.list = x; } }
    }
    return c;
}
/* 卡片的引题 = 条目第一个行内节点; 前面可以先有一张图标图(和它后面那个只有空格的文本节点: "![](x.png) **引题**" 解析出来中间夹一个 " ") */
static const office_node *card_head(const office_node *it, const office_node **icon) {
    const office_node *f = it->child; *icon = NULL;
    if (f && f->kind == OFF_IMAGE) { *icon = f; f = f->next; if (f && f->kind == OFF_TEXT && !f->bold && f->text && !f->text[strspn(f->text, " \t")]) f = f->next; }
    return f;
}
static bool card_list(const office_node *ls) {   /* 2~6 条, 每条以加粗开头(前面可以先放一张图标图), 不嵌套 */
    const int n = office_count(ls, OFF_ITEM); if (n < 2 || n > 6 || office_count(ls, OFF_LIST) > 1) return false;
    for (const office_node *it = ls->child; it; it = it->next) {
        const office_node *icon, *f = card_head(it, &icon);
        if (!f || f->kind != OFF_TEXT || !f->bold || !f->text || !f->text[0]) return false;
    }
    return true;
}
/* 封面 / 章节页的照片: 铺满右 45%, 整页高, 裁满; 左边留给字 */
static long photo_half(pptx_sw *w, const office_node *img) {
    const long px = PPTX_W * 55 / 100;
    pptx_picture(w, img, px, 0, PPTX_W - px, PPTX_H, true);
    return px - PPTX_MARGIN - GAP;   /* 字的宽度 */
}

static void slide(pptx_sw *w, const office_node *blocks, bool first) {
    const pptx_theme_t *T = w->t; const parts c = classify(blocks); ds4_buf notes = {0};
    for (const office_node *x = blocks; x; x = x->next) if (x->kind == OFF_NOTES) { if (notes.len) ds4_buf_putc(&notes, '\n'); ds4_buf_puts(&notes, x->text ? x->text : ""); }
    w->xml.len = 0; w->rels.len = 0; w->nid = 2; w->rid = 1; w->nanim = 0; w->grp = 0;
    ds4_buf_puts(&w->xml, XML_HEAD "<p:sld " A_NS " " R_NS " " P_NS "><p:cSld>");
    pptx_bg(w);
    ds4_buf_puts(&w->xml, "<p:spTree><p:nvGrpSpPr><p:cNvPr id=\"1\" name=\"\"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/><a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>");
    ds4_buf_puts(&w->rels, XML_HEAD "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout\" Target=\"../slideLayouts/slideLayout1.xml\"/>");
    ds4_buf p = {0};
    const bool plain = c.title && !c.img && !c.tbl && !c.chart, words = c.title && !c.tbl && !c.chart;   /* plain = 只有字; words = 字 + 至多一张图 */
    if (words && first && c.others <= 1 && (!c.others || c.paras == 1)) {   /* 封面: 有图 → 右半幅照片, 字靠左; 没图 → 居中 + 圆弧 */
        const long tw = c.img ? photo_half(w, c.img) : PPTX_BODY_W; const char *al = c.img ? NULL : "ctr";
        if (!c.img) arcs(w);
        pptx_para(&p, T, c.title->child, 4800, true, false, T->title, 0, NULL, al); pptx_text_box(w, PPTX_MARGIN, 1900000, tw, 1700000, "b", &p, NULL); p.len = 0;
        if (c.para) { pptx_para(&p, T, c.para->child, 2400, false, false, T->muted, 0, NULL, al); pptx_text_box(w, PPTX_MARGIN, 3750000, tw, 1200000, "t", &p, NULL); p.len = 0; }
    } else if (words && !c.others) {   /* 章节页 */
        const long tw = c.img ? photo_half(w, c.img) : PPTX_BODY_W - 2500000;
        if (!c.img) arcs(w);
        w->nsec++; char num[16]; snprintf(num, sizeof num, "%02d", w->nsec);
        pptx_para_str(&p, T, num, 9600, true, T->accent, NULL); pptx_text_box(w, PPTX_MARGIN - 91440, 1300000, 4000000, 1900000, "b", &p, NULL); p.len = 0;
        pptx_para(&p, T, c.title->child, 4000, true, false, T->title, 0, NULL, NULL); pptx_text_box(w, PPTX_MARGIN - 91440, 3300000, tw, 1600000, "t", &p, NULL); p.len = 0;
        if (!c.img) page_number(w);
    } else if (c.title && c.tbl && !c.img && !c.chart && office_count(c.tbl, OFF_ROW) == 2 && c.tbl->ncols >= 2 && c.tbl->ncols <= 5 && c.others == c.paras && c.paras <= 1) {   /* 数字卡 */
        title_box(w, c.title); stat_cards(w, c.tbl, c.para); page_number(w);
    } else if (plain && c.lists == 1 && c.others == c.lists + c.paras && c.paras <= 1 && card_list(c.list)) {   /* 卡片格 */
        title_box(w, c.title); card_grid(w, c.list, c.para); page_number(w);
    } else if (!c.img && !c.tbl && !c.chart && c.h3 == 2 && c.h3a->next && c.h3a->next != c.h3b && c.h3b->next) {   /* 双栏 */
        if (c.title) title_box(w, c.title);
        two_columns(w, c.h3a, c.h3b, c.title); page_number(w);
    } else if (!c.img && !c.tbl && !c.chart && c.quote && c.others == 1) {   /* 金句 */
        if (c.title) title_box(w, c.title);
        quote_page(w, c.quote, c.title != NULL); page_number(w);
    } else {   /* 内容页 */
        if (c.title) title_box(w, c.title);
        const long by = c.title ? PPTX_BODY_Y : PPTX_TITLE_Y, bh = c.title ? PPTX_BODY_H : PPTX_BODY_H + PPTX_BODY_Y - PPTX_TITLE_Y;
        const office_node *vis = c.img ? c.img : c.chart;   /* 图或图表: 右边 58%; 没字就整页 */
        const long tw = vis ? PPTX_BODY_W * 42 / 100 : PPTX_BODY_W, iw = PPTX_BODY_W - tw - GAP;
        const int sz = c.others > 7 ? 1800 : c.others > 4 ? 2000 : 2400;
        const int np = pptx_body_paras(&p, T, blocks, NULL, sz, c.title);
        const char *anchor = c.others <= 4 ? "ctr" : "t";   /* 三五条要点垂直居中(水平仍左对齐): 顶着标题挤在左上, 下面六成空白(10-18 实看) */
        if (c.tbl) {
            const long th = np > 1 ? bh / 3 : 800000;   /* 一句引言只占一行高, 表紧跟其后 */
            if (np) { pptx_text_box(w, PPTX_MARGIN, by, tw, th, "t", &p, NULL); pptx_table(w, c.tbl, PPTX_MARGIN, by + th + 100000, tw, bh - th - 100000, false); }
            else pptx_table(w, c.tbl, PPTX_MARGIN, by, tw, bh, true);
        } else if (np || !vis) pptx_anim(w, pptx_text_box(w, PPTX_MARGIN, by, tw, bh, anchor, &p, NULL), true);
        p.len = 0;
        if (vis) { const bool side = np || c.tbl; const long vx = side ? PPTX_MARGIN + tw + GAP : PPTX_MARGIN, vw = side ? iw : PPTX_BODY_W;
            if (c.img) pptx_picture(w, c.img, vx, by, vw, bh, false); else pptx_chart(w, c.chart, vx, by, vw, bh); }
        page_number(w);
    }
    ds4_buf_free(&p);
    ds4_buf_puts(&w->xml, "</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>"); pptx_timing(w); ds4_buf_puts(&w->xml, "</p:sld>");
    if (notes.len) { w->rid++; ds4_buf_printf(&w->rels, "<Relationship Id=\"rId%d\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide\" Target=\"../notesSlides/notesSlide%d.xml\"/>", w->rid, w->slide); }
    ds4_buf_puts(&w->rels, "</Relationships>");
    char name[64]; snprintf(name, sizeof name, "ppt/slides/slide%d.xml", w->slide); office_zip_add(w->zip, name, w->xml.ptr, w->xml.len);
    snprintf(name, sizeof name, "ppt/slides/_rels/slide%d.xml.rels", w->slide); office_zip_add(w->zip, name, w->rels.ptr, w->rels.len);
    if (notes.len) {
        ds4_buf nb = {0}, nr = {0}; pptx_notes_slide(&nb, &nr, w->slide, notes.ptr);
        snprintf(name, sizeof name, "ppt/notesSlides/notesSlide%d.xml", w->slide); office_zip_add(w->zip, name, nb.ptr, nb.len);
        snprintf(name, sizeof name, "ppt/notesSlides/_rels/notesSlide%d.xml.rels", w->slide); office_zip_add(w->zip, name, nr.ptr, nr.len);
        ds4_buf_free(&nb); ds4_buf_free(&nr);
    }
    ds4_buf_free(&notes);
}

uint8_t *office_pptx_build(const office_node *doc, const office_opts *o, office_report *rep, size_t *len) {
    office_zip z; office_zip_begin(&z, o->now);
    const pptx_theme_t *T = pptx_theme_find(o->theme); if (!T) T = pptx_theme_find(NULL);   /* 不认识的名字调用方该先拦(tool_office.c); 这里退回 clean 而不是炸 */
    pptx_sw w = { &z, o, rep, T, {0}, {0}, 0, 0, 2, 1, 0, 0, false, false, {0}, {0}, 0, 0 };
    /* 按 --- 切片: 把链表临时断开成每片一段 */
    office_node *head = doc->child; bool any_notes = false;
    while (head) {
        office_node *cut = head, *prev = NULL; int n = 0;
        while (cut && cut->kind != OFF_HR) { if (cut->kind == OFF_NOTES) any_notes = true; prev = cut; cut = cut->next; n++; }
        office_node *next = cut ? cut->next : NULL;
        if (prev) prev->next = NULL;
        if (n) { w.slide++; slide(&w, head, w.slide == 1); }
        if (prev) prev->next = cut;
        head = next;
    }
    if (!w.slide) { w.slide = 1; office_node empty = { OFF_PARA, 0, 0, 0, 0, 0, 0, NULL, NULL, 0, NULL, NULL }; slide(&w, &empty, true); }
    ds4_buf b = {0}, r = {0};
    pptx_content_types(&b, w.slide, any_notes, w.png, w.jpeg, w.nchart); office_zip_add(&z, "[Content_Types].xml", b.ptr, b.len); b.len = 0;
    pptx_root_rels(&b); office_zip_add(&z, "_rels/.rels", b.ptr, b.len); b.len = 0;
    pptx_presentation(&b, &r, w.slide, any_notes); office_zip_add(&z, "ppt/presentation.xml", b.ptr, b.len); office_zip_add(&z, "ppt/_rels/presentation.xml.rels", r.ptr, r.len); b.len = 0; r.len = 0;
    ds4_buf pp = {0}, vp = {0}, ts = {0}; pptx_small_parts(&pp, &vp, &ts);
    office_zip_add(&z, "ppt/presProps.xml", pp.ptr, pp.len); office_zip_add(&z, "ppt/viewProps.xml", vp.ptr, vp.len); office_zip_add(&z, "ppt/tableStyles.xml", ts.ptr, ts.len);
    ds4_buf_free(&pp); ds4_buf_free(&vp); ds4_buf_free(&ts);
    pptx_theme(&b, "DwarfStar", T); office_zip_add(&z, "ppt/theme/theme1.xml", b.ptr, b.len); b.len = 0;
    pptx_master(&b, &r); office_zip_add(&z, "ppt/slideMasters/slideMaster1.xml", b.ptr, b.len); office_zip_add(&z, "ppt/slideMasters/_rels/slideMaster1.xml.rels", r.ptr, r.len); b.len = 0; r.len = 0;
    pptx_layout(&b, &r); office_zip_add(&z, "ppt/slideLayouts/slideLayout1.xml", b.ptr, b.len); office_zip_add(&z, "ppt/slideLayouts/_rels/slideLayout1.xml.rels", r.ptr, r.len); b.len = 0; r.len = 0;
    if (any_notes) {
        pptx_theme(&b, "DwarfStar Notes", pptx_theme_find(NULL)); office_zip_add(&z, "ppt/theme/theme2.xml", b.ptr, b.len); b.len = 0;   /* 备注页永远白底: 深色主题的浅字在白色备注页上看不见 */
        pptx_notes_master(&b, &r); office_zip_add(&z, "ppt/notesMasters/notesMaster1.xml", b.ptr, b.len); office_zip_add(&z, "ppt/notesMasters/_rels/notesMaster1.xml.rels", r.ptr, r.len); b.len = 0; r.len = 0;
    }
    char *title = o->title ? strdup(o->title) : office_first_heading(doc);
    pptx_props(&b, &r, title ? title : "", w.slide, o->now); office_zip_add(&z, "docProps/core.xml", b.ptr, b.len); office_zip_add(&z, "docProps/app.xml", r.ptr, r.len); free(title);
    ds4_buf_free(&b); ds4_buf_free(&r); ds4_buf_free(&w.xml); ds4_buf_free(&w.rels);
    return office_zip_end(&z, len);
}
