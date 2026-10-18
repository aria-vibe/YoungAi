/* office_pptx_draw.c — pptx 绘图原语: 段落(a:p)/文本框/装饰形状/图片/表格/列表与正文段。版式判定与页面编排在 office_pptx.c。
 * 文本框显式定位(不靠占位符): WPS / Keynote / LibreOffice 打开位置一致。
 * 段落间距跟字号走(spcBef = 字号的一半): 要点 24 磅配 12 磅段距, 表格 16 磅配 8 磅; 写死 6 磅时大字挤成一团(10-18 实看)。 */
#include "office_internal.h"
#include <stdio.h>

static void run_props(ds4_buf *b, int sz, bool bold, bool italic, bool code, const char *color) {
    ds4_buf_printf(b, "<a:rPr lang=\"zh-CN\" sz=\"%d\"%s%s>", sz, bold ? " b=\"1\"" : "", italic ? " i=\"1\"" : "");
    if (color) ds4_buf_printf(b, "<a:solidFill><a:srgbClr val=\"%s\"/></a:solidFill>", color);
    if (code) ds4_buf_puts(b, "<a:latin typeface=\"Consolas\"/><a:ea typeface=\"Microsoft YaHei\"/>");
    ds4_buf_puts(b, "</a:rPr>");
}
/* 每个 run 都显式写颜色: 不写的话靠母版 clrMap 推 tx1, LibreOffice / PowerPoint 推得对, 手机上的 WPS / 微信预览推成黑字, 深色主题整页正文黑在深蓝底上(10-18 用户实看) */
static void inlines(ds4_buf *b, const pptx_theme_t *t, const office_node *n, int sz, bool bold, bool italic, const char *color) {
    if (!color) color = t->text;
    for (; n; n = n->next) {
        if (n->kind == OFF_TEXT || n->kind == OFF_LINK) {
            ds4_buf_puts(b, "<a:r>"); run_props(b, sz, bold || n->bold, italic || n->italic, n->code, n->kind == OFF_LINK ? t->link : n->bold && !bold ? t->accent : color);   /* 正文里的 **加粗** 上强调色: 一页最多两处(技能里定), 正好当"重点数字" */
            ds4_buf_puts(b, "<a:t>"); office_xml_text(b, n->text ? n->text : ""); ds4_buf_puts(b, "</a:t></a:r>");
        } else if (n->kind == OFF_IMAGE) { ds4_buf_puts(b, "<a:r>"); run_props(b, sz, false, true, false, color); ds4_buf_puts(b, "<a:t>["); office_xml_text(b, n->text ? n->text : "image"); ds4_buf_puts(b, "]</a:t></a:r>"); }
        else if (n->kind == OFF_BREAK) ds4_buf_puts(b, "<a:br/>");
    }
}
void pptx_para(ds4_buf *b, const pptx_theme_t *t, const office_node *inl, int sz, bool bold, bool italic, const char *color, int lvl, const char *bullet, const char *align) {
    ds4_buf_puts(b, "<a:p><a:pPr");
    if (align) ds4_buf_printf(b, " algn=\"%s\"", align);
    if (bullet) ds4_buf_printf(b, " marL=\"%d\" indent=\"-285750\" lvl=\"%d\"", 342900 + 457200 * lvl, lvl);
    ds4_buf_printf(b, "><a:spcBef><a:spcPts val=\"%d\"/></a:spcBef>", sz / 2);
    if (bullet) ds4_buf_printf(b, "<a:buFont typeface=\"Arial\"/><a:buChar char=\"%s\"/>", bullet); else ds4_buf_puts(b, "<a:buNone/>");
    ds4_buf_puts(b, "</a:pPr>");
    if (inl) inlines(b, t, inl, sz, bold, italic, color);
    else ds4_buf_printf(b, "<a:endParaRPr lang=\"zh-CN\" sz=\"%d\"/>", sz);
    ds4_buf_puts(b, "</a:p>");
}
/* 一行纯文本的段落(页码/序号/卡片标签这类不来自文档树的字) */
void pptx_para_str(ds4_buf *b, const pptx_theme_t *t, const char *s, int sz, bool bold, const char *color, const char *align) {
    office_node n = { OFF_TEXT, 0, 0, 0, 0, 0, 0, (char *)s, NULL, 0, NULL, NULL };
    pptx_para(b, t, &n, sz, bold, false, color, 0, NULL, align);
}
int pptx_text_box(pptx_sw *w, long x, long y, long cx, long cy, const char *anchor, const ds4_buf *paras, const char *fill) {
    const int id = w->nid++;
    ds4_buf_printf(&w->xml, "<p:sp><p:nvSpPr><p:cNvPr id=\"%d\" name=\"Text %d\"/><p:cNvSpPr txBox=\"1\"/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x=\"%ld\" y=\"%ld\"/><a:ext cx=\"%ld\" cy=\"%ld\"/></a:xfrm><a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom>", id, id, x, y, cx, cy);
    if (fill) ds4_buf_printf(&w->xml, "<a:solidFill><a:srgbClr val=\"%s\"/></a:solidFill>", fill);
    ds4_buf_printf(&w->xml, "</p:spPr><p:txBody><a:bodyPr wrap=\"square\" anchor=\"%s\" lIns=\"91440\" rIns=\"91440\"><a:normAutofit/></a:bodyPr><a:lstStyle/>", anchor);
    ds4_buf_append(&w->xml, paras->ptr, paras->len);
    ds4_buf_puts(&w->xml, "</p:txBody></p:sp>");
    return id;
}
/* 形状: 矩形 / 圆角矩形 / 椭圆, 填充与描边各带透明度(百分比, 100 = 不透明); paras 非空就把字写进形状里(卡片序号球) */
int pptx_shape(pptx_sw *w, const char *prst, long x, long y, long cx, long cy, const char *fill, int falpha, const char *line, long lw, int lalpha, const ds4_buf *paras) {
    const int id = w->nid++;
    ds4_buf_printf(&w->xml, "<p:sp><p:nvSpPr><p:cNvPr id=\"%d\" name=\"Shape %d\"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x=\"%ld\" y=\"%ld\"/><a:ext cx=\"%ld\" cy=\"%ld\"/></a:xfrm><a:prstGeom prst=\"%s\">", id, id, x, y, cx, cy, prst);
    if (!strcmp(prst, "roundRect")) ds4_buf_puts(&w->xml, "<a:avLst><a:gd name=\"adj\" fmla=\"val 8000\"/></a:avLst>"); else ds4_buf_puts(&w->xml, "<a:avLst/>");   /* 圆角小一点(默认 16.7% 像药丸) */
    ds4_buf_puts(&w->xml, "</a:prstGeom>");
    if (fill) { ds4_buf_printf(&w->xml, "<a:solidFill><a:srgbClr val=\"%s\">", fill); if (falpha < 100) ds4_buf_printf(&w->xml, "<a:alpha val=\"%d\"/>", falpha * 1000); ds4_buf_puts(&w->xml, "</a:srgbClr></a:solidFill>"); }
    else ds4_buf_puts(&w->xml, "<a:noFill/>");
    if (line) { ds4_buf_printf(&w->xml, "<a:ln w=\"%ld\"><a:solidFill><a:srgbClr val=\"%s\">", lw, line); if (lalpha < 100) ds4_buf_printf(&w->xml, "<a:alpha val=\"%d\"/>", lalpha * 1000); ds4_buf_puts(&w->xml, "</a:srgbClr></a:solidFill></a:ln>"); }
    else ds4_buf_puts(&w->xml, "<a:ln><a:noFill/></a:ln>");
    if (fill && !strcmp(prst, "roundRect") && !w->t->dark) ds4_buf_puts(&w->xml, "<a:effectLst><a:outerShdw blurRad=\"76200\" dist=\"25400\" dir=\"5400000\" algn=\"t\" rotWithShape=\"0\"><a:srgbClr val=\"000000\"><a:alpha val=\"14000\"/></a:srgbClr></a:outerShdw></a:effectLst>");   /* 浅色主题的卡片: 柔和下投影, 和底色分开 */
    ds4_buf_puts(&w->xml, "</p:spPr><p:txBody><a:bodyPr anchor=\"ctr\" lIns=\"0\" rIns=\"0\" tIns=\"0\" bIns=\"0\"/><a:lstStyle/>");
    if (paras && paras->len) ds4_buf_append(&w->xml, paras->ptr, paras->len); else ds4_buf_puts(&w->xml, "<a:p><a:endParaRPr lang=\"zh-CN\"/></a:p>");
    ds4_buf_puts(&w->xml, "</p:txBody></p:sp>");
    return id;
}
int pptx_card(pptx_sw *w, long x, long y, long cx, long cy) {
    return pptx_shape(w, "roundRect", x, y, cx, cy, w->t->panel, 100, w->t->dark ? w->t->accent : NULL, 9525, 22, NULL);   /* 深底上阴影看不见, 换强调色 22% 细描边 */
}
/* 光晕: 圆形径向渐变, 中心 alpha% → 边缘全透明; 封面/章节页右下角一团"光", 比两道圆弧多一层纵深 */
void pptx_glow(pptx_sw *w, long x, long y, long d, const char *color, int alpha) {
    const int id = w->nid++;
    ds4_buf_printf(&w->xml, "<p:sp><p:nvSpPr><p:cNvPr id=\"%d\" name=\"Glow %d\"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x=\"%ld\" y=\"%ld\"/><a:ext cx=\"%ld\" cy=\"%ld\"/></a:xfrm><a:prstGeom prst=\"ellipse\"><a:avLst/></a:prstGeom>"
                            "<a:gradFill rotWithShape=\"1\"><a:gsLst><a:gs pos=\"0\"><a:srgbClr val=\"%s\"><a:alpha val=\"%d\"/></a:srgbClr></a:gs><a:gs pos=\"100000\"><a:srgbClr val=\"%s\"><a:alpha val=\"0\"/></a:srgbClr></a:gs></a:gsLst>"
                            "<a:path path=\"circle\"><a:fillToRect l=\"50000\" t=\"50000\" r=\"50000\" b=\"50000\"/></a:path></a:gradFill><a:ln><a:noFill/></a:ln></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang=\"zh-CN\"/></a:p></p:txBody></p:sp>",
                   id, id, x, y, d, d, color, alpha * 1000, color);
}
/* 每页自己写底色(母版 bg1 在 WPS/Keynote 里不一定被尊重): 左上 bg → 右下 bg2 的 45° 线性渐变, 纯色底太"PPT 默认" */
void pptx_bg(pptx_sw *w) {
    ds4_buf_printf(&w->xml, "<p:bg><p:bgPr><a:gradFill rotWithShape=\"1\"><a:gsLst><a:gs pos=\"0\"><a:srgbClr val=\"%s\"/></a:gs><a:gs pos=\"100000\"><a:srgbClr val=\"%s\"/></a:gs></a:gsLst><a:lin ang=\"2700000\" scaled=\"0\"/></a:gradFill><a:effectLst/></p:bgPr></p:bg>", w->t->bg, w->t->bg2);
}
void pptx_anim(pptx_sw *w, int spid, bool new_group) {
    if (new_group) w->grp++;
    if (w->nanim < PPTX_ANIM_MAX) { w->anim_id[w->nanim] = spid; w->anim_grp[w->nanim] = w->grp; w->nanim++; }
}
/* p:transition + p:timing。结构照 PowerPoint 自己存的"第一个效果 = 上一动画之后(自动开始)": 主序列下一个点击组, 组的 stCondLst 带 onBegin 指回主序列(id 2)
 * 就不用点鼠标; 组里每个形状一个 par, 同组 delay 相同、下一组 +250 ms; 效果 = 先 set visibility 再 400 ms fade。bldLst 给每个形状一条 bldP(animBg: 连底色一起出现)。 */
void pptx_timing(pptx_sw *w) {
    ds4_buf_puts(&w->xml, "<p:transition spd=\"med\"><p:fade/></p:transition>");
    if (!w->nanim) return;
    int id = 3;
    ds4_buf_puts(&w->xml, "<p:timing><p:tnLst><p:par><p:cTn id=\"1\" dur=\"indefinite\" restart=\"never\" nodeType=\"tmRoot\"><p:childTnLst><p:seq concurrent=\"1\" nextAc=\"seek\"><p:cTn id=\"2\" dur=\"indefinite\" nodeType=\"mainSeq\"><p:childTnLst>");
    ds4_buf_printf(&w->xml, "<p:par><p:cTn id=\"%d\" fill=\"hold\"><p:stCondLst><p:cond delay=\"indefinite\"/><p:cond evt=\"onBegin\" delay=\"0\"><p:tn val=\"2\"/></p:cond></p:stCondLst><p:childTnLst>", id++);
    for (int i = 0; i < w->nanim; i++) {
        const bool first = i == 0 || w->anim_grp[i] != w->anim_grp[i - 1];
        ds4_buf_printf(&w->xml, "<p:par><p:cTn id=\"%d\" fill=\"hold\"><p:stCondLst><p:cond delay=\"%d\"/></p:stCondLst><p:childTnLst>", id++, (w->anim_grp[i] - w->anim_grp[0]) * 250);
        ds4_buf_printf(&w->xml, "<p:par><p:cTn id=\"%d\" presetID=\"10\" presetClass=\"entr\" presetSubtype=\"0\" fill=\"hold\" grpId=\"0\" nodeType=\"%s\"><p:stCondLst><p:cond delay=\"0\"/></p:stCondLst><p:childTnLst>", id++, first ? "afterEffect" : "withEffect");
        ds4_buf_printf(&w->xml, "<p:set><p:cBhvr><p:cTn id=\"%d\" dur=\"1\" fill=\"hold\"><p:stCondLst><p:cond delay=\"0\"/></p:stCondLst></p:cTn><p:tgtEl><p:spTgt spid=\"%d\"/></p:tgtEl><p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr><p:to><p:strVal val=\"visible\"/></p:to></p:set>", id++, w->anim_id[i]);
        ds4_buf_printf(&w->xml, "<p:animEffect transition=\"in\" filter=\"fade\"><p:cBhvr><p:cTn id=\"%d\" dur=\"400\"/><p:tgtEl><p:spTgt spid=\"%d\"/></p:tgtEl></p:cBhvr></p:animEffect>", id++, w->anim_id[i]);
        ds4_buf_puts(&w->xml, "</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>");
    }
    ds4_buf_puts(&w->xml, "</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn><p:prevCondLst><p:cond evt=\"onPrev\" delay=\"0\"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>"
                          "<p:nextCondLst><p:cond evt=\"onNext\" delay=\"0\"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq></p:childTnLst></p:cTn></p:par></p:tnLst><p:bldLst>");
    for (int i = 0; i < w->nanim; i++) ds4_buf_printf(&w->xml, "<p:bldP spid=\"%d\" grpId=\"0\" animBg=\"1\"/>", w->anim_id[i]);
    ds4_buf_puts(&w->xml, "</p:bldLst></p:timing>");
}
/* fill=false: 等比缩进框居中(示意图 / 图表不能裁); fill=true: 铺满框, 多出的部分从两边或上下对称裁掉(srcRect, 像素一个不丢, 只是不显示): 封面 / 章节页的半幅照片 */
int pptx_picture(pptx_sw *w, const office_node *im, long bx, long by, long bw, long bh, bool fill) {
    char path[4096]; office_join(w->o->base_dir, im->url, path, sizeof path);
    size_t n = 0; uint8_t *data = office_slurp(path, &n); office_imginfo info;
    if (!data || !office_img_info(data, n, &info)) { free(data); w->rep->missing_images++; ds4_buf_printf(&w->rep->notes, "image not usable (missing or not png/jpeg): %s\n", im->url); return 0; }
    w->rep->images++; w->nimg++; w->rid++;
    if (!strcmp(info.ext, "png")) w->png = true; else w->jpeg = true;
    char part[80]; snprintf(part, sizeof part, "ppt/media/image%d.%s", w->nimg, info.ext);
    office_zip_add(w->zip, part, data, n); free(data);
    ds4_buf_printf(&w->rels, "<Relationship Id=\"rId%d\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"../media/image%d.%s\"/>", w->rid, w->nimg, info.ext);
    const double sx = (double)bw / info.w, sy = (double)bh / info.h, s = sx < sy ? sx : sy;
    long cw = fill ? bw : (long)(info.w * s), ch = fill ? bh : (long)(info.h * s); char src[96] = "";
    if (fill) {   /* 图比框宽 → 裁左右; 比框高 → 裁上下; 千分比对称 */
        const double ia = (double)info.w / info.h, ba = (double)bw / bh;
        const long cut = (long)((1.0 - (ia > ba ? ba / ia : ia / ba)) * 50000);
        if (ia > ba) snprintf(src, sizeof src, "<a:srcRect l=\"%ld\" r=\"%ld\"/>", cut, cut); else snprintf(src, sizeof src, "<a:srcRect t=\"%ld\" b=\"%ld\"/>", cut, cut);
    }
    const int id = w->nid++;
    ds4_buf_printf(&w->xml, "<p:pic><p:nvPicPr><p:cNvPr id=\"%d\" name=\"Picture %d\" descr=\"", id, w->nimg); office_xml_text(&w->xml, im->text ? im->text : "");
    ds4_buf_printf(&w->xml, "\"/><p:cNvPicPr><a:picLocks noChangeAspect=\"1\"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed=\"rId%d\"/>%s<a:stretch><a:fillRect/></a:stretch></p:blipFill>"
                            "<p:spPr><a:xfrm><a:off x=\"%ld\" y=\"%ld\"/><a:ext cx=\"%ld\" cy=\"%ld\"/></a:xfrm><a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom></p:spPr></p:pic>", w->rid, src, bx + (bw - cw) / 2, by + (bh - ch) / 2, cw, ch);
    return id;
}
/* 表格: 表头实底, 正文隔行浅底, 只留横向细线(竖线和满框黑线是 Office 默认样式, 看着像 Excel 截图) */
void pptx_table(pptx_sw *w, const office_node *t, long x, long y, long cx, long cy, bool center) {
    const pptx_theme_t *T = w->t;
    const int nc = t->ncols > 0 ? t->ncols : 1, nr = office_count(t, OFF_ROW); const long colw = cx / nc;
    const long want = 502920;   /* 0.55 英寸: 16 磅一行加上下留白; 行多才压缩, 不按正文区拉满 */
    const long rowh = nr && cy / nr < want ? (cy / nr > 320000 ? cy / nr : 320000) : want;
    const int sz = rowh < 400000 ? 1200 : rowh < 460000 ? 1400 : 1600;
    if (center && rowh * nr < cy) y += (cy - rowh * nr) / 2;   /* 单独一张短表垂直居中: 三行表顶在标题下, 底下六成空白; 上面有引言时紧贴引言, 不居中(居中就和引言脱开) */
    ds4_buf_printf(&w->xml, "<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id=\"%d\" name=\"Table %d\"/><p:cNvGraphicFramePr><a:graphicFrameLocks noGrp=\"1\"/></p:cNvGraphicFramePr><p:nvPr/></p:nvGraphicFramePr>"
                            "<p:xfrm><a:off x=\"%ld\" y=\"%ld\"/><a:ext cx=\"%ld\" cy=\"%ld\"/></p:xfrm><a:graphic><a:graphicData uri=\"http://schemas.openxmlformats.org/drawingml/2006/table\"><a:tbl><a:tblPr firstRow=\"1\" bandRow=\"1\"/><a:tblGrid>", w->nid, w->nid, x, y, cx, rowh * nr);
    w->nid++;
    for (int i = 0; i < nc; i++) ds4_buf_printf(&w->xml, "<a:gridCol w=\"%ld\"/>", colw);
    ds4_buf_puts(&w->xml, "</a:tblGrid>");
    int ri = 0;
    for (const office_node *r = t->child; r; r = r->next, ri++) {
        ds4_buf_printf(&w->xml, "<a:tr h=\"%ld\">", rowh);
        const char *fill = r->header ? T->th_fill : ri % 2 ? T->panel : NULL;
        int k = 0;
        for (const office_node *c = r->child; c && k < nc; c = c->next, k++) {
            ds4_buf_puts(&w->xml, "<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>");
            pptx_para(&w->xml, T, c->child, sz, r->header, false, r->header ? T->th_text : T->text, 0, NULL, NULL);   /* 单元格字色显式给: 默认跟表格样式走(黑), 深色底上看不见(10-14 LibreOffice 实撞) */
            ds4_buf_puts(&w->xml, "</a:txBody><a:tcPr marL=\"120000\" marR=\"120000\" marT=\"60000\" marB=\"60000\" anchor=\"ctr\"><a:lnL><a:noFill/></a:lnL><a:lnR><a:noFill/></a:lnR><a:lnT><a:noFill/></a:lnT>");
            ds4_buf_printf(&w->xml, "<a:lnB w=\"6350\"><a:solidFill><a:srgbClr val=\"%s\"><a:alpha val=\"35000\"/></a:srgbClr></a:solidFill></a:lnB>", T->muted);
            if (fill) ds4_buf_printf(&w->xml, "<a:solidFill><a:srgbClr val=\"%s\"/></a:solidFill>", fill);
            ds4_buf_puts(&w->xml, "</a:tcPr></a:tc>");
        }
        for (; k < nc; k++) ds4_buf_puts(&w->xml, "<a:tc><a:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang=\"zh-CN\"/></a:p></a:txBody><a:tcPr/></a:tc>");
        ds4_buf_puts(&w->xml, "</a:tr>");
    }
    ds4_buf_puts(&w->xml, "</a:tbl></a:graphicData></a:graphic></p:graphicFrame>");
}
static void list_paras(ds4_buf *b, const pptx_theme_t *t, const office_node *ls, int lvl, int sz) {
    int k = 0;
    for (const office_node *it = ls->child; it; it = it->next, k++) {
        ds4_buf_printf(b, "<a:p><a:pPr marL=\"%d\" indent=\"-342900\" lvl=\"%d\"><a:spcBef><a:spcPts val=\"%d\"/></a:spcBef>", 342900 + 457200 * lvl, lvl, lvl ? sz / 3 : sz / 2);
        if (ls->ordered) ds4_buf_puts(b, "<a:buFont typeface=\"+mj-lt\"/><a:buAutoNum type=\"arabicPeriod\"/>");
        else ds4_buf_printf(b, "<a:buClr><a:srgbClr val=\"%s\"/></a:buClr><a:buFont typeface=\"Arial\"/><a:buChar char=\"%s\"/>", t->accent, lvl ? "\xE2\x80\x93" : "\xE2\x80\xA2");   /* 项目符号用强调色: 不加图标时唯一的"一点颜色" */
        ds4_buf_puts(b, "</a:pPr>");
        bool any = false;
        for (const office_node *c = it->child; c; c = c->next) if (c->kind != OFF_LIST) { office_node one = *c; one.next = NULL; inlines(b, t, &one, sz, false, false, NULL); any = true; }
        if (!any) ds4_buf_printf(b, "<a:endParaRPr lang=\"zh-CN\" sz=\"%d\"/>", sz);
        ds4_buf_puts(b, "</a:p>");
        for (const office_node *c = it->child; c; c = c->next) if (c->kind == OFF_LIST) list_paras(b, t, c, lvl + 1 < 4 ? lvl + 1 : 3, sz > 1400 ? sz - 200 : sz);
    }
}
/* [from, stop) 里的正文块 → 段落; 图和表不进这里(单独定位); title 跳过 */
int pptx_body_paras(ds4_buf *b, const pptx_theme_t *t, const office_node *from, const office_node *stop, int sz, const office_node *title) {
    int n = 0;
    for (const office_node *x = from; x && x != stop; x = x->next) {
        if (x == title || x->kind == OFF_IMAGE || x->kind == OFF_TABLE || x->kind == OFF_NOTES || x->kind == OFF_HR) continue;
        n++;
        if (x->kind == OFF_HEADING) pptx_para(b, t, x->child, sz + 200, true, false, t->title, 0, NULL, NULL);
        else if (x->kind == OFF_PARA) pptx_para(b, t, x->child, sz, false, false, NULL, 0, NULL, NULL);
        else if (x->kind == OFF_QUOTE) pptx_para(b, t, x->child, sz, false, true, t->muted, 0, NULL, NULL);
        else if (x->kind == OFF_LIST) list_paras(b, t, x, 0, sz);
        else if (x->kind == OFF_CODE) {
            for (const char *p = x->text ? x->text : ""; *p; ) {
                const char *nl = strchr(p, '\n'); const size_t len = nl ? (size_t)(nl - p) : strlen(p);
                if (!nl && !len) break;
                office_node c = { OFF_TEXT, 0, 0, 0, 0, 0, 1, NULL, NULL, 0, NULL, NULL }; char *line = malloc(len + 1); memcpy(line, p, len); line[len] = 0; c.text = line;
                ds4_buf_puts(b, "<a:p><a:pPr><a:spcBef><a:spcPts val=\"0\"/></a:spcBef><a:buNone/></a:pPr>"); inlines(b, t, &c, sz - 400 > 1000 ? sz - 400 : 1000, false, false, NULL); ds4_buf_puts(b, "</a:p>");
                free(line); if (!nl) break; p = nl + 1;
            }
        }
    }
    return n;
}
