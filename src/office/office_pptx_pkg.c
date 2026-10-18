/* office_pptx_pkg.c — pptx 包里不随内容变的部分: [Content_Types] / 根 rels / presentation(+rels) / presProps / viewProps / tableStyles / theme / 母版 / 版式 / 备注母版。
 * 16:9(12192000 × 6858000 EMU), 一个主题色, 中文 Microsoft YaHei、西文 Calibri。
 * 幻灯片不用占位符(ph), 文本框显式定位 —— WPS / Keynote / LibreOffice 打开位置一致, 不依赖各家对版式占位符的解释。
 * 母版/版式/主题是 PowerPoint 要求的最小闭包(缺 theme 的 fmtScheme 之类 PowerPoint 直接"需要修复"), 内容再简单也得全有。 */
#include "office_internal.h"
#include <stdio.h>

#define A_NS "xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\""
#define R_NS "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\""
#define P_NS "xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\""
#define REL "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"

void pptx_content_types(ds4_buf *b, int nslides, bool notes, bool png, bool jpeg, int ncharts) {
    ds4_buf_puts(b, XML_HEAD "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
                    "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"application/xml\"/>");
    if (png) ds4_buf_puts(b, "<Default Extension=\"png\" ContentType=\"image/png\"/>");
    if (jpeg) ds4_buf_puts(b, "<Default Extension=\"jpeg\" ContentType=\"image/jpeg\"/>");
    ds4_buf_puts(b, "<Override PartName=\"/ppt/presentation.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml\"/>"
                    "<Override PartName=\"/ppt/slideMasters/slideMaster1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml\"/>"
                    "<Override PartName=\"/ppt/slideLayouts/slideLayout1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml\"/>"
                    "<Override PartName=\"/ppt/theme/theme1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.theme+xml\"/>"
                    "<Override PartName=\"/ppt/presProps.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.presProps+xml\"/>"
                    "<Override PartName=\"/ppt/viewProps.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml\"/>"
                    "<Override PartName=\"/ppt/tableStyles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml\"/>"
                    "<Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>"
                    "<Override PartName=\"/docProps/app.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/>");
    for (int i = 1; i <= nslides; i++) ds4_buf_printf(b, "<Override PartName=\"/ppt/slides/slide%d.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slide+xml\"/>", i);
    for (int i = 1; i <= ncharts; i++) ds4_buf_printf(b, "<Override PartName=\"/ppt/charts/chart%d.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.drawingml.chart+xml\"/>", i);
    if (notes) {
        ds4_buf_puts(b, "<Override PartName=\"/ppt/notesMasters/notesMaster1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.notesMaster+xml\"/>"
                        "<Override PartName=\"/ppt/theme/theme2.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.theme+xml\"/>");
        for (int i = 1; i <= nslides; i++) ds4_buf_printf(b, "<Override PartName=\"/ppt/notesSlides/notesSlide%d.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml\"/>", i);
    }
    ds4_buf_puts(b, "</Types>");
}

void pptx_root_rels(ds4_buf *b) {
    ds4_buf_puts(b, XML_HEAD "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
                    "<Relationship Id=\"rId1\" Type=\"" REL "officeDocument\" Target=\"ppt/presentation.xml\"/>"
                    "<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/>"
                    "<Relationship Id=\"rId3\" Type=\"" REL "extended-properties\" Target=\"docProps/app.xml\"/></Relationships>");
}

void pptx_presentation(ds4_buf *b, ds4_buf *rels, int nslides, bool notes) {
    ds4_buf_puts(b, XML_HEAD "<p:presentation " A_NS " " R_NS " " P_NS " saveSubsetFonts=\"1\"><p:sldMasterIdLst><p:sldMasterId id=\"2147483648\" r:id=\"rId1\"/></p:sldMasterIdLst>");
    if (notes) ds4_buf_printf(b, "<p:notesMasterIdLst><p:notesMasterId r:id=\"rId%d\"/></p:notesMasterIdLst>", nslides + 6);
    ds4_buf_puts(b, "<p:sldIdLst>");
    for (int i = 1; i <= nslides; i++) ds4_buf_printf(b, "<p:sldId id=\"%d\" r:id=\"rId%d\"/>", 255 + i, i + 1);
    ds4_buf_puts(b, "</p:sldIdLst><p:sldSz cx=\"12192000\" cy=\"6858000\"/><p:notesSz cx=\"6858000\" cy=\"9144000\"/>"
                    "<p:defaultTextStyle><a:defPPr><a:defRPr lang=\"zh-CN\"/></a:defPPr></p:defaultTextStyle></p:presentation>");
    ds4_buf_puts(rels, XML_HEAD "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"" REL "slideMaster\" Target=\"slideMasters/slideMaster1.xml\"/>");
    for (int i = 1; i <= nslides; i++) ds4_buf_printf(rels, "<Relationship Id=\"rId%d\" Type=\"" REL "slide\" Target=\"slides/slide%d.xml\"/>", i + 1, i);
    ds4_buf_printf(rels, "<Relationship Id=\"rId%d\" Type=\"" REL "theme\" Target=\"theme/theme1.xml\"/><Relationship Id=\"rId%d\" Type=\"" REL "presProps\" Target=\"presProps.xml\"/>"
                         "<Relationship Id=\"rId%d\" Type=\"" REL "viewProps\" Target=\"viewProps.xml\"/><Relationship Id=\"rId%d\" Type=\"" REL "tableStyles\" Target=\"tableStyles.xml\"/>", nslides + 2, nslides + 3, nslides + 4, nslides + 5);
    if (notes) ds4_buf_printf(rels, "<Relationship Id=\"rId%d\" Type=\"" REL "notesMaster\" Target=\"notesMasters/notesMaster1.xml\"/>", nslides + 6);
    ds4_buf_puts(rels, "</Relationships>");
}

void pptx_small_parts(ds4_buf *pres_props, ds4_buf *view_props, ds4_buf *table_styles) {
    ds4_buf_puts(pres_props, XML_HEAD "<p:presentationPr " A_NS " " R_NS " " P_NS "/>");
    ds4_buf_puts(view_props, XML_HEAD "<p:viewPr " A_NS " " R_NS " " P_NS "><p:normalViewPr><p:restoredLeft sz=\"15620\"/><p:restoredTop sz=\"94660\"/></p:normalViewPr><p:gridSpacing cx=\"72008\" cy=\"72008\"/></p:viewPr>");
    ds4_buf_puts(table_styles, XML_HEAD "<a:tblStyleLst " A_NS " def=\"{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}\"/>");
}

/* 主题表(10-14 用户: 要"科技风"的 PPT 出来还是白底蓝字 —— 之前只有一套版式, 没有"风格"这个入口; 10-18 用户: "配色不行" → 重调并加到五套, 一个主题一种主色 + 一个锐利强调色)。
 * 值进 theme1.xml 的 clrScheme(dk1=正文 lt1=底 dk2=标题 lt2=面板 accent1=强调), 母版 clrMap 把 tx1→dk1 bg1→lt1, 所以没显式上色的文字自动跟主题走: 深色主题正文自然是浅字。
 * 深色主题的表头字单给(浅灰不够亮); 浅色主题表头是实色底配白字。
 * 列顺序: 底色 / 正文 / 标题 / 面板(卡片底、表格隔行) / 强调 / 次要字 / 表头底 / 表头字 / 链接 / 渐变另一端 / 图表系列 2~4 色 / 圆弧 / 深色 */
static const pptx_theme_t THEMES[] = {
    { "clean",  "FFFFFF", "1F2937", "0F172A", "F1F5F9", "0F766E", "64748B", "0F172A", "FFFFFF", "0F766E", "F3F6FA", "F59E0B", "6366F1", "94A3B8", false, false },   /* 白底近黑标题 + 一点青绿: 商务 / 正式 / 没说 */
    { "tech",   "0B1220", "E2E8F0", "F8FAFC", "162133", "22D3EE", "94A3B8", "1E293B", "7DD3FC", "67E8F9", "16213F", "A78BFA", "34D399", "F472B6", true,  true },    /* 深海军蓝底 + 青色强调: 科技 / AI / 产品发布 */
    { "warm",   "FBF7F2", "3B2F2F", "7A3B2E", "F0E6D8", "B85042", "8A7B72", "B85042", "FFFFFF", "9A3F33", "F2E6D6", "D9A066", "7A8F6C", "8A7B72", true,  false },   /* 浅沙底 + 赤陶强调: 生活 / 教育 / 餐饮 / 亲切 */
    { "forest", "FFFFFF", "1F2A1F", "1E4620", "EAF2E3", "2C5F2D", "6B7A6B", "2C5F2D", "FFFFFF", "3F7A3F", "F0F7EA", "97BC62", "D9A441", "6B7A6B", true,  false },   /* 白底深绿标题 + 森林绿: 环保 / 农业 / 健康 / 可持续 */
    { "berry",  "2B0F1E", "ECE2D0", "FFF7EE", "47203A", "E0A458", "C9AFA8", "5C2A45", "F2CDBB", "E8B48F", "3D162D", "C9AFA8", "8FB3A5", "E8B48F", true,  true },    /* 莓红深底 + 金色强调: 文化 / 艺术 / 品牌 / 庆典 */
};
#define NTHEMES (sizeof THEMES / sizeof THEMES[0])
const pptx_theme_t *pptx_theme_find(const char *name) {
    if (!name || !name[0]) return &THEMES[0];
    for (size_t i = 0; i < NTHEMES; i++) if (!strcmp(name, THEMES[i].name)) return &THEMES[i];
    return NULL;
}
const char *office_pptx_themes(void) { return "clean, tech, warm, forest, berry"; }
bool office_pptx_theme_ok(const char *name) { return pptx_theme_find(name) != NULL; }

/* 主题: 颜色 + 字体 + 三档填充/线/效果(PowerPoint 要求的最小闭包) */
void pptx_theme(ds4_buf *b, const char *name, const pptx_theme_t *t) {
    ds4_buf_printf(b, XML_HEAD "<a:theme " A_NS " name=\"%s\"><a:themeElements><a:clrScheme name=\"DwarfStar\">"
                      "<a:dk1><a:srgbClr val=\"%s\"/></a:dk1><a:lt1><a:srgbClr val=\"%s\"/></a:lt1><a:dk2><a:srgbClr val=\"%s\"/></a:dk2><a:lt2><a:srgbClr val=\"%s\"/></a:lt2>"
                      "<a:accent1><a:srgbClr val=\"%s\"/></a:accent1><a:accent2><a:srgbClr val=\"ED7D31\"/></a:accent2><a:accent3><a:srgbClr val=\"A5A5A5\"/></a:accent3>"
                      "<a:accent4><a:srgbClr val=\"FFC000\"/></a:accent4><a:accent5><a:srgbClr val=\"5B9BD5\"/></a:accent5><a:accent6><a:srgbClr val=\"70AD47\"/></a:accent6>"
                      "<a:hlink><a:srgbClr val=\"%s\"/></a:hlink><a:folHlink><a:srgbClr val=\"954F72\"/></a:folHlink></a:clrScheme>", name, t->text, t->bg, t->title, t->panel, t->accent, t->link);
    ds4_buf_puts(b,
                      "<a:fontScheme name=\"DwarfStar\"><a:majorFont><a:latin typeface=\"Calibri\"/><a:ea typeface=\"Microsoft YaHei\"/><a:cs typeface=\"\"/></a:majorFont>"
                      "<a:minorFont><a:latin typeface=\"Calibri\"/><a:ea typeface=\"Microsoft YaHei\"/><a:cs typeface=\"\"/></a:minorFont></a:fontScheme>"
                      "<a:fmtScheme name=\"DwarfStar\"><a:fillStyleLst><a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill><a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill><a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill></a:fillStyleLst>"
                      "<a:lnStyleLst><a:ln w=\"6350\"><a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill></a:ln><a:ln w=\"12700\"><a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill></a:ln><a:ln w=\"19050\"><a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill></a:ln></a:lnStyleLst>"
                      "<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>"
                      "<a:bgFillStyleLst><a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill><a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill><a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>"
                      "</a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/></a:theme>");
}

#define EMPTY_TREE "<p:spTree><p:nvGrpSpPr><p:cNvPr id=\"1\" name=\"\"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/><a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr></p:spTree>"
#define CLRMAP "<p:clrMap bg1=\"lt1\" tx1=\"dk1\" bg2=\"lt2\" tx2=\"dk2\" accent1=\"accent1\" accent2=\"accent2\" accent3=\"accent3\" accent4=\"accent4\" accent5=\"accent5\" accent6=\"accent6\" hlink=\"hlink\" folHlink=\"folHlink\"/>"

void pptx_master(ds4_buf *b, ds4_buf *rels) {
    ds4_buf_puts(b, XML_HEAD "<p:sldMaster " A_NS " " R_NS " " P_NS "><p:cSld><p:bg><p:bgRef idx=\"1001\"><a:schemeClr val=\"bg1\"/></p:bgRef></p:bg>" EMPTY_TREE "</p:cSld>" CLRMAP
                    "<p:sldLayoutIdLst><p:sldLayoutId id=\"2147483649\" r:id=\"rId1\"/></p:sldLayoutIdLst>"
                    "<p:txStyles><p:titleStyle><a:lvl1pPr><a:defRPr sz=\"3600\" b=\"1\"><a:solidFill><a:schemeClr val=\"tx2\"/></a:solidFill><a:latin typeface=\"+mj-lt\"/><a:ea typeface=\"+mj-ea\"/></a:defRPr></a:lvl1pPr></p:titleStyle>"
                    "<p:bodyStyle><a:lvl1pPr><a:defRPr sz=\"2000\"><a:latin typeface=\"+mn-lt\"/><a:ea typeface=\"+mn-ea\"/></a:defRPr></a:lvl1pPr></p:bodyStyle><p:otherStyle><a:lvl1pPr><a:defRPr sz=\"1800\"/></a:lvl1pPr></p:otherStyle></p:txStyles></p:sldMaster>");
    ds4_buf_puts(rels, XML_HEAD "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"" REL "slideLayout\" Target=\"../slideLayouts/slideLayout1.xml\"/>"
                       "<Relationship Id=\"rId2\" Type=\"" REL "theme\" Target=\"../theme/theme1.xml\"/></Relationships>");
}
void pptx_layout(ds4_buf *b, ds4_buf *rels) {
    ds4_buf_puts(b, XML_HEAD "<p:sldLayout " A_NS " " R_NS " " P_NS " type=\"blank\" preserve=\"1\"><p:cSld name=\"Blank\">" EMPTY_TREE "</p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>");
    ds4_buf_puts(rels, XML_HEAD "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"" REL "slideMaster\" Target=\"../slideMasters/slideMaster1.xml\"/></Relationships>");
}
void pptx_notes_master(ds4_buf *b, ds4_buf *rels) {
    ds4_buf_puts(b, XML_HEAD "<p:notesMaster " A_NS " " R_NS " " P_NS "><p:cSld>" EMPTY_TREE "</p:cSld>" CLRMAP "<p:notesStyle><a:lvl1pPr><a:defRPr sz=\"1200\"/></a:lvl1pPr></p:notesStyle></p:notesMaster>");
    ds4_buf_puts(rels, XML_HEAD "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"" REL "theme\" Target=\"../theme/theme2.xml\"/></Relationships>");
}
/* 备注页: 一个 body 占位符装讲稿 */
void pptx_notes_slide(ds4_buf *b, ds4_buf *rels, int slide, const char *text) {
    ds4_buf_puts(b, XML_HEAD "<p:notes " A_NS " " R_NS " " P_NS "><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id=\"1\" name=\"\"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/><a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>"
                    "<p:sp><p:nvSpPr><p:cNvPr id=\"2\" name=\"Notes\"/><p:cNvSpPr><a:spLocks noGrp=\"1\"/></p:cNvSpPr><p:nvPr><p:ph type=\"body\" idx=\"1\"/></p:nvPr></p:nvSpPr>"
                    "<p:spPr><a:xfrm><a:off x=\"685800\" y=\"4343400\"/><a:ext cx=\"5486400\" cy=\"4114800\"/></a:xfrm><a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang=\"zh-CN\"/><a:t>");
    office_xml_text(b, text);
    ds4_buf_puts(b, "</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>");
    ds4_buf_printf(rels, XML_HEAD "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"" REL "notesMaster\" Target=\"../notesMasters/notesMaster1.xml\"/>"
                         "<Relationship Id=\"rId2\" Type=\"" REL "slide\" Target=\"../slides/slide%d.xml\"/></Relationships>", slide);
}
void pptx_props(ds4_buf *core, ds4_buf *app, const char *title, int nslides, time_t now) {
    char ts[32]; office_iso_time(now, ts);
    ds4_buf_puts(core, XML_HEAD "<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" "
                       "xmlns:dcterms=\"http://purl.org/dc/terms/\" xmlns:dcmitype=\"http://purl.org/dc/dcmitype/\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"><dc:title>");
    office_xml_text(core, title ? title : "");
    ds4_buf_printf(core, "</dc:title><dc:creator>DwarfStar</dc:creator><dcterms:created xsi:type=\"dcterms:W3CDTF\">%s</dcterms:created><dcterms:modified xsi:type=\"dcterms:W3CDTF\">%s</dcterms:modified></cp:coreProperties>", ts, ts);
    ds4_buf_printf(app, XML_HEAD "<Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\" xmlns:vt=\"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes\"><Application>DwarfStar</Application><Slides>%d</Slides></Properties>", nslides);
}
