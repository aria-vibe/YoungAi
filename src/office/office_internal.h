/* office_internal.h — src/office 内部的跨文件声明(写出器的包骨架模板与小工具); 对外接口在 ds4_office.h。 */
#ifndef DS4_OFFICE_INTERNAL_H
#define DS4_OFFICE_INTERNAL_H
#include "ds4_office.h"
#include <stdlib.h>
#include <string.h>

#define XML_HEAD "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
#define A_NS_DECL "xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\""
#define R_NS_DECL "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\""
#define EMU_PER_PX 9525               /* 96 dpi: 914400 EMU/英寸 ÷ 96 */
#define DOCX_TEXT_W_TWIPS 9026        /* A4 11906 − 两边 1440 */
#define DOCX_TEXT_W_EMU 5731510       /* 同一宽度换成 EMU(9026 twips × 635) */

/* 读整个文件(malloc, NUL 结尾); 不存在 NULL。写出器只在这里碰文件系统(读图片) */
uint8_t *office_slurp(const char *path, size_t *n);
/* 相对路径接到 base_dir 后面; 绝对路径原样 */
void office_join(const char *base, const char *rel, char *out, size_t n);
/* 树里第一个标题的纯文本(malloc), 没有 → NULL */
char *office_first_heading(const office_node *doc);
/* 行内节点序列的纯文本(malloc) */
char *office_plain(const office_node *inl);

/* docx 包骨架(office_docx_pkg.c) */
void docx_content_types(ds4_buf *b, bool png, bool jpeg);
void docx_root_rels(ds4_buf *b);
void docx_styles(ds4_buf *b);
void docx_footer(ds4_buf *b);
void docx_props(ds4_buf *core, ds4_buf *app, const char *title, time_t now);
/* 列表编号定义: kinds[i] 两位 —— bit0 外层有序, bit1 内层有序 */
void docx_numbering(ds4_buf *b, const uint8_t *kinds, int n);

/* pptx 主题(office_pptx_pkg.c 的表): 颜色全是 6 位十六进制。金标 slides*.pptx.golden 跟着表走, 改值 = 改金标(先 --keep 看图再 --update-golden)。
 * 每个主题都写底色 + 右下页码; 卡片/表格隔行用 panel。arcs = 封面与章节页画两道圆弧(clean 只靠留白, 不画) */
typedef struct {
    const char *name;
    const char *bg, *text, *title, *panel, *accent, *muted, *th_fill, *th_text, *link;   /* 底色 / 正文 / 标题 / 面板(卡片底) / 强调 / 次要文字 / 表头底 / 表头字 / 链接 */
    const char *bg2, *c2, *c3, *c4;   /* 底色渐变的另一端(每页 45° 线性渐变 bg→bg2) / 图表第 2~4 个系列色(第 1 个 = accent) */
    bool arcs, dark;                  /* dark: 深色底 —— 卡片用强调色细描边代替阴影(深底上阴影看不见) */
} pptx_theme_t;
const pptx_theme_t *pptx_theme_find(const char *name);   /* NULL/"" → clean; 不认识 → NULL */

/* pptx 页面几何(EMU, 914400 = 1 英寸; 16:9 13.33 × 7.5 英寸)。正文区下面留 0.6 英寸给页码 */
#define PPTX_W 12192000
#define PPTX_H 6858000
#define PPTX_MARGIN 685800
#define PPTX_TITLE_Y 457200
#define PPTX_TITLE_H 1000000
#define PPTX_BODY_Y 1600000
#define PPTX_BODY_H 4650000
#define PPTX_BODY_W (PPTX_W - 2 * PPTX_MARGIN)
/* 写出器状态: 一页一份 xml/rels; nsec = 已出的章节页数(章节页左上的大序号); nchart = 已出的图表部件数(全篇编号)。
 * 入场动画: 版式把要逐个淡入的形状 id 登记进 anim(同一组 = 同时出现), 页末 pptx_timing 写成 p:timing; 只登记 p:sp(文本框/卡片/序号球), 图和表不动 */
#define PPTX_ANIM_MAX 48
typedef struct { office_zip *zip; const office_opts *o; office_report *rep; const pptx_theme_t *t; ds4_buf xml, rels; int slide, nsec, nid, rid, nimg, nchart; bool png, jpeg;
                 int anim_id[PPTX_ANIM_MAX], anim_grp[PPTX_ANIM_MAX], nanim, grp; } pptx_sw;
/* 绘图原语(office_pptx_draw.c) */
void pptx_para(ds4_buf *b, const pptx_theme_t *t, const office_node *inl, int sz, bool bold, bool italic, const char *color, int lvl, const char *bullet, const char *align);
void pptx_para_str(ds4_buf *b, const pptx_theme_t *t, const char *s, int sz, bool bold, const char *color, const char *align);
int pptx_text_box(pptx_sw *w, long x, long y, long cx, long cy, const char *anchor, const ds4_buf *paras, const char *fill);   /* 返回形状 id(登记动画用) */
int pptx_shape(pptx_sw *w, const char *prst, long x, long y, long cx, long cy, const char *fill, int falpha, const char *line, long lw, int lalpha, const ds4_buf *paras);
int pptx_card(pptx_sw *w, long x, long y, long cx, long cy);   /* 圆角面板: 浅色主题带柔和阴影, 深色主题强调色细描边 */
void pptx_glow(pptx_sw *w, long x, long y, long d, const char *color, int alpha);   /* 径向渐变光晕(封面/章节页右下) */
void pptx_bg(pptx_sw *w);                                      /* 本页底色: bg→bg2 45° 线性渐变 */
void pptx_anim(pptx_sw *w, int spid, bool new_group);          /* 登记入场动画; new_group = 从这个形状起晚一拍出现 */
void pptx_timing(pptx_sw *w);                                  /* 淡入切换 + 登记过的形状按组逐个淡入(p:transition + p:timing); 没登记只写切换 */
int pptx_picture(pptx_sw *w, const office_node *im, long bx, long by, long bw, long bh, bool fill);   /* 回形状 id(给动画); 图读不到回 0 */
void pptx_table(pptx_sw *w, const office_node *t, long x, long y, long cx, long cy, bool center);
int pptx_body_paras(ds4_buf *b, const pptx_theme_t *t, const office_node *from, const office_node *stop, int sz, const office_node *title);
/* 原生图表(office_pptx_chart.c): 表头第一格是 bar/line/pie 的表 → ppt/charts/chartN.xml + 本页 graphicFrame; 规格见文件头。不是图表的表 → false, 照常排表格 */
bool pptx_chart_ok(const office_node *tbl);
void pptx_chart(pptx_sw *w, const office_node *tbl, long x, long y, long cx, long cy);

/* pptx 包骨架(office_pptx_pkg.c) */
void pptx_content_types(ds4_buf *b, int nslides, bool notes, bool png, bool jpeg, int ncharts);
void pptx_root_rels(ds4_buf *b);
void pptx_presentation(ds4_buf *b, ds4_buf *rels, int nslides, bool notes);
void pptx_small_parts(ds4_buf *pres_props, ds4_buf *view_props, ds4_buf *table_styles);
void pptx_theme(ds4_buf *b, const char *name, const pptx_theme_t *t);
void pptx_master(ds4_buf *b, ds4_buf *rels);
void pptx_layout(ds4_buf *b, ds4_buf *rels);
void pptx_notes_master(ds4_buf *b, ds4_buf *rels);
void pptx_notes_slide(ds4_buf *b, ds4_buf *rels, int slide, const char *text);
void pptx_props(ds4_buf *core, ds4_buf *app, const char *title, int nslides, time_t now);

/* xlsx 包骨架(office_xlsx_pkg.c) */
void xlsx_content_types(ds4_buf *b, int nsheets);
void xlsx_root_rels(ds4_buf *b);
void xlsx_workbook(ds4_buf *b, ds4_buf *rels, const office_sheet *sheets, int n);
void xlsx_styles(ds4_buf *b);
void xlsx_props(ds4_buf *core, ds4_buf *app, time_t now);

#endif
