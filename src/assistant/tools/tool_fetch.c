/* tool_fetch.c — fetch_image: 从网上拿一张图进工作区(10-18 用户: "图确实生成了, 但好看的背景图标没有, 最起码有一些设备图片等等, 这些交给 web search")。
 * 两种:
 *   photo = Bing 图片搜索(大图筛选)前几条按顺序试: 火狐直接打开图片地址(页面本身就是这张图, 同源, canvas 不染色), 最长边缩到 1600, 统一转 JPEG。
 *   icon  = Iconify 公开接口(api.iconify.design, 20 万开源图标: Material / Tabler / Lucide 单色线条, logos / devicon 彩色品牌, fluent-emoji / noto 彩色表情;
 *           许可各自是 MIT / Apache / CC-BY)按名字搜, 取 SVG 并按给定颜色上色, 火狐 canvas 转成透明 PNG(PPT 卡片格里顶替序号球, office_pptx.c)。
 * 为什么走火狐不自己发 HTTP: 代理 / TLS / 真实 UA / 不带 Referer 的直开(防盗链放行)火狐都现成(tool_browser.c), webp / avif / gif 它都解, 一律 canvas 重编码成 PPT 认的 jpeg / png。
 * 出错会怎样: 前几张都下不来(不是位图 / 太小 / 站点不通)→ error 里逐条写原因; 照片的版权归原作者, 结果句带来源页, 模型对外发布前要提醒用户。 */
#include "../asst_internal.h"
#include "../../office/ds4_office.h"
#include <limits.h>
#include "../../web/web_internal.h"
#include <ctype.h>

#define PHOTO_MAX 1600       /* 照片最长边: 幻灯片半幅 1600 像素够清楚, 再大只是把 pptx 撑肥 */
#define PHOTO_MIN 480        /* 比这小的是缩略图, 放大就糊 */
#define PHOTO_TRIES 5
#define ICON_DEF 512
#define ICON_MIN 64
#define ICON_MAX 2048
#define ICON_LIST 12

static const char bing_img_js[] =   /* 结果页每张图一个 a.iusc, m 属性是 JSON: murl 原图地址, purl 所在网页, t 标题 */
    "(() => { const out = []; for (const a of document.querySelectorAll('a.iusc')) { try { const m = JSON.parse(a.getAttribute('m') || '{}');"
    " if (m.murl) out.push(m.murl + '\\t' + (m.purl || '') + '\\t' + String(m.t || '').replace(/[\\t\\n]+/g, ' ').trim().slice(0, 80)); } catch (e) {}"
    " if (out.length >= 12) break; } return out.join('\\n'); })()";
static const char grab_js[] =   /* 页面 = 这张图(火狐的图片文档): 等它解码, 缩到最长边 %d, 白底(透明 → 白)出 JPEG */
    "(() => new Promise(res => { const img = document.images && document.images[0]; const ct = document.contentType || '';"
    " if (!img || !ct.startsWith('image/')) return res('error: not an image (' + ct + ')');"
    " const go = () => { try { const W = img.naturalWidth, H = img.naturalHeight; if (!W || !H) return res('error: image did not decode');"
    " const s = Math.min(1, %d / Math.max(W, H)), w = Math.max(1, Math.round(W * s)), h = Math.max(1, Math.round(H * s));"
    " const c = document.createElement('canvas'); c.width = w; c.height = h; const g = c.getContext('2d'); g.fillStyle = '#fff'; g.fillRect(0, 0, w, h); g.drawImage(img, 0, 0, w, h);"
    " res(W + 'x' + H + ' ' + c.toDataURL('image/jpeg', 0.9)); } catch (e) { res('error: ' + e.message); } };"
    " if (img.complete) go(); else { img.onload = go; img.onerror = () => res('error: image failed to load'); } }))()";
static const char icon_js[] =   /* 页面 = SVG 文档: 序列化成 data: 再画进 %d 见方的透明画布(data: 图不染色 canvas) */
    "(() => new Promise(res => { const svg = document.documentElement; if (!svg || svg.tagName.toLowerCase() !== 'svg') return res('error: not an svg (' + (document.contentType || '') + ')');"
    " const xml = new XMLSerializer().serializeToString(svg), S = %d, img = new Image();"
    " img.onload = () => { try { const c = document.createElementNS('http://www.w3.org/1999/xhtml', 'canvas'); c.width = S; c.height = S; const g = c.getContext('2d');"   /* SVG 文档里 createElement 建的是 SVG 元素, 没有 getContext(10-18 实撞) */
    " const r = Math.min(S / img.width, S / img.height), w = img.width * r, h = img.height * r; g.drawImage(img, (S - w) / 2, (S - h) / 2, w, h);"
    " res(S + 'x' + S + ' ' + c.toDataURL('image/png')); } catch (e) { res('error: ' + e.message); } };"
    " img.onerror = () => res('error: svg did not rasterize'); img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(xml))); }))()";

static char *fail(const char *fmt, const char *a) { ds4_buf b = {0}; ds4_buf_printf(&b, fmt, a); return ds4_buf_take(&b); }
static bool hex_ok(const char *s) { const size_t n = strlen(s); if (n != 3 && n != 6) return false; for (; *s; s++) if (!isxdigit((unsigned char)*s)) return false; return true; }

/* "WxH data:image/...;base64,…" → 字节; 失败回 NULL 并把原因留在 why */
static uint8_t *decode(char *r, int *w, int *h, size_t *len, const char **why) {
    if (!r) { *why = "the page gave nothing back"; return NULL; }
    if (!strncmp(r, "error: ", 7)) { *why = r + 7; return NULL; }
    const char *b64 = strstr(r, "base64,");
    if (sscanf(r, "%dx%d", w, h) != 2 || !b64) { *why = "unexpected page result"; return NULL; }
    uint8_t *o = malloc(strlen(b64) * 3 / 4 + 4); *len = asst_b64dec(b64 + 7, o);
    if (!*len) { free(o); *why = "empty image data"; return NULL; }
    return o;
}

/* 照片: 搜索 → 逐条打开原图 → 第一张够大的 */
static uint8_t *photo(const char *query, size_t *len, int *w, int *h, char *src, size_t srcn, ds4_buf *log) {
    char err[512] = ""; char *enc = web_url_encode(query); ds4_buf u = {0};
    ds4_buf_printf(&u, "https://www.bing.com/images/search?q=%s&qft=+filterui:imagesize-large&form=IRFLTR&first=1", enc); free(enc);
    char *list = asst_web_js(u.ptr, bing_img_js, false, err, sizeof err); ds4_buf_free(&u);
    if (!list || !list[0]) { ds4_buf_printf(log, "image search gave no results%s%s", err[0] ? ": " : "", err); free(list); return NULL; }
    char js[sizeof grab_js + 16]; snprintf(js, sizeof js, grab_js, PHOTO_MAX);
    uint8_t *got = NULL; int tries = 0;
    for (char *line = list, *nl; line && *line && tries < PHOTO_TRIES && !got; line = nl ? nl + 1 : NULL) {
        nl = strchr(line, '\n'); if (nl) *nl = 0;
        char *tab = strchr(line, '\t'); if (tab) *tab = 0;
        const char *purl = tab ? tab + 1 : "", *title = ""; char *t2 = tab ? strchr(tab + 1, '\t') : NULL; if (t2) { *t2 = 0; title = t2 + 1; }
        char host[256]; if (strncmp(line, "http", 4) || !asst_url_host(line, host, sizeof host) || asst_host_private(host)) continue;
        tries++; err[0] = 0;
        char *r = asst_web_js(line, js, true, err, sizeof err); const char *why = err;
        got = decode(r, w, h, len, &why);
        if (got && *w < PHOTO_MIN && *h < PHOTO_MIN) { free(got); got = NULL; why = "too small"; }
        if (got) snprintf(src, srcn, "%s (page %s%s%s%s)", line, purl[0] ? purl : "?", title[0] ? ", \"" : "", title, title[0] ? "\"" : "");
        else ds4_buf_printf(log, "%s: %s\n", line, why && why[0] ? why : "failed");
        free(r);
    }
    free(list);
    if (!got && !tries) ds4_buf_puts(log, "the search results had no usable image links");
    return got;
}

/* 图标: 名字没给就搜; 取 SVG 上色转 PNG */
static uint8_t *icon(const char *query, const char *name, const char *set, const char *color, int size, size_t *len, int *w, int *h, char *src, size_t srcn, ds4_buf *log) {
    char err[512] = ""; char picked[128] = ""; ds4_buf u = {0};
    if (name && name[0]) snprintf(picked, sizeof picked, "%s", name);
    else {
        char *enc = web_url_encode(query); ds4_buf_printf(&u, "https://api.iconify.design/search?query=%s&limit=%d", enc, ICON_LIST); free(enc);
        if (set && set[0]) { enc = web_url_encode(set); ds4_buf_printf(&u, "&prefixes=%s", enc); free(enc); }
        char *json = asst_web_text(u.ptr, err, sizeof err); u.len = 0;
        char *arr = NULL; if (!json || !asst_jraw(json, "icons", &arr)) { ds4_buf_printf(log, "icon search failed%s%s", err[0] ? ": " : "", err); free(json); return NULL; }
        char *e = NULL; ds4_buf others = {0};
        for (int i = 0; i <= ICON_LIST && asst_jarr_at(arr, i, &e); i++) {   /* 接口最少回 32 条; 给模型看 12 条够挑 */
            char *s = e; size_t n = strlen(s); if (n >= 2 && s[0] == '"') { s[n - 1] = 0; s++; }
            if (!picked[0]) snprintf(picked, sizeof picked, "%s", s); else ds4_buf_printf(&others, "%s%s", others.len ? ", " : "", s);
            free(e);
        }
        free(arr); free(json);
        if (!picked[0]) { ds4_buf_printf(log, "no icon matches \"%s\" (try an English word: server, phone, shield, chart)", query); ds4_buf_free(&others); return NULL; }
        if (others.len) ds4_buf_printf(log, "Other matches: %s. Call again with name to use one of them instead.\n", others.ptr);
        ds4_buf_free(&others);
    }
    const char *colon = strchr(picked, ':'); if (!colon || !colon[1]) { ds4_buf_printf(log, "icon name must look like prefix:name, got %s", picked); return NULL; }
    ds4_buf_printf(&u, "https://api.iconify.design/%.*s/%s.svg?height=%d", (int)(colon - picked), picked, colon + 1, size);
    if (color && color[0]) ds4_buf_printf(&u, "&color=%%23%s", color);
    char js[sizeof icon_js + 16]; snprintf(js, sizeof js, icon_js, size);
    char *r = asst_web_js(u.ptr, js, true, err, sizeof err); ds4_buf_free(&u);
    const char *why = err; uint8_t *got = decode(r, w, h, len, &why);
    if (!got) ds4_buf_printf(log, "%s: %s", picked, why && why[0] ? why : "failed"); else snprintf(src, srcn, "icon %s%s%s", picked, color && color[0] ? " colored #" : "", color && color[0] ? color : "");
    free(r);
    return got;
}

char *asst_fetch_exec(const char *args, const asst_gres *g, const char *trash, int *ok) {
    *ok = 0;
    char *query = NULL, *kind = NULL, *color = NULL, *set = NULL, *name = NULL; long size = ICON_DEF;
    asst_jstr(args, "query", &query); asst_jstr(args, "kind", &kind); asst_jstr(args, "color", &color); asst_jstr(args, "set", &set); asst_jstr(args, "name", &name); asst_jint(args, "size", &size);
    const bool is_icon = kind && !strcmp(kind, "icon");
    const char *dot = strrchr(g->path, '.'), *sl = strrchr(g->path, '/'); if (dot && sl && dot < sl) dot = NULL;
    char *out = NULL;
    if (kind && kind[0] && !is_icon && strcmp(kind, "photo")) out = fail("error: kind must be photo or icon, got %s\n", kind);
    else if (!is_icon && (!query || !query[0])) out = strdup("error: query is required: what the photo should show, e.g. server rack in a data center\n");
    else if (is_icon && (!query || !query[0]) && (!name || !name[0])) out = strdup("error: query (what the icon shows, in English: server, shield, chart) or name (prefix:name) is required\n");
    else if (!is_icon && (!dot || (strcasecmp(dot, ".jpg") && strcasecmp(dot, ".jpeg")))) out = strdup("error: a photo is saved as JPEG: the path must end with .jpg\n");
    else if (is_icon && (!dot || strcasecmp(dot, ".png"))) out = strdup("error: an icon is saved as transparent PNG: the path must end with .png\n");
    else if (is_icon && (size < ICON_MIN || size > ICON_MAX)) out = fail("error: size must be between 64 and 2048 pixels%s\n", "");
    else if (is_icon && color && color[0] && (color[0] == '#' ? !hex_ok(color + 1) : !hex_ok(color))) out = fail("error: color must be a hex color like #22D3EE, got %s\n", color);
    if (out) { free(query); free(kind); free(color); free(set); free(name); return out; }
    if (color && color[0] == '#') memmove(color, color + 1, strlen(color));
    size_t len = 0; int w = 0, h = 0; char src[1200] = ""; ds4_buf log = {0};
    uint8_t *data = is_icon ? icon(query, name, set, color, (int)size, &len, &w, &h, src, sizeof src, &log) : photo(query, &len, &w, &h, src, sizeof src, &log);
    free(query); free(kind); free(color); free(set); free(name);
    ds4_buf b = {0};
    if (!data) { ds4_buf_printf(&b, "error: no %s fetched.\n%s\n", is_icon ? "icon" : "photo", log.ptr ? log.ptr : ""); ds4_buf_free(&log); return ds4_buf_take(&b); }
    office_imginfo info; const bool sized = office_img_info(data, len, &info);
    char *saved = NULL; const asst_sbx_rc rc = asst_save_bytes(g, trash, data, len, &saved); free(data);
    if (rc != ASST_SBX_OK || !saved || !strncmp(saved, "error", 5)) { ds4_buf_printf(&b, "error: %s\n", saved ? (!strncmp(saved, "error: ", 7) ? saved + 7 : saved) : "sandbox failed"); free(saved); ds4_buf_free(&log); return ds4_buf_take(&b); }
    const char *rel = asst_under(g->path, asst_ws) && strlen(g->path) > strlen(asst_ws) ? g->path + strlen(asst_ws) + 1 : g->path;
    ds4_buf_printf(&b, "image: %s\n%s %s (%s %dx%d, %.0f KB)%s", rel, saved, rel, is_icon ? "PNG, transparent" : "JPEG", sized ? info.w : w, sized ? info.h : h, len / 1024.0, !strcmp(saved, "replaced") ? "; old version copied to .trash" : "");
    if (is_icon) ds4_buf_printf(&b, ", %s.\n%s%s", src, log.len ? log.ptr : "", log.len ? "\n" : "");
    else ds4_buf_printf(&b, " from %s; the original is %dx%d. The photo belongs to its owner: fine inside a deck or document, name the source if it is published.\n", src, w, h);
    ds4_buf_printf(&b, "Check the picture above; use it as ![caption](%s) in write_document%s.\n", rel, is_icon ? " (in a card list, put ![](path) right before the bold phrase)" : "");
    *ok = 1;
    free(saved); ds4_buf_free(&log);
    return ds4_buf_take(&b);
}
