/* tool_view.c — view_file 的执行点: 模型"像用户那样看"一个工作区文件, 用来核自己的活(校验相 asst_verify.c 的那只手; 方案稿 docs/assistant_verify.md)。
 * 一个工具一个概念, 实现里按格式走不同的路是不可避免的, 但对外只有"看": 网页 → 火狐渲染截图(有动的话 1 s 后再截一张) + 页面自己报的脚本错误 /
 *   被拦的外网 / 没加载到的资源; 图 → 图本身; 视频 → make_video 存下的首中末帧; docx/pptx/xlsx → 正文(tool_view_doc.c); pdf → 工具箱有 pdftotext 就抽字;
 *   别的 → 开头 16 KB。只报事实, 不判好坏 —— 判是模型的事(10-23 用户: 不要硬编码)。
 * 网页走 make_video / render_image 同一条火狐路(tool_video_srv.c 的本机只读 HTTP 源 + tool_video.c 的虚拟时钟预载): 页面挂在它在工作区里的相对路径上,
 *   同目录的 css/js/图按相对路径都取得到; 没有网络, 引了外网就是一条 CSP 事实。截图落工作区 .view/(点开头: 不进账本、不进预览口令子树)。
 * 出错会怎样: 本机没火狐 → 网页类回"这台机器渲染不了, 用户在文件卡里看", 别的类照常; 页面脚本死循环 → 火狐 20 s 没回音被杀, 回"页面没响应"。 */
#include "../asst_verify.h"
#include "../../office/ds4_office.h"
#include <dirent.h>
#include <strings.h>
#include <sys/stat.h>

#define VIEW_W 1280
#define VIEW_H 800
#define VIEW_TEXT_MAX ASST_READ_BYTES
#define VIEW_TIMES_MAX 4      /* 一次最多看几个时刻: 每张 1280×800 图进上下文上千 token, 再多就该分几次看 */
#define VIEW_TIME_LAST 300.0  /* 最晚看到第几秒: 与 make_video 的时长上限同 */

static const char *ext_of(const char *p) { const char *dot = strrchr(p, '.'), *sl = strrchr(p, '/'); return dot && (!sl || dot > sl) ? dot + 1 : ""; }
static bool is_ext(const char *e, const char *list) {   /* list = "html htm svg" */
    const size_t n = strlen(e); if (!n) return false;
    for (const char *p = list; *p; ) { const char *q = p + strcspn(p, " "); if ((size_t)(q - p) == n && !strncasecmp(p, e, n)) return true; p = *q ? q + 1 : q; }
    return false;
}
static void rel_of(const char *abs, char *rel, size_t n) { snprintf(rel, n, "%s", asst_under(abs, asst_ws) && strlen(abs) > strlen(asst_ws) ? abs + strlen(asst_ws) + 1 : abs); }

/* 一块字节落到工作区 .view/<名>(沙盒子进程写, 旧版进 .trash); 成功写 "image: 相对路径" 一行 */
static bool save_shot(const asst_gres *g, const char *trash, const char *name, const uint8_t *png, size_t len, ds4_buf *out) {
    asst_gres h = *g; h.rw = true; snprintf(h.path, sizeof h.path, "%s/.view/%s", asst_ws, name);
    char *r = NULL; const asst_sbx_rc rc = asst_save_bytes(&h, trash, png, len, &r);
    const bool ok = rc == ASST_SBX_OK && r && strncmp(r, "error", 5);
    if (ok) ds4_buf_printf(out, "image: .view/%s\n", name);
    free(r); return ok;
}

static char *shot(const char *ctx, size_t *len, char *err, size_t errn) {
    ds4_buf c = {0}; ds4_buf_puts(&c, "{\"context\":"); ds4_json_escape(&c, ctx); ds4_buf_puts(&c, ",\"origin\":\"viewport\",\"format\":{\"type\":\"image/png\"}}");
    char *s = asst_ff_call("browsingContext.captureScreenshot", c.ptr, err, errn), *data = NULL; ds4_buf_free(&c);
    if (!s || !asst_jstr(s, "data", &data)) { free(s); return NULL; }
    free(s);
    char *png = malloc(strlen(data) * 3 / 4 + 4); *len = asst_b64dec(data, (uint8_t *)png); free(data);
    return png;
}

static char *view_web(const char *abs, const asst_gres *g, const char *trash, long w, long h, const double *times, int nt, int *ok) {
    char rel[ASST_PATH]; rel_of(abs, rel, sizeof rel);
    if (!asst_browser_kind()[0]) { ds4_buf b = {0}; ds4_buf_printf(&b, "%s cannot be rendered on this machine (no browser installed); the user sees it rendered in the file card under your answer. Use read_file to check the source.\n", rel); *ok = 1; return ds4_buf_take(&b); }
    size_t n = 0; char *page = asst_slurp(abs, &n);
    if (!page) return strdup("error: cannot read the file\n");
    /* 预载 = 视频工具的虚拟时钟(含脚本错误 / CSP 拦截收集) + 一个捕获阶段的资源加载失败监听(img/script/link 的 error 不冒泡到 window) */
    ds4_buf pre = {0};
    ds4_buf_printf(&pre, "() => { (%s)(); if (!window.__ds4v) { window.__ds4v = []; window.addEventListener('error', e => { const t = e.target; if (t && t !== window && (t.src || t.href) && window.__ds4v.length < 50) window.__ds4v.push(String(t.src || t.href).split('/').slice(3).join('/')); }, true); } }", asst_ff_clock_js());
    char url[ASST_PATH + 128], err[512] = "", ctx[128] = "", script[128] = "";
    const bool inws = asst_under(abs, asst_ws) && strlen(abs) > strlen(asst_ws);
    asst_vsrv *srv = asst_vsrv_start_at(page, n, inws ? rel : strrchr(abs, '/') + 1, url, sizeof url, err, sizeof err);
    free(page);
    if (!srv) { ds4_buf_free(&pre); ds4_buf b = {0}; ds4_buf_printf(&b, "error: %s\n", err); return ds4_buf_take(&b); }
    if (!asst_ff_tab_open((int)w, (int)h, pre.ptr, ctx, sizeof ctx, script, sizeof script, err, sizeof err)) { ds4_buf_free(&pre); asst_vsrv_stop(srv); ds4_buf b = {0}; ds4_buf_printf(&b, "error: browser: %s\n", err); return ds4_buf_take(&b); }
    ds4_buf_free(&pre);
    ds4_buf q = {0}; ds4_buf_puts(&q, "{\"context\":"); ds4_json_escape(&q, ctx); ds4_buf_puts(&q, ",\"url\":"); ds4_json_escape(&q, url); ds4_buf_puts(&q, ",\"wait\":\"complete\"}");
    char *r = asst_ff_call("browsingContext.navigate", q.ptr, err, sizeof err); ds4_buf_free(&q); free(r);
    char *ready = r ? asst_ff_eval(ctx, "document.fonts.ready.then(() => Promise.all([...document.images].map(i => i.complete ? 0 : new Promise(f => { i.onload = i.onerror = f; })))).then(() => typeof __ds4)", true, err, sizeof err) : NULL;
    const bool loaded = ready && !strcmp(ready, "\"object\""); free(ready);
    ds4_buf b = {0}, facts = {0};
    char *png[VIEW_TIMES_MAX] = {0}, *perr = NULL, *pres = NULL; size_t plen[VIEW_TIMES_MAX] = {0}; int got = 0;
    if (loaded) {
        for (int i = 0; i < nt; i++) {   /* 虚拟时钟按时刻拨(时刻已升序): 动画停在那一刻再截, 两张图的时间差严格等于给的差 */
            char expr[64]; snprintf(expr, sizeof expr, "__ds4.step(%.3f)", times[i] * 1000.0);
            char *v = asst_ff_eval(ctx, expr, false, err, sizeof err);
            if (!v) { ds4_buf_printf(&facts, "The page stopped responding when time was advanced to %g s (%s).\n", times[i], err[0] ? err : "endless loop in a script?"); break; }
            free(v);
            if (!(png[i] = shot(ctx, &plen[i], err, sizeof err))) break;
            got = i + 1;
        }
        char e2[256] = "";
        ds4_buf x = {0}; ds4_buf_puts(&x, "(location.href === "); ds4_json_escape(&x, url); ds4_buf_puts(&x, " ? '' : 'the page navigated away to ' + location.href + '\\n') + __ds4.errors()");
        char *raw = asst_ff_eval(ctx, x.ptr, false, e2, sizeof e2); ds4_buf_free(&x);
        const char *pp = raw; if (raw && raw[0] == '"') json_string(&pp, &perr); free(raw);
        raw = asst_ff_eval(ctx, "(window.__ds4v || []).join('\\n')", false, e2, sizeof e2);
        pp = raw; if (raw && raw[0] == '"') json_string(&pp, &pres); free(raw);
    }
    asst_ff_tab_close(ctx, script); asst_vsrv_stop(srv);
    if (!got) { for (int i = 0; i < nt; i++) free(png[i]); free(perr); free(pres); ds4_buf_free(&facts); ds4_buf_printf(&b, "error: %s\n", err[0] ? err : loaded ? "screenshot failed" : "the page did not load in the renderer"); return ds4_buf_take(&b); }
    /* 截图按工作区相对路径命名(/ 换成 _): 只按文件名时 a/index.html 与 b/index.html 互相覆盖(10-23 实撞) */
    char flat[ASST_PATH]; snprintf(flat, sizeof flat, "%s", rel); for (char *c = flat; *c; c++) if (*c == '/') *c = '_';
    ds4_buf said = {0};
    for (int i = 0; i < got; i++) {
        const bool same = i && plen[i] == plen[i - 1] && !memcmp(png[i], png[i - 1], plen[i]);
        if (same) { ds4_buf_printf(&said, "%s%g s identical to %g s", said.len ? ", " : "", times[i], times[i - 1]); continue; }
        char name[ASST_PATH + 32]; snprintf(name, sizeof name, "%s.%gs.png", flat, times[i]);
        save_shot(g, trash, name, (uint8_t *)png[i], plen[i], &b);
        ds4_buf_printf(&said, "%spicture at %g s", said.len ? ", " : "", times[i]);
    }
    ds4_buf_printf(&b, "%s rendered at %ldx%ld the way the user sees it: %s.\n", rel, w, h, said.ptr);
    if (got > 1 && strstr(said.ptr, "identical") && !strstr(said.ptr, ", picture at")) ds4_buf_puts(&b, "Nothing moves between those moments.\n");
    ds4_buf_free(&said);
    if (facts.len) ds4_buf_append(&b, facts.ptr, facts.len);
    if (perr && perr[0]) ds4_buf_printf(&b, "Page errors reported by the browser:\n%.2000s\n", perr); else ds4_buf_puts(&b, "No page errors.\n");
    if (pres && pres[0]) ds4_buf_printf(&b, "Resources that failed to load (missing file or blocked external URL):\n%.1000s\n", pres);
    for (int i = 0; i < nt; i++) free(png[i]);
    free(perr); free(pres); ds4_buf_free(&facts);
    *ok = 1;
    return ds4_buf_take(&b);
}

static char *view_image(const char *abs, int *ok) {
    char rel[ASST_PATH]; rel_of(abs, rel, sizeof rel);
    size_t n = 0; char *d = asst_slurp(abs, &n);
    if (!d) return strdup("error: cannot read the file\n");
    office_imginfo info; const bool sized = office_img_info((const uint8_t *)d, n, &info); free(d);
    ds4_buf b = {0}; ds4_buf_printf(&b, "image: %s\n%s: ", rel, rel);
    if (sized) ds4_buf_printf(&b, "%s %dx%d, ", info.ext, info.w, info.h);
    ds4_buf_printf(&b, "%.0f KB. The picture above is what the user sees.\n", n / 1024.0);
    *ok = 1; return ds4_buf_take(&b);
}

static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }
static char *view_video(const char *abs, int *ok) {   /* make_video 存的首中末帧在 .video/<名>/frame_*.png */
    char rel[ASST_PATH]; rel_of(abs, rel, sizeof rel);
    const char *base = strrchr(rel, '/') ? strrchr(rel, '/') + 1 : rel; const char *dot = strrchr(base, '.');
    char dir[ASST_PATH * 2]; snprintf(dir, sizeof dir, "%s/.video/%.*s", asst_ws, (int)(dot ? dot - base : (long)strlen(base)), base);
    DIR *dd = opendir(dir); char *names[4096]; int k = 0;
    if (dd) { struct dirent *de; while ((de = readdir(dd)) && k < 4096) if (!strncmp(de->d_name, "frame_", 6) && strstr(de->d_name, ".png")) names[k++] = strdup(de->d_name); closedir(dd); }
    ds4_buf b = {0};
    if (!k) { struct stat st; ds4_buf_printf(&b, "%s: %.1f MB; no saved frames for it (frames are kept only for videos made by make_video in this workspace). The user plays it in the file card.\n", rel, stat(abs, &st) ? 0.0 : st.st_size / 1048576.0); *ok = 1; return ds4_buf_take(&b); }
    qsort(names, (size_t)k, sizeof *names, cmp_str);
    const int pick[3] = { 0, k / 2, k - 1 };
    for (int i = 0; i < 3; i++) if (i == 0 || pick[i] != pick[i - 1]) ds4_buf_printf(&b, "image: %s/%s\n", dir + strlen(asst_ws) + 1, names[pick[i]]);
    ds4_buf_printf(&b, "%s: %d frames were saved when it was made; above are the first, middle and last.\n", rel, k);
    for (int i = 0; i < k; i++) free(names[i]);
    *ok = 1; return ds4_buf_take(&b);
}

static char *view_pdf(const char *abs, const asst_gres *g, int *ok) {
    char rel[ASST_PATH]; rel_of(abs, rel, sizeof rel);
    ds4_buf b = {0};
    if (!asst_toolbox_has("pdftotext")) { struct stat st; ds4_buf_printf(&b, "%s: PDF, %.0f KB. This machine has no pdftotext, so the text cannot be shown here; the user opens it in the file card.\n", rel, stat(abs, &st) ? 0.0 : st.st_size / 1024.0); *ok = 1; return ds4_buf_take(&b); }
    asst_sbx_spec sp; asst_guard_spec(g, true, &sp); sp.timeout_s = 35;
    ds4_buf cmd = {0}; ds4_buf_puts(&cmd, "pdftotext -layout "); ds4_json_escape(&cmd, abs); ds4_buf_puts(&cmd, " - | head -c 16384");
    asst_bash_args a = { cmd.ptr, asst_ws, 30 }; char *out = NULL;
    const asst_sbx_rc rc = asst_sbx_run(&sp, asst_bash_exec, &a, &out); ds4_buf_free(&cmd);
    if (rc != ASST_SBX_OK || !out) { free(out); return strdup("error: pdftotext did not run in the sandbox\n"); }
    ds4_buf_printf(&b, "%s: text as extracted from the PDF (layout kept; images are not shown):\n%s\n", rel, out); free(out);
    *ok = 1; return ds4_buf_take(&b);
}

static char *view_text(const char *abs, int *ok) {
    char rel[ASST_PATH]; rel_of(abs, rel, sizeof rel);
    size_t n = 0; char *d = asst_slurp(abs, &n);
    if (!d) return strdup("error: cannot read the file\n");
    ds4_buf b = {0};
    if (memchr(d, 0, n)) { ds4_buf_printf(&b, "%s: binary file, %.0f KB; nothing to show as text.\n", rel, n / 1024.0); free(d); *ok = 1; return ds4_buf_take(&b); }
    size_t cut = n < VIEW_TEXT_MAX ? n : VIEW_TEXT_MAX; while (cut > 0 && cut < n && ((unsigned char)d[cut] & 0xC0) == 0x80) cut--;
    long lines = 0; for (size_t i = 0; i < n; i++) if (d[i] == '\n') lines++;
    ds4_buf_printf(&b, "%s: %ld lines, %.1f KB%s\n", rel, lines, n / 1024.0, cut < n ? " (first 16 KB shown; use read_file with start for the rest):" : ":");
    ds4_buf_append(&b, d, cut); ds4_buf_putc(&b, '\n'); free(d);
    *ok = 1; return ds4_buf_take(&b);
}

char *asst_view_exec(const char *args, const asst_gres *g, const char *trash, int *ok) {
    *ok = 0;
    struct stat st;
    if (!g->path[0] || stat(g->path, &st) || !S_ISREG(st.st_mode)) return strdup("error: no such file\n");
    const char *e = ext_of(g->path);
    long w = VIEW_W, h = VIEW_H; asst_jint(args, "width", &w); asst_jint(args, "height", &h);
    if (w < 64 || w > 4096 || h < 64 || h > 4096) return strdup("error: width and height must be between 64 and 4096 pixels\n");
    if (is_ext(e, "html htm svg")) {   /* times: 看哪几个时刻(秒), 缺省 0 和 1; 升序去重。10-23 鹈鹕那趟模型想看中间一帧, 原话 "view_file gives 0s and 1s only" */
        double t[VIEW_TIMES_MAX] = { 0, 1 }; int nt = 2; char *raw = NULL;
        if (asst_jraw(args, "times", &raw) && strcmp(raw, "null")) {
            nt = 0; char *el = NULL;
            for (int i = 0; asst_jarr_at(raw, i, &el); i++) {
                char *end; const double v = strtod(el[0] == '"' ? el + 1 : el, &end); free(el); el = NULL;
                if (nt == VIEW_TIMES_MAX || !(v >= 0 && v <= VIEW_TIME_LAST)) { free(raw); ds4_buf b = {0}; ds4_buf_printf(&b, "error: times takes 1 to %d moments, each between 0 and %g seconds\n", VIEW_TIMES_MAX, VIEW_TIME_LAST); return ds4_buf_take(&b); }
                int k = nt++; while (k && t[k - 1] > v) { t[k] = t[k - 1]; k--; } t[k] = v;
            }
            int u = 0; for (int i = 0; i < nt; i++) if (!u || t[i] != t[u - 1]) t[u++] = t[i];
            nt = u;
            if (!nt) { free(raw); ds4_buf b = {0}; ds4_buf_printf(&b, "error: times takes 1 to %d moments, each between 0 and %g seconds\n", VIEW_TIMES_MAX, VIEW_TIME_LAST); return ds4_buf_take(&b); }
        }
        free(raw);
        return view_web(g->path, g, trash, w, h, t, nt, ok);
    }
    if (is_ext(e, "png jpg jpeg gif webp bmp")) return view_image(g->path, ok);
    if (is_ext(e, "mp4")) return view_video(g->path, ok);
    if (is_ext(e, "pdf")) return view_pdf(g->path, g, ok);
    if (is_ext(e, "docx pptx xlsx")) {
        char *t = asst_view_office(g->path, VIEW_TEXT_MAX);
        if (!t) return strdup("error: not a readable Office package (zip structure or XML parts missing)\n");
        char rel[ASST_PATH]; rel_of(g->path, rel, sizeof rel);
        ds4_buf b = {0}; ds4_buf_printf(&b, "%s: text content as the reader will see it (layout, images and charts are not shown):\n%s\n", rel, t); free(t);
        *ok = 1; return ds4_buf_take(&b);
    }
    return view_text(g->path, ok);
}
