/* tool_render.c — render_image 的执行点: 模型把图"写"成一页 HTML(内联 SVG / CSS / canvas 脚本) → 火狐按给定像素截一张图 → PNG 落工作区 → 结果带 "image: 路径" 让模型当场核对。
 * 为什么有它(10-18 用户: "PPT 素材图片太少了, 可以调用工具生成"): 本机没有文生图模型, 但有火狐(make_video 的同一条路: 本机只读 HTTP 源 + BiDi 截图), 插画 / 示意图 / 图标 / 封面配图用矢量画出来质量稳定、零新依赖。
 * 分工和 make_video 一样: 主进程只碰火狐; 落盘在沙盒子进程(写工作区的规矩和别的产出工具一致, 旧版先进 .trash)。
 * 出错会怎样: 页面引了外网图 → 页面里 securitypolicyviolation 被记下来, 结果句原文给模型改; 路径不是 .png → 当场拒。 */
#include "../asst_internal.h"
#include "../../office/ds4_office.h"
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>

#define SIDE_MIN 64
#define SIDE_MAX 4096
#define DEF_W 1600
#define DEF_H 900

static const char ERR_JS[] =   /* 预载: 只收集页面错误, 没有虚拟时钟(静态画面, 动画停在加载完成那一刻) */
    "() => {\n"
    " if (window.__ds4e) return;\n"
    " const errs = []; const note = s => { if (errs.length < 50) errs.push(String(s).slice(0, 500)); };\n"
    " window.addEventListener('error', e => note(e.message + (e.filename ? ' (' + e.filename.split('/').pop() + ':' + e.lineno + ')' : '')));\n"
    " window.addEventListener('securitypolicyviolation', e => note('blocked (no network; only workspace files): ' + e.blockedURI));\n"
    " window.__ds4e = () => errs.join('\\n');\n"
    "}";

typedef struct { const char *path, *trash; const uint8_t *png; size_t len; } save_job;   /* 子进程要碰的一切都在堆上(asst_internal.h 的 asst_sbx_fn 注释) */
static char *save_child(void *ud) {
    const save_job *j = ud; ds4_buf b = {0}; char part[ASST_PATH + 8]; snprintf(part, sizeof part, "%s.part", j->path);
    { char dir[ASST_PATH]; snprintf(dir, sizeof dir, "%s", j->path); char *sl = strrchr(dir, '/'); if (sl && sl != dir) { *sl = 0; asst_mkdir_p(dir); } }   /* figures/x.png: 父目录不存在就建(10-18 spark 实撞 "No such file or directory") */
    FILE *f = fopen(part, "wb"); if (!f || fwrite(j->png, 1, j->len, f) != j->len) { if (f) fclose(f); unlink(part); ds4_buf_printf(&b, "error: cannot write %s: %s\n", j->path, strerror(errno)); return ds4_buf_take(&b); }
    fclose(f);
    struct stat sb; const bool existed = !stat(j->path, &sb);
    if (existed) asst_fs_trash(j->trash, j->path);
    if (rename(part, j->path)) { unlink(part); ds4_buf_printf(&b, "error: cannot write %s: %s\n", j->path, strerror(errno)); return ds4_buf_take(&b); }
    ds4_buf_printf(&b, "%s", existed ? "replaced" : "created");
    return ds4_buf_take(&b);
}
/* 一块字节落到 g->path(沙盒子进程里写; 旧版进 .trash): render_image 与 fetch_image 共用。*out = "created" / "replaced" / "error: …" */
asst_sbx_rc asst_save_bytes(const asst_gres *g, const char *trash, const uint8_t *data, size_t len, char **out) {
    save_job *j = calloc(1, sizeof *j); j->path = strdup(g->path); j->trash = trash ? strdup(trash) : NULL; j->png = data; j->len = len;
    asst_sbx_spec sp; asst_guard_spec(g, false, &sp); *out = NULL;
    const asst_sbx_rc rc = asst_sbx_run(&sp, save_child, j, out);
    free((char *)j->path); free((char *)j->trash); free(j);
    return rc;
}

static char *fail(char *html, const char *fmt, const char *why) { free(html); ds4_buf b = {0}; ds4_buf_printf(&b, fmt, why); return ds4_buf_take(&b); }

char *asst_render_exec(const char *args, const asst_gres *g, const char *trash, int *ok) {
    *ok = 0;
    const char *dot = strrchr(g->path, '.'), *sl = strrchr(g->path, '/');
    if (!dot || (sl && dot < sl) || strcasecmp(dot, ".png")) return strdup("error: render_image writes PNG: the path must end with .png\n");
    char *html = NULL; long w = DEF_W, h = DEF_H;
    asst_jstr(args, "html", &html); asst_jint(args, "width", &w); asst_jint(args, "height", &h);
    if (!html || !html[0]) return fail(html, "error: html is required: one self-contained page that draws the picture%s\n", "");
    if (w < SIDE_MIN || w > SIDE_MAX || h < SIDE_MIN || h > SIDE_MAX) { char s[64]; snprintf(s, sizeof s, "%d and %d", SIDE_MIN, SIDE_MAX); return fail(html, "error: width and height must be between %s pixels\n", s); }
    char url[256], err[512] = "", ctx[128] = "", script[128] = "";
    asst_vsrv *srv = asst_vsrv_start(html, strlen(html), url, sizeof url, err, sizeof err);
    if (!srv) return fail(html, "error: %s\n", err);
    if (!asst_ff_tab_open((int)w, (int)h, ERR_JS, ctx, sizeof ctx, script, sizeof script, err, sizeof err)) { asst_vsrv_stop(srv); return fail(html, "error: browser: %s\n", err); }
    ds4_buf q = {0}; ds4_buf_puts(&q, "{\"context\":"); ds4_json_escape(&q, ctx); ds4_buf_puts(&q, ",\"url\":"); ds4_json_escape(&q, url); ds4_buf_puts(&q, ",\"wait\":\"complete\"}");
    char *r = asst_ff_call("browsingContext.navigate", q.ptr, err, sizeof err); ds4_buf_free(&q); free(r);
    /* 字体与图片都到齐再拍(不等的话是系统后备字体 / 裂图); 再等两帧让 CSS 动画的首帧和布局稳定 */
    char *ready = r ? asst_ff_eval(ctx, "document.fonts.ready.then(() => Promise.all([...document.images].map(i => i.complete ? 0 : new Promise(f => { i.onload = i.onerror = f; })))).then(() => new Promise(f => requestAnimationFrame(() => requestAnimationFrame(f)))).then(() => typeof __ds4e)", true, err, sizeof err) : NULL;
    const bool loaded = ready && !strcmp(ready, "\"function\""); free(ready);
    uint8_t *png = NULL; size_t len = 0; char *perr = NULL;
    if (loaded) {
        ds4_buf c = {0}; ds4_buf_puts(&c, "{\"context\":"); ds4_json_escape(&c, ctx); ds4_buf_puts(&c, ",\"origin\":\"viewport\",\"format\":{\"type\":\"image/png\"}}");
        char *shot = asst_ff_call("browsingContext.captureScreenshot", c.ptr, err, sizeof err), *data = NULL; ds4_buf_free(&c);
        if (shot && asst_jstr(shot, "data", &data)) { png = malloc(strlen(data) * 3 / 4 + 4); len = asst_b64dec(data, png); free(data); }
        free(shot);
        char e2[256] = ""; char *raw = asst_ff_eval(ctx, "__ds4e()", false, e2, sizeof e2);
        const char *p = raw; if (raw && raw[0] == '"') json_string(&p, &perr);
        free(raw);
    }
    asst_ff_tab_close(ctx, script); asst_vsrv_stop(srv); free(html);
    if (!png || !len) { free(png); free(perr); ds4_buf b = {0}; ds4_buf_printf(&b, "error: %s\n", err[0] ? err : loaded ? "screenshot failed" : "the page did not load in the renderer"); return ds4_buf_take(&b); }
    office_imginfo info; const bool sized = office_img_info(png, len, &info);
    char *out = NULL; const asst_sbx_rc rc = asst_save_bytes(g, trash, png, len, &out);
    free(png);
    ds4_buf b = {0};
    if (rc != ASST_SBX_OK || !out || !strncmp(out, "error", 5)) { ds4_buf_printf(&b, "error: %s\n", out ? (!strncmp(out, "error: ", 7) ? out + 7 : out) : "sandbox failed"); free(out); free(perr); return ds4_buf_take(&b); }
    char rel[ASST_PATH]; snprintf(rel, sizeof rel, "%s", asst_under(g->path, asst_ws) && strlen(g->path) > strlen(asst_ws) ? g->path + strlen(asst_ws) + 1 : g->path);
    ds4_buf_printf(&b, "image: %s\n%s %s (PNG %dx%d, %.0f KB)%s. ", rel, out, rel, sized ? info.w : (int)w, sized ? info.h : (int)h, len / 1024.0, !strcmp(out, "replaced") ? "; old version copied to .trash" : "");
    if (perr && perr[0]) ds4_buf_printf(&b, "\nPage errors (fix the HTML and call again):\n%.2000s\n", perr);
    else ds4_buf_puts(&b, "No page errors. ");
    ds4_buf_printf(&b, "Check the picture above; use it as ![caption](%s) in write_document, or hand it to the user with share.\n", rel);
    *ok = 1;
    free(out); free(perr);
    return ds4_buf_take(&b);
}
