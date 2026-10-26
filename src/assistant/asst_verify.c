/* asst_verify.c — 校验相(2026-10-23, 用户: "鹈鹕测试只是实现了功能, 没有复验和修正; 是一个 agent 范式, 不是单独在生成网页的时候起作用; 不要硬编码")。方案稿 docs/assistant_verify.md。
 * 做 → 核 → 答: 这一趟动过手(调过工具)的运行, 模型第一次想收口时框架不结束, 而是摆一条事实消息再给它一步: 这趟调了什么、产出/改动了哪些文件、
 *   哪些产出改完后它没再看过、回答里点名了却不存在的文件。核什么、怎么核、要不要修, 全是模型的事(它有 view_file / read_file / run 这些手);
 *   框架不按文件类型写一条判定规则 —— 判定规则是硬编码, 今天写"网页要有动画"明天就要写"表格要有合计", 没完。
 * 为什么是框架给而不是提示词说"写完要看一眼": 10-20 九趟鹈鹕, 模型从没看过自己写的页面, 只有 2 趟自己想起来调 render_image; 10-13 加进提示词的
 *   "别在思考里先写一遍"实测它不听。框架机械地在收口前把事实摆出来, 每趟都摆。
 * 账本只认工具真做了的事: 产出工具成功 → 它的路径进账; run → 执行前后扫工作区比出改了什么; 看的工具 → 记"看过"。页面与服务端不猜意图(铁律)。
 * 轮数上限 ASST_VERIFY_ROUNDS: 第 2 条之后无论如何结束, 用户看得见终稿 —— 禁兜底不等于禁事实, 两轮后还没修好就该让它如实说。 */
#include "asst_verify.h"
#include <dirent.h>
#include <sys/stat.h>

void asst_ledger_init(asst_ledger *l) { memset(l, 0, sizeof *l); }

static bool rel_of(const char *abs, char *rel, size_t n) {   /* 工作区内 → 相对路径; 点开头目录(.trash/.view/.img/.web/.video/.phone)不进账: 内部产物 */
    if (!abs || !abs[0] || !asst_under(abs, asst_ws) || strlen(abs) <= strlen(asst_ws)) return false;
    snprintf(rel, n, "%s", abs + strlen(asst_ws) + 1);
    return rel[0] && rel[0] != '.' && !strstr(rel, "/.");
}

static asst_ledger_item *find(asst_ledger *l, const char *rel) { for (int i = 0; i < l->n; i++) if (!strcmp(l->it[i].rel, rel)) return &l->it[i]; return NULL; }

static void wrote(asst_ledger *l, const char *rel, const char *by) {
    asst_ledger_item *x = find(l, rel);
    if (!x) { if (l->n == ASST_LEDGER_MAX) { l->more = true; return; } x = &l->it[l->n++]; memset(x, 0, sizeof *x); snprintf(x->rel, sizeof x->rel, "%s", rel); }
    snprintf(x->by, sizeof x->by, "%s", by); x->wrote_at = l->step;
    char abs[ASST_PATH + 8]; struct stat st; snprintf(abs, sizeof abs, "%s/%s", asst_ws, rel);
    x->size = stat(abs, &st) ? 0 : (long)st.st_size;
}

static void tally(asst_ledger *l, const char *tool) {
    for (int i = 0; i < l->ntool; i++) if (!strcmp(l->tool[i].name, tool)) { l->tool[i].count++; return; }
    if (l->ntool < ASST_LEDGER_TOOLS) { snprintf(l->tool[l->ntool].name, sizeof l->tool[0].name, "%s", tool); l->tool[l->ntool++].count = 1; }
}

/* 产出工具 = 守门按写类走、g->path 是产物的; 看的工具 = 读类里拿单个文件的。edit_image 的成品在结果开头的 "image: 路径" 行里(源图不是产物) */
static bool writes_path(const char *t) { return !strcmp(t, "write_file") || !strcmp(t, "edit_file") || !strcmp(t, "write_document") || !strcmp(t, "write_xlsx") || !strcmp(t, "render_image") || !strcmp(t, "make_video") || !strcmp(t, "fetch_image"); }
static bool looks(const char *t) { return !strcmp(t, "view_file") || !strcmp(t, "read_file") || !strcmp(t, "view_image"); }

void asst_ledger_note(asst_ledger *l, const char *tool, const asst_gres *g, const char *result, bool ok) {
    l->step++; tally(l, tool);
    if (!ok) return;
    char rel[ASST_PATH];
    if (writes_path(tool) && rel_of(g->path, rel, sizeof rel)) wrote(l, rel, tool);
    else if (!strcmp(tool, "edit_image")) {
        for (const char *ln = result; ln && !strncmp(ln, "image: ", 7); ln = strchr(ln, '\n') ? strchr(ln, '\n') + 1 : NULL) {
            const char *nl = strchr(ln, '\n'); const size_t L = nl ? (size_t)(nl - ln - 7) : strlen(ln + 7);
            if (L && L < sizeof rel) { snprintf(rel, sizeof rel, "%.*s", (int)L, ln + 7); if (rel[0] != '.' && !strstr(rel, "/.")) wrote(l, rel, tool); }
        }
    } else if (looks(tool) && rel_of(g->path, rel, sizeof rel)) { asst_ledger_item *x = find(l, rel); if (x) x->seen_at = l->step; }
}

/* ---------------- run 前后的工作区快照: 不进点开头的目录与文件; 按路径排好, 比对时两边归并 ---------------- */
typedef struct { char *rel; long mtime, size; } snap_ent;
typedef struct { snap_ent *e; int n, cap; bool more; } snap;

static void walk(snap *s, const char *dir, const char *relbase) {
    DIR *d = opendir(dir); if (!d) return;
    struct dirent *de;
    while ((de = readdir(d))) {
        if (de->d_name[0] == '.') continue;
        char p[ASST_PATH * 2], r[ASST_PATH * 2]; snprintf(p, sizeof p, "%s/%s", dir, de->d_name); snprintf(r, sizeof r, "%s%s%s", relbase, relbase[0] ? "/" : "", de->d_name);
        struct stat st; if (lstat(p, &st)) continue;
        if (S_ISDIR(st.st_mode)) { walk(s, p, r); continue; }
        if (!S_ISREG(st.st_mode)) continue;
        if (s->n >= ASST_LEDGER_SCAN_MAX) { s->more = true; break; }
        if (s->n == s->cap) { s->cap = s->cap ? s->cap * 2 : 256; s->e = realloc(s->e, (size_t)s->cap * sizeof *s->e); }
#ifdef __APPLE__
        const long mt = (long)st.st_mtimespec.tv_sec * 1000 + st.st_mtimespec.tv_nsec / 1000000;
#else
        const long mt = (long)st.st_mtim.tv_sec * 1000 + st.st_mtim.tv_nsec / 1000000;
#endif
        s->e[s->n++] = (snap_ent){ strdup(r), mt, (long)st.st_size };
    }
    closedir(d);
}
static int cmp_ent(const void *a, const void *b) { return strcmp(((const snap_ent *)a)->rel, ((const snap_ent *)b)->rel); }
static snap *take(void) { snap *s = calloc(1, sizeof *s); walk(s, asst_ws, ""); if (s->n) qsort(s->e, (size_t)s->n, sizeof *s->e, cmp_ent); return s; }
static void drop(snap *s) { if (!s) return; for (int i = 0; i < s->n; i++) free(s->e[i].rel); free(s->e); free(s); }

void *asst_ledger_snap(void) { return take(); }

void asst_ledger_diff(asst_ledger *l, void *before, const char *by) {
    snap *a = before, *b = take();
    if (a->more || b->more) l->more = true;
    for (int i = 0, j = 0; j < b->n; ) {   /* 归并: b 里有而 a 里没有的 = 新增; 两边都有但 mtime/size 不同 = 改动 */
        const int c = i < a->n ? strcmp(a->e[i].rel, b->e[j].rel) : 1;
        if (c < 0) { i++; continue; }
        if (c > 0 || a->e[i].mtime != b->e[j].mtime || a->e[i].size != b->e[j].size) wrote(l, b->e[j].rel, by);
        if (c == 0) i++;
        j++;
    }
    drop(a); drop(b);
}

/* ---------------- 收口前的事实消息 ---------------- */
static void human_size(long n, char *out, size_t on) { if (n >= 1 << 20) snprintf(out, on, "%.1f MB", n / 1048576.0); else if (n >= 1024) snprintf(out, on, "%.1f KB", n / 1024.0); else snprintf(out, on, "%ld B", n); }

char *asst_verify_note(asst_ledger *l, const char *answer) {
    if (l->rounds >= ASST_VERIFY_ROUNDS) return NULL;
    char *missing = asst_file_mentioned_missing(answer);
    bool changed = false;   /* 上一条之后又写了东西 → 第二轮要核; 第一轮只要动过手就核 */
    for (int i = 0; i < l->n; i++) if (l->it[i].wrote_at > l->noted_step) changed = true;
    if (!(l->rounds == 0 ? l->step > 0 : changed) && !missing) { free(missing); return NULL; }
    ds4_buf b = {0};
    ds4_buf_printf(&b, "[verify before answering — check %d of %d. This run so far: %d tool call%s", l->rounds + 1, ASST_VERIFY_ROUNDS, l->step, l->step == 1 ? "" : "s");
    for (int i = 0; i < l->ntool; i++) ds4_buf_printf(&b, "%s%s %d", i ? ", " : " (", l->tool[i].name, l->tool[i].count);
    ds4_buf_puts(&b, l->ntool ? "). " : ". ");
    if (l->n) {
        ds4_buf_puts(&b, "Files produced or changed: ");
        for (int i = 0; i < l->n; i++) {
            const asst_ledger_item *x = &l->it[i]; char sz[32]; human_size(x->size, sz, sizeof sz);
            ds4_buf_printf(&b, "%s%s (%s, %s%s, %s)", i ? ", " : "", x->rel, sz, x->by, x->wrote_at > l->noted_step && l->rounds ? ", changed since the last check" : "",
                           x->seen_at >= x->wrote_at && x->seen_at ? "looked at after that" : "not looked at since");
        }
        ds4_buf_puts(&b, l->more ? " (and more not listed). " : ". ");
    }
    if (missing) {
        ds4_buf_puts(&b, "Named in your answer but not in the workspace: ");
        char *el = NULL; for (int i = 0; asst_jarr_at(missing, i, &el); i++) { const char *p = el; char *s = NULL; if (json_string(&p, &s)) ds4_buf_printf(&b, "%s%s", i ? ", " : "", s); free(s); free(el); el = NULL; }
        ds4_buf_puts(&b, ". ");
    }
    ds4_buf_puts(&b, "Before the final answer, check the work against what the user asked: look at what you produced the way the user will see it (view_file), check the facts you state against the tool results, fix what is wrong, then answer again in full.");
    if (l->rounds + 1 == ASST_VERIFY_ROUNDS) ds4_buf_puts(&b, " This is the last check in this run: after your next answer the run ends, so say plainly what could not be fixed.");
    ds4_buf_putc(&b, ']');
    l->rounds++; l->noted_step = l->step;
    free(missing);
    return ds4_buf_take(&b);
}
