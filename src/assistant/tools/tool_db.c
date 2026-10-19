/* tool_db.c — query_db 的执行(方案稿 docs/assistant_connectors.md §2): 参数 → 连接器(开着才行)→ 密码从 secrets 读进栈 → src/db 查 → 头 16 KB 给模型, 全量落工作区 db/。
 * 网络库(MongoDB / PostgreSQL / MySQL / Redis)在主进程直连 TCP(沙盒子进程没有网络); SQLite 是本机文件, 在沙盒子进程里跑工具箱的 sqlite3 -readonly, 只挂那个文件为只读根。
 * 模型常把 filter 写成 JSON 对象而不是字符串: 两种都收(asst_jraw 拿原文, 是字符串就解一层)。
 * 为什么全量落盘: 模型要画 300 行的图, 16 KB 装不下; 落在 db/<连接器>-<集合>-<时间>.json 里, read_file 分页读或 run 里处理。 */
#include "../asst_internal.h"
#include "../../db/db.h"
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define DB_HEAD ASST_READ_BYTES   /* 给模型的头: 和 read_file 一页同量 */

static char *arg_json(const char *args, const char *key) {   /* 字符串 → 解一层; 对象/数组 → 原文 */
    char *raw = NULL; if (!asst_jraw(args, key, &raw)) return NULL;
    if (raw[0] == '"') { const char *p = raw; char *s = NULL; json_string(&p, &s); free(raw); return s; }
    if (!strcmp(raw, "null")) { free(raw); return NULL; }
    return raw;
}

/* ---- SQLite: 沙盒子进程里 execve sqlite3(不经 bash, SQL 走 argv 不用转义) ---- */
typedef struct { char file[ASST_PATH]; char sql[DB_SQL_MAX + 64]; } sq_args;
static char *sqlite_fn(void *ud) {
    const sq_args *a = ud;
    int pf[2]; if (pipe(pf) < 0) return strdup("error: pipe failed\n");
    const pid_t pid = fork();
    if (pid < 0) return strdup("error: fork failed\n");
    if (pid == 0) {
        const int nul = open("/dev/null", O_RDONLY); if (nul >= 0) dup2(nul, 0);
        dup2(pf[1], 1); dup2(pf[1], 2); close(pf[0]); close(pf[1]);
        char *const argv[] = { "sqlite3", "-readonly", "-batch", "-json", (char *)a->file, (char *)a->sql, NULL };
        char *const envp[] = { "PATH=" ASST_SBX_PATH, "HOME=/", "LANG=C.UTF-8", NULL };   /* HOME 不设 sqlite3 会往输出里插一行 "cannot find home directory" 警告 */
        char dirs[] = ASST_SBX_PATH; char p[ASST_PATH];
        for (char *s = dirs, *d; (d = strsep(&s, ":")); ) { if (!d[0]) continue; snprintf(p, sizeof p, "%s/sqlite3", d); execve(p, argv, envp); }
        _exit(127);
    }
    close(pf[1]);
    ds4_buf out = {0}; char buf[8192]; ssize_t n;
    while ((n = read(pf[0], buf, sizeof buf)) > 0) { if (out.len < ASST_SBX_OUT_MAX) ds4_buf_append(&out, buf, (size_t)n); }
    close(pf[0]);
    int st = 0; waitpid(pid, &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st)) { ds4_buf e = {0}; ds4_buf_printf(&e, "error: sqlite3: %s", out.len ? out.ptr : "exited with an error\n"); ds4_buf_free(&out); return ds4_buf_take(&e); }
    return ds4_buf_take(&out);
}
static char *sqlite_run(const char *file, const char *sql, char *err, size_t errn) {
    if (!asst_toolbox_has("sqlite3")) { snprintf(err, errn, "sqlite3 is not installed on this machine (the sandbox toolbox has no sqlite3)"); return NULL; }
    struct stat stt; if (stat(file, &stt) || !S_ISREG(stt.st_mode)) { snprintf(err, errn, "no such file: %s", file); return NULL; }
    asst_sbx_spec sp; memset(&sp, 0, sizeof sp); sp.timeout_s = 30; sp.exec = true;   /* exec = 系统只读根(sqlite3 自己和它的库); 工作区都不挂, 只挂这个文件 */
    if (!asst_sbx_add(&sp, false, file)) { snprintf(err, errn, "cannot open %s", file); return NULL; }
    sq_args *a = calloc(1, sizeof *a); snprintf(a->file, sizeof a->file, "%s", file); snprintf(a->sql, sizeof a->sql, "%s", sql);   /* 堆上: 子进程不能用别的线程栈上的东西 */
    char *out = NULL; const asst_sbx_rc rc = asst_sbx_run(&sp, sqlite_fn, a, &out); free(a);
    if (rc != ASST_SBX_OK) { snprintf(err, errn, "sandbox: %s", out ? out : "failed"); free(out); return NULL; }
    if (out && !strncmp(out, "error", 5)) { const char *m = out + 5; while (*m == ':' || *m == ' ') m++; snprintf(err, errn, "%s", m); free(out); return NULL; }   /* 去掉自带的 "error:", 调用方会再加 */
    return out ? out : strdup("");
}
/* sqlite3 -json 给的是 [{…},{…}] 一整个数组(空结果什么都不给): 拆成一行一条, 与别家同形 */
static void json_rows(const char *arr, ds4_buf *o, long *rows) {
    const char *p = arr; json_ws(&p); *rows = 0;
    while (*p && *p != '[') { p = strchr(p, '\n'); if (!p) { ds4_buf_puts(o, arr); return; } p++; json_ws(&p); }   /* 数组前若有别的行(sqlite3 的提示)跳过 */
    if (*p != '[') { ds4_buf_puts(o, arr); return; }
    p++;
    for (;;) { json_ws(&p); if (*p == ']' || !*p) break; const char *s = p; if (!json_skip_value(&p)) break; ds4_buf_append(o, s, (size_t)(p - s)); ds4_buf_putc(o, '\n'); (*rows)++; json_ws(&p); if (*p == ',') p++; }
}
static char *sqlite_query(const asst_conn *c, const db_req *r, bool *ok) {
    char err[600]; ds4_buf o = {0}; *ok = false; long rows = 0;
    const char *what = r->what ? r->what : "";
    char *res = NULL;
    if (!strcmp(what, "tables")) res = sqlite_run(c->path, "SELECT name AS \"table\", type AS kind FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name", err, sizeof err);
    else if (!strcmp(what, "fields")) {
        if (!r->coll || !r->coll[0]) { ds4_buf_puts(&o, "error: fields needs collection (the table name)\n"); return ds4_buf_take(&o); }
        char q[600]; ds4_buf qq = {0}; ds4_buf_puts(&qq, "SELECT name AS column_name, type AS data_type, \"notnull\", dflt_value AS column_default, pk FROM pragma_table_info('"); for (const char *s = r->coll; *s; s++) { if (*s == '\'') ds4_buf_putc(&qq, '\''); ds4_buf_putc(&qq, *s); } ds4_buf_puts(&qq, "')");
        snprintf(q, sizeof q, "%s", qq.ptr); ds4_buf_free(&qq);
        res = sqlite_run(c->path, q, err, sizeof err);
    } else if (!strcmp(what, "sql")) {
        const char *why = sql_readonly_check(r->sql);
        if (why) { ds4_buf_printf(&o, "error: %s\n", why); return ds4_buf_take(&o); }
        char *w = sql_wrap_limit(r->sql, r->limit, "sqlite"); res = sqlite_run(c->path, w, err, sizeof err); free(w);
    } else { ds4_buf_puts(&o, "error: what must be tables, fields or sql\n"); return ds4_buf_take(&o); }
    if (!res) { ds4_buf_printf(&o, "error: %s\n", err); return ds4_buf_take(&o); }
    ds4_buf lines = {0}; json_rows(res, &lines, &rows); free(res);
    if (!strcmp(what, "tables")) ds4_buf_printf(&o, "%ld tables in %s:\n", rows, c->path);
    else if (!strcmp(what, "fields")) { if (!rows) { ds4_buf_free(&lines); ds4_buf_printf(&o, "error: no table %s (call what=tables to see the names)\n", r->coll); return ds4_buf_take(&o); } ds4_buf_printf(&o, "%ld columns of %s:\n", rows, r->coll); }
    else ds4_buf_printf(&o, "%ld rows (limit %ld)%s:\n", rows, r->limit, rows == r->limit ? "; there may be more" : "");
    ds4_buf_append(&o, lines.ptr, lines.len); ds4_buf_free(&lines);
    *ok = true;
    return ds4_buf_take(&o);
}
bool asst_sqlite_probe(const char *path, char *line, size_t n, char *err, size_t errn) {
    char *v = sqlite_run(path, "SELECT sqlite_version() AS v, (SELECT count(*) FROM sqlite_master WHERE type IN ('table','view')) AS t", err, errn);
    if (!v) return false;
    char ver[32] = "?", cnt[32] = "?";
    const char *p = strstr(v, "\"v\":\""); if (p) snprintf(ver, sizeof ver, "%.*s", (int)strcspn(p + 5, "\""), p + 5);
    p = strstr(v, "\"t\":"); if (p) snprintf(cnt, sizeof cnt, "%.*s", (int)strspn(p + 4, "0123456789"), p + 4);
    free(v);
    snprintf(line, n, "SQLite %s · %s · %s tables", ver, path, cnt);
    return true;
}

/* 结果交给模型: 头 DB_HEAD 字节; 全量落 db/<连接器>-<集合或 what>-<时刻>.json, 截了就在末尾指路 */
static char *deliver(char *full, const char *conn, const char *tag) {
    const size_t n = strlen(full);
    char d[ASST_PATH + 8]; snprintf(d, sizeof d, "%s/db", asst_ws); mkdir(d, 0700);
    char safe[64]; snprintf(safe, sizeof safe, "%.48s", tag && tag[0] ? tag : "result"); for (char *c = safe; *c; c++) if (!((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') || (*c >= '0' && *c <= '9') || *c == '_' || *c == '-' || *c == '.')) *c = '_';
    time_t now = time(NULL); struct tm tm; localtime_r(&now, &tm);
    char rel[256]; snprintf(rel, sizeof rel, "db/%s-%s-%04d%02d%02dT%02d%02d%02d.json", conn, safe, tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    char path[ASST_PATH + 300]; snprintf(path, sizeof path, "%s/%s", asst_ws, rel);
    const bool saved = asst_write_file(path, full, n, 0600);
    if (n <= DB_HEAD) { if (saved) { ds4_buf b = {0}; ds4_buf_append(&b, full, n); ds4_buf_printf(&b, "(full result also saved as %s)\n", rel); free(full); return ds4_buf_take(&b); } return full; }
    size_t cut = DB_HEAD; while (cut > 0 && full[cut] != '\n') cut--;   /* 在整行处截 */
    ds4_buf b = {0}; ds4_buf_append(&b, full, cut);
    ds4_buf_printf(&b, "\n…[%zu of %zu bytes shown; the full result is saved as %s: read_file it with start= to page, or process it with run]\n", cut, n, saved ? rel : "(could not be saved)");
    free(full);
    return ds4_buf_take(&b);
}

char *asst_db_exec(const char *sid, const char *args, int *ok) {
    (void)sid; *ok = 0;
    char *cn = NULL, *what = NULL; asst_jstr(args, "connector", &cn); asst_jstr(args, "what", &what);
    asst_conn c;
    if (!cn || !asst_conn_load(cn, &c)) { free(cn); free(what); return strdup("error: no such connector\n"); }
    if (!c.enabled) { free(cn); free(what); return strdup("error: this connector is turned off by the user\n"); }
    const db_kind *k = db_kind_find(c.kind);
    char tag[40]; snprintf(tag, sizeof tag, ",%s,", what ? what : ""); char whats[300]; snprintf(whats, sizeof whats, ",%s,", k->whats);
    if (!strstr(whats, tag)) { ds4_buf b = {0}; ds4_buf_printf(&b, "error: for a %s connector what must be one of: %s\n", k->label, k->whats); free(cn); free(what); return ds4_buf_take(&b); }
    db_req r = { .what = what };
    char *coll = arg_json(args, "collection"), *filter = arg_json(args, "filter"), *proj = arg_json(args, "projection"), *sort = arg_json(args, "sort"), *pipe = arg_json(args, "pipeline"), *field = arg_json(args, "field"), *sql = arg_json(args, "sql"), *key = arg_json(args, "key"), *pat = arg_json(args, "pattern"), *cmd = arg_json(args, "command");
    r.coll = coll; r.filter = filter; r.projection = proj; r.sort = sort; r.pipeline = pipe; r.field = field; r.sql = sql; r.key = key; r.pattern = pat; r.command = cmd;
    long lim = DB_LIMIT_DEFAULT; asst_jint(args, "limit", &lim); r.limit = lim < 1 ? 1 : lim > DB_LIMIT_MAX ? DB_LIMIT_MAX : lim;
    bool done = false; char *out;
    if (!strcmp(c.kind, "sqlite")) out = sqlite_query(&c, &r, &done);
    else {
        char pw[257] = ""; asst_conn_password(c.name, pw, sizeof pw);
        const db_target t = { c.kind, c.host, c.db, c.user, pw, c.auth_db, c.path, c.port };
        out = db_query(&t, &r, &done);
        memset(pw, 0, sizeof pw);
    }
    if (done) out = deliver(out, c.name, coll ? coll : what);
    *ok = done;
    free(cn); free(what); free(coll); free(filter); free(proj); free(sort); free(pipe); free(field); free(sql); free(key); free(pat); free(cmd);
    return out;
}
