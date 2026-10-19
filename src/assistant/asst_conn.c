/* asst_conn.c — 连接器(方案稿 docs/assistant_connectors.md): 用户在技能页登记的数据源, 一个一份 connectors/<名>.json, 密码另存 secrets/conn_<名>(0600)。
 * 页面拿到的 JSON 永远没有 password 字段(只有 has_password); 模型那头只在 query_db 的工具描述里看到"名字: 类型, 库名", 密码只在握手那一刻从 secrets 读进栈。
 * 开关: json 里 enabled; 关掉的不进工具表, 模型凭旧名字硬调也拒(tool_db.c)。删除: json 挪进 trash/connectors/, 密码直接删(密码不进回收站)。
 * 种类、字段、默认端口都从 src/db/db.c 的 DB_KINDS 来: 页面下拉按 GET 回的 kinds 画, 加一种库这里和页面都不用改。 */
#include "asst_internal.h"
#include "../db/db.h"
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define CONN_MAX 64

static long now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000L + t.tv_nsec / 1000000; }

static void conn_dir(char *p, size_t n) { snprintf(p, n, "%s/connectors", asst_root); }
static void conn_path(const char *name, char *p, size_t n) { snprintf(p, n, "%s/connectors/%s.json", asst_root, name); }
static void secret_name(const char *name, char *p, size_t n) { snprintf(p, n, "conn_%s", name); for (char *c = p; *c; c++) if (*c == '-') *c = '_'; }   /* secrets 的名字只收 [a-z0-9_] */

bool asst_conn_name_ok(const char *name) {
    if (!name || !name[0] || name[0] == '-' || strlen(name) > ASST_CONN_NAME_MAX) return false;
    for (const char *p = name; *p; p++) if (!((*p >= 'a' && *p <= 'z') || (*p >= '0' && *p <= '9') || *p == '-')) return false;
    return true;
}

static void jget(const char *j, const char *key, char *out, size_t n) { char *s = NULL; asst_jstr(j, key, &s); snprintf(out, n, "%s", s ? s : ""); free(s); }
bool asst_conn_load(const char *name, asst_conn *c) {
    memset(c, 0, sizeof *c);
    if (!asst_conn_name_ok(name)) return false;
    char p[ASST_PATH + 64]; conn_path(name, p, sizeof p);
    char *j = asst_slurp(p, NULL); if (!j) return false;
    snprintf(c->name, sizeof c->name, "%s", name);
    jget(j, "kind", c->kind, sizeof c->kind); jget(j, "host", c->host, sizeof c->host); jget(j, "db", c->db, sizeof c->db); jget(j, "user", c->user, sizeof c->user); jget(j, "auth_db", c->auth_db, sizeof c->auth_db); jget(j, "path", c->path, sizeof c->path);
    long v = 0; asst_jint(j, "port", &v); c->port = (int)v; v = 0; asst_jint(j, "updated", &v); c->updated = v;
    c->enabled = true; asst_jbool(j, "enabled", &c->enabled);
    free(j);
    return db_kind_find(c->kind) != NULL;
}
static bool conn_save(const asst_conn *c) {
    char d[ASST_PATH + 32]; conn_dir(d, sizeof d); mkdir(d, 0700);
    ds4_buf b = {0};
    ds4_buf_puts(&b, "{\"kind\":"); ds4_json_escape(&b, c->kind); ds4_buf_puts(&b, ",\"host\":"); ds4_json_escape(&b, c->host); ds4_buf_printf(&b, ",\"port\":%d,\"db\":", c->port); ds4_json_escape(&b, c->db);
    ds4_buf_puts(&b, ",\"user\":"); ds4_json_escape(&b, c->user); ds4_buf_puts(&b, ",\"auth_db\":"); ds4_json_escape(&b, c->auth_db); ds4_buf_puts(&b, ",\"path\":"); ds4_json_escape(&b, c->path);
    ds4_buf_printf(&b, ",\"enabled\":%s,\"updated\":%ld}\n", c->enabled ? "true" : "false", c->updated);
    char p[ASST_PATH + 64]; conn_path(c->name, p, sizeof p);
    const bool ok = asst_write_file(p, b.ptr, b.len, 0600); ds4_buf_free(&b);
    return ok;
}
static int scan(char names[CONN_MAX][ASST_CONN_NAME_MAX + 1]) {   /* 按名排序: 工具表顺序稳定 ⇒ KV 前缀稳定 */
    char d[ASST_PATH + 32]; conn_dir(d, sizeof d);
    DIR *dir = opendir(d); if (!dir) return 0;
    int n = 0; struct dirent *e;
    while ((e = readdir(dir)) && n < CONN_MAX) {
        const size_t l = strlen(e->d_name);
        if (l < 6 || strcmp(e->d_name + l - 5, ".json")) continue;
        char nm[64]; snprintf(nm, sizeof nm, "%.*s", (int)(l - 5), e->d_name);
        if (asst_conn_name_ok(nm)) snprintf(names[n++], ASST_CONN_NAME_MAX + 1, "%s", nm);
    }
    closedir(dir);
    for (int i = 1; i < n; i++) for (int j = i; j > 0 && strcmp(names[j - 1], names[j]) > 0; j--) { char t[ASST_CONN_NAME_MAX + 1]; memcpy(t, names[j], sizeof t); memcpy(names[j], names[j - 1], sizeof t); memcpy(names[j - 1], t, sizeof t); }
    return n;
}
bool asst_conn_enabled(const char *name) { asst_conn c; return asst_conn_load(name, &c) && c.enabled; }
bool asst_conn_password(const char *name, char *out, size_t n) { char s[48]; secret_name(name, s, sizeof s); return asst_secret_get(s, out, n); }
static bool has_password(const char *name) { char pw[257]; const bool h = asst_conn_password(name, pw, sizeof pw); memset(pw, 0, sizeof pw); return h; }

/* query_db 的 schema: 开着的连接器一个都没有 → false(工具表里没这行)。描述列连接器 + 用到的种类各一句怎么查; what 枚举是用到的种类的并集 */
bool asst_conn_build_tool(const asst_tctx *ctx, ds4_buf *b) {
    (void)ctx;
    char names[CONN_MAX][ASST_CONN_NAME_MAX + 1]; const int all = scan(names);
    asst_conn cs[CONN_MAX]; int n = 0;
    for (int i = 0; i < all; i++) if (asst_conn_load(names[i], &cs[n]) && cs[n].enabled) n++;
    if (!n) return false;
    ds4_buf d = {0};
    ds4_buf_puts(&d, "Read-only queries against the user's own databases (connectors the user set up and turned on). Nothing here can write, delete or run server-side code. "
                     "Start with the listing call (collections / tables / keys) and then fields on one collection or table to learn its real keys and types before querying; never guess names. "
                     "The result is a summary line followed by one JSON object per row; when it is cut, the full result is saved in the workspace under db/ and can be read with read_file. "
                     "To chart data, write one self-contained .html with inline SVG or canvas (no network in the preview). Connectors:");
    const db_kind *used[8] = {0}; int nused = 0;
    for (int i = 0; i < n; i++) {
        const db_kind *k = db_kind_find(cs[i].kind);
        if (!strcmp(k->kind, "sqlite")) ds4_buf_printf(&d, "\n- %s: %s, file %s", cs[i].name, k->label, cs[i].path);
        else ds4_buf_printf(&d, "\n- %s: %s at %s:%d%s%s", cs[i].name, k->label, cs[i].host, cs[i].port ? cs[i].port : k->port, cs[i].db[0] ? ", database " : "", cs[i].db);
        bool seen = false; for (int j = 0; j < nused; j++) seen |= used[j] == k;
        if (!seen && nused < 8) used[nused++] = k;
    }
    ds4_buf_puts(&d, "\nHow each kind is queried:");
    for (int j = 0; j < nused; j++) ds4_buf_printf(&d, "\n- %s", used[j]->hint);
    ds4_buf_puts(b, "{\"type\":\"function\",\"function\":{\"name\":\"query_db\",\"description\":"); ds4_json_escape(b, d.ptr); ds4_buf_free(&d);
    ds4_buf_puts(b, ",\"parameters\":{\"type\":\"object\",\"properties\":{\"connector\":{\"type\":\"string\",\"description\":\"Which connector\",\"enum\":[");
    for (int i = 0; i < n; i++) ds4_buf_printf(b, "%s\"%s\"", i ? "," : "", cs[i].name);
    ds4_buf_puts(b, "]},\"what\":{\"type\":\"string\",\"description\":\"What to do; the choices depend on the connector's kind (see the description)\",\"enum\":[");
    int k = 0; char seen[512] = "";
    for (int j = 0; j < nused; j++) { char ws[256]; snprintf(ws, sizeof ws, "%s", used[j]->whats);
        for (char *s = ws, *w; (w = strsep(&s, ",")); ) { char tag[40]; snprintf(tag, sizeof tag, ",%s,", w); if (strstr(seen, tag)) continue; strncat(seen, tag, sizeof seen - strlen(seen) - 1); ds4_buf_printf(b, "%s\"%s\"", k++ ? "," : "", w); } }
    ds4_buf_puts(b, "]},"
        "\"collection\":{\"type\":\"string\",\"description\":\"Collection or table name (for fields, find, count, distinct, aggregate)\"},"
        "\"filter\":{\"type\":\"string\",\"description\":\"MongoDB query filter as a JSON string, e.g. {\\\"symbol\\\":\\\"600519\\\",\\\"date\\\":{\\\"$gte\\\":{\\\"$date\\\":\\\"2026-10-01\\\"}}}\"},"
        "\"projection\":{\"type\":\"string\",\"description\":\"MongoDB projection as a JSON string, e.g. {\\\"date\\\":1,\\\"close\\\":1} (keeps rows small)\"},"
        "\"sort\":{\"type\":\"string\",\"description\":\"MongoDB sort as a JSON string, e.g. {\\\"date\\\":-1}\"},"
        "\"pipeline\":{\"type\":\"string\",\"description\":\"MongoDB aggregation pipeline as a JSON array string\"},"
        "\"field\":{\"type\":\"string\",\"description\":\"Field name for distinct\"},"
        "\"sql\":{\"type\":\"string\",\"description\":\"One read-only SQL statement (for what=sql); LIMIT is applied for you\"},"
        "\"key\":{\"type\":\"string\",\"description\":\"Redis key for get\"},"
        "\"pattern\":{\"type\":\"string\",\"description\":\"Redis key pattern for keys, e.g. user:* (default *)\"},"
        "\"command\":{\"type\":\"string\",\"description\":\"One read-only Redis command as text, for what=command\"},"
        "\"limit\":{\"type\":\"integer\",\"description\":\"Max rows to return (default 50, max 500)\"}},\"required\":[\"connector\",\"what\"]}}}");
    return true;
}

void asst_conns_json(ds4_buf *b) {
    ds4_buf_puts(b, "{\"kinds\":[");
    for (int i = 0; i < DB_NKINDS; i++) { ds4_buf_printf(b, "%s{\"kind\":\"%s\",\"label\":", i ? "," : "", DB_KINDS[i].kind); ds4_json_escape(b, DB_KINDS[i].label); ds4_buf_printf(b, ",\"port\":%d,\"fields\":\"%s\"}", DB_KINDS[i].port, DB_KINDS[i].fields); }
    ds4_buf_puts(b, "],\"connectors\":[");
    char names[CONN_MAX][ASST_CONN_NAME_MAX + 1]; const int n = scan(names); int k = 0;
    for (int i = 0; i < n; i++) {
        asst_conn c; if (!asst_conn_load(names[i], &c)) continue;
        ds4_buf_printf(b, "%s{\"name\":\"%s\",\"kind\":\"%s\",\"host\":", k++ ? "," : "", c.name, c.kind); ds4_json_escape(b, c.host);
        ds4_buf_printf(b, ",\"port\":%d,\"db\":", c.port); ds4_json_escape(b, c.db); ds4_buf_puts(b, ",\"user\":"); ds4_json_escape(b, c.user); ds4_buf_puts(b, ",\"auth_db\":"); ds4_json_escape(b, c.auth_db); ds4_buf_puts(b, ",\"path\":"); ds4_json_escape(b, c.path);
        ds4_buf_printf(b, ",\"has_password\":%s,\"enabled\":%s,\"updated\":%ld}", has_password(c.name) ? "true" : "false", c.enabled ? "true" : "false", c.updated);
    }
    ds4_buf_puts(b, "]}");
}

/* 表单 → asst_conn(没给的字段留空); password 单独处理 */
static bool from_body(const char *body, asst_conn *c, char *err, size_t errn) {
    memset(c, 0, sizeof *c);
    jget(body, "name", c->name, sizeof c->name); jget(body, "kind", c->kind, sizeof c->kind); jget(body, "host", c->host, sizeof c->host); jget(body, "db", c->db, sizeof c->db); jget(body, "user", c->user, sizeof c->user); jget(body, "auth_db", c->auth_db, sizeof c->auth_db); jget(body, "path", c->path, sizeof c->path);
    long v = 0; asst_jint(body, "port", &v); c->port = (int)v;
    const db_kind *k = db_kind_find(c->kind);
    if (!k) { snprintf(err, errn, "kind must be one of the listed database kinds"); return false; }
    if (!c->port) c->port = k->port;
    if (strstr(k->fields, "host") && !c->host[0]) { snprintf(err, errn, "host is required"); return false; }
    if (strstr(k->fields, "path")) { if (!c->path[0]) { snprintf(err, errn, "path (the database file) is required"); return false; } char abs[ASST_PATH]; if (!asst_path_resolve(c->path, abs, sizeof abs)) { snprintf(err, errn, "cannot resolve path %s", c->path); return false; } snprintf(c->path, sizeof c->path, "%s", abs); }
    c->enabled = true;
    return true;
}

/* 测试连接: 拿表单当前值连一次; 带 name 且没给 password 时用已存的密码。成功 {ok:true, line}, 失败 {error} */
bool asst_conn_test(const char *body, ds4_buf *out) {
    asst_conn c; char err[400];
    if (!from_body(body, &c, err, sizeof err)) { ds4_buf_puts(out, "{\"error\":"); ds4_json_escape(out, err); ds4_buf_putc(out, '}'); return false; }
    char *pw = NULL; asst_jstr(body, "password", &pw);
    char saved[257] = "";
    if ((!pw || !pw[0]) && asst_conn_name_ok(c.name)) asst_conn_password(c.name, saved, sizeof saved);
    const db_target t = { c.kind, c.host, c.db, c.user, pw && pw[0] ? pw : saved, c.auth_db, c.path, c.port };
    char line[512]; const long t0 = now_ms();
    bool ok;
    if (!strcmp(c.kind, "sqlite")) ok = asst_sqlite_probe(c.path, line, sizeof line, err, sizeof err);   /* 文件库走沙盒(tool_db.c) */
    else ok = db_probe(&t, line, sizeof line, err, sizeof err);
    if (pw) { memset(pw, 0, strlen(pw)); free(pw); }
    memset(saved, 0, sizeof saved);
    if (ok) { ds4_buf_puts(out, "{\"ok\":true,\"line\":"); ds4_json_escape(out, line); ds4_buf_printf(out, ",\"ms\":%ld}", now_ms() - t0); }
    else { ds4_buf_puts(out, "{\"error\":"); ds4_json_escape(out, err); ds4_buf_putc(out, '}'); }
    return ok;
}

/* 页面: {op:"save", name, kind, host, port, db, user, auth_db, path, password?} / {op:"delete", name} / {op:"enable", name, on} */
bool asst_conn_op(const char *body, char *err, size_t errn) {
    char *op = NULL, *name = NULL; asst_jstr(body, "op", &op); asst_jstr(body, "name", &name);
    bool ok = false;
    if (!asst_conn_name_ok(name)) snprintf(err, errn, "connector name must be 1-%d lowercase letters, digits or hyphens", ASST_CONN_NAME_MAX);
    else if (op && !strcmp(op, "save")) {
        asst_conn c;
        if (from_body(body, &c, err, errn)) {
            asst_conn old; if (asst_conn_load(name, &old)) c.enabled = old.enabled;   /* 改字段不碰开关 */
            c.updated = (long)time(NULL);
            char *pw = NULL; asst_jstr(body, "password", &pw);
            char sn[48]; secret_name(name, sn, sizeof sn);
            ok = conn_save(&c);
            if (ok && pw && pw[0]) ok = asst_secret_set(sn, pw, err, errn);   /* 没给 = 不改已存的 */
            else if (!ok) snprintf(err, errn, "cannot write the connector file");
            if (pw) { memset(pw, 0, strlen(pw)); free(pw); }
        }
    } else if (op && !strcmp(op, "enable")) {
        asst_conn c; bool on = true;
        if (!asst_conn_load(name, &c)) snprintf(err, errn, "no such connector");
        else if (!asst_jbool(body, "on", &on)) snprintf(err, errn, "on must be true or false");
        else { c.enabled = on; ok = conn_save(&c); if (!ok) snprintf(err, errn, "cannot write the connector file"); }
    } else if (op && !strcmp(op, "delete")) {
        char t[ASST_PATH + 32]; snprintf(t, sizeof t, "%s/trash", asst_root); mkdir(t, 0700); snprintf(t, sizeof t, "%s/trash/connectors", asst_root); mkdir(t, 0700);
        char p[ASST_PATH + 64], to[ASST_PATH + 128]; conn_path(name, p, sizeof p); snprintf(to, sizeof to, "%s/trash/connectors/%s-%ld.json", asst_root, name, (long)time(NULL));
        ok = rename(p, to) == 0;
        if (!ok) snprintf(err, errn, "no such connector");
        else { char sn[48]; secret_name(name, sn, sizeof sn); char e2[64]; asst_secret_set(sn, "", e2, sizeof e2); }   /* 密码不进回收站 */
    } else snprintf(err, errn, "op must be save, enable or delete");
    free(op); free(name);
    return ok;
}
