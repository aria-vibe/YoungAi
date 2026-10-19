/* redis.c — Redis RESP2 客户端 + 只读命令(db.h)。
 * 只读 = 命令白名单(READONLY): keys / get / info 由我们拼命令, command 让模型写一条文本命令, 第一个词不在表里就拒。
 * 回复 → JSON: 字符串原样(是数字也加引号, Redis 的值都是字节串), 整数裸写, 数组递归, nil → null, 错误 → "error: …"。
 * 鉴权: 有 user 走 AUTH user pw(ACL, Redis 6+), 否则 AUTH pw; db 是编号(SELECT n)。不做 TLS。 */
#include "db.h"
#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#define SCAN_COUNT 1000

typedef struct { int fd; char err[400]; } rd;

static const char *READONLY[] = { "GET", "MGET", "GETRANGE", "STRLEN", "EXISTS", "TYPE", "TTL", "PTTL", "KEYS", "SCAN", "DBSIZE", "INFO", "PING", "TIME", "RANDOMKEY", "OBJECT", "MEMORY",
    "HGET", "HMGET", "HGETALL", "HKEYS", "HVALS", "HLEN", "HEXISTS", "HSCAN", "LRANGE", "LLEN", "LINDEX", "LPOS", "SMEMBERS", "SISMEMBER", "SCARD", "SSCAN", "SRANDMEMBER", "SINTER", "SUNION", "SDIFF",
    "ZRANGE", "ZREVRANGE", "ZRANGEBYSCORE", "ZREVRANGEBYSCORE", "ZRANGEBYLEX", "ZSCORE", "ZCARD", "ZCOUNT", "ZRANK", "ZREVRANK", "ZSCAN", "ZMSCORE", "XRANGE", "XREVRANGE", "XLEN", "XINFO",
    "PFCOUNT", "GEOPOS", "GEODIST", "GEOSEARCH", "BITCOUNT", "GETBIT", "CONFIG", "CLIENT", "LASTSAVE", "ECHO", "JSON.GET", "JSON.TYPE", "FT.SEARCH", "FT.INFO", "TS.RANGE", "TS.GET" };

static bool send_cmd(rd *c, int argc, const char **argv) {
    ds4_buf b = {0}; ds4_buf_printf(&b, "*%d\r\n", argc);
    for (int i = 0; i < argc; i++) ds4_buf_printf(&b, "$%zu\r\n%s\r\n", strlen(argv[i]), argv[i]);
    const bool ok = db_io_all(c->fd, b.ptr, b.len, true); ds4_buf_free(&b);
    if (!ok) snprintf(c->err, sizeof c->err, "send failed (%s)", strerror(errno));
    return ok;
}
static bool read_line(rd *c, char *out, size_t n) {   /* 到 \r\n */
    size_t k = 0; char ch;
    for (;;) {
        if (!db_io_all(c->fd, &ch, 1, false)) { snprintf(c->err, sizeof c->err, "no reply (%s)", errno == EAGAIN || errno == EWOULDBLOCK ? "timed out" : "connection closed"); return false; }
        if (ch == '\n') { if (k && out[k - 1] == '\r') k--; out[k] = 0; return true; }
        if (k + 1 < n) out[k++] = ch;
    }
}
/* 一条回复 → JSON 追加到 out; 错误回复 → 返回 false 且 c->err; max_items 限数组条数(其余丢弃但要读完) */
static bool reply(rd *c, ds4_buf *out, int depth, long max_items) {
    char line[4096]; if (!read_line(c, line, sizeof line)) return false;
    if (depth > 16) { snprintf(c->err, sizeof c->err, "reply nested too deep"); return false; }
    switch (line[0]) {
    case '+': ds4_json_escape(out, line + 1); return true;
    case ':': ds4_buf_puts(out, line + 1); return true;
    case '-': snprintf(c->err, sizeof c->err, "server: %s", line + 1); return false;
    case '$': {
        const long n = strtol(line + 1, NULL, 10);
        if (n < 0) { ds4_buf_puts(out, "null"); return true; }
        char *s = malloc((size_t)n + 3);
        if (!s || !db_io_all(c->fd, s, (size_t)n + 2, false)) { free(s); snprintf(c->err, sizeof c->err, "reply cut short"); return false; }
        s[n] = 0; ds4_json_escape(out, s); free(s); return true;
    }
    case '*': {
        const long n = strtol(line + 1, NULL, 10);
        if (n < 0) { ds4_buf_puts(out, "null"); return true; }
        ds4_buf_putc(out, '[');
        for (long i = 0; i < n; i++) {
            if (i < max_items) { if (i) ds4_buf_putc(out, ','); if (!reply(c, out, depth + 1, max_items)) return false; }
            else { ds4_buf sink = {0}; const bool ok = reply(c, &sink, depth + 1, max_items); ds4_buf_free(&sink); if (!ok) return false; }
        }
        if (n > max_items) ds4_buf_printf(out, ",\"…%ld more not shown\"", n - max_items);
        ds4_buf_putc(out, ']'); return true;
    }
    default: snprintf(c->err, sizeof c->err, "malformed reply"); return false;
    }
}
static bool call(rd *c, ds4_buf *out, long max_items, int argc, ...) {   /* 变参方便拼命令 */
    const char *argv[16]; va_list ap; va_start(ap, argc); for (int i = 0; i < argc && i < 16; i++) argv[i] = va_arg(ap, const char *); va_end(ap);
    if (!send_cmd(c, argc, argv)) return false;
    ds4_buf sink = {0}; const bool ok = reply(c, out ? out : &sink, 0, max_items); ds4_buf_free(&sink);
    return ok;
}
/* 回复里的裸字符串(去引号); 给 TYPE / SCAN 游标用 */
static void plain(const ds4_buf *b, char *out, size_t n) { const char *p = b->ptr ? b->ptr : ""; if (*p == '"') { snprintf(out, n, "%s", p + 1); char *e = strrchr(out, '"'); if (e) *e = 0; } else snprintf(out, n, "%s", p); }

static bool rd_open(rd *c, const db_target *t) {
    c->err[0] = 0;
    if ((c->fd = db_tcp_connect(t->host, t->port > 0 ? t->port : 6379, c->err, sizeof c->err)) < 0) return false;
    if (t->pw && t->pw[0]) {
        const bool ok = t->user && t->user[0] ? call(c, NULL, 0, 3, "AUTH", t->user, t->pw) : call(c, NULL, 0, 2, "AUTH", t->pw);
        if (!ok) { if (strstr(c->err, "WRONGPASS") || strstr(c->err, "invalid password")) snprintf(c->err, sizeof c->err, "authentication failed%s%s (wrong password or user)", t->user && t->user[0] ? " for user " : "", t->user && t->user[0] ? t->user : ""); close(c->fd); return false; }
    }
    if (t->db && t->db[0] && strcmp(t->db, "0")) { if (strspn(t->db, "0123456789") != strlen(t->db) || !call(c, NULL, 0, 2, "SELECT", t->db)) { if (!c->err[0]) snprintf(c->err, sizeof c->err, "db must be a database number (0-15)"); close(c->fd); return false; } }
    if (!call(c, NULL, 0, 1, "PING")) { close(c->fd); return false; }
    return true;
}
static void rd_close(rd *c) { if (c->fd >= 0) close(c->fd); c->fd = -1; }

/* 文本命令切词: 空白分隔, 单双引号包空格 */
static int split(const char *s, char **argv, int max) {
    int n = 0; const char *p = s;
    while (*p && n < max) {
        while (isspace((unsigned char)*p)) p++;
        if (!*p) break;
        ds4_buf w = {0}; char q = 0;
        while (*p && (q || !isspace((unsigned char)*p))) { if (!q && (*p == '"' || *p == '\'')) q = *p; else if (q && *p == q) q = 0; else ds4_buf_putc(&w, *p); p++; }
        argv[n++] = ds4_buf_take(&w);
    }
    return n;
}
static bool allowed(const char *cmd) { for (size_t i = 0; i < sizeof READONLY / sizeof READONLY[0]; i++) if (!strcasecmp(cmd, READONLY[i])) return true; return false; }

static bool do_get(rd *c, ds4_buf *o, const char *key, long limit) {
    ds4_buf ty = {0}; if (!call(c, &ty, 0, 2, "TYPE", key)) return false;
    char t[32]; plain(&ty, t, sizeof t); ds4_buf_free(&ty);
    char lim[32]; snprintf(lim, sizeof lim, "%ld", limit);
    char last[32]; snprintf(last, sizeof last, "%ld", limit - 1);
    ds4_buf_printf(o, "%s · type %s", key, t);
    ds4_buf v = {0}; bool ok;
    if (!strcmp(t, "none")) { ds4_buf_puts(o, " (no such key)\n"); return true; }
    if (!strcmp(t, "string")) ok = call(c, &v, limit, 2, "GET", key);
    else if (!strcmp(t, "hash")) ok = call(c, &v, limit * 2, 2, "HGETALL", key);
    else if (!strcmp(t, "list")) ok = call(c, &v, limit, 4, "LRANGE", key, "0", last);
    else if (!strcmp(t, "set")) ok = call(c, &v, limit, 2, "SMEMBERS", key);
    else if (!strcmp(t, "zset")) ok = call(c, &v, limit * 2, 5, "ZRANGE", key, "0", last, "WITHSCORES");
    else if (!strcmp(t, "stream")) ok = call(c, &v, limit, 6, "XRANGE", key, "-", "+", "COUNT", lim);
    else { ds4_buf_printf(o, " (use what=command to read this type)\n"); return true; }
    if (ok) { ds4_buf_printf(o, "%s:\n", !strcmp(t, "hash") ? " (field, value alternating)" : !strcmp(t, "zset") ? " (member, score alternating)" : ""); ds4_buf_append(o, v.ptr, v.len); ds4_buf_putc(o, '\n'); }
    ds4_buf_free(&v);
    return ok;
}

char *redis_query(const db_target *t, const db_req *r, bool *ok) {
    *ok = false; rd c; ds4_buf o = {0};
    const char *what = r->what ? r->what : "";
    if (!rd_open(&c, t)) { ds4_buf_printf(&o, "error: %s\n", c.err); return ds4_buf_take(&o); }
    bool done = false;
    if (!strcmp(what, "keys")) {
        char cursor[32] = "0", cnt[16]; snprintf(cnt, sizeof cnt, "%d", SCAN_COUNT); long n = 0; ds4_buf keys = {0};
        const char *pat = r->pattern && r->pattern[0] ? r->pattern : "*";
        do {
            ds4_buf v = {0};
            if (!call(&c, &v, SCAN_COUNT, 6, "SCAN", cursor, "MATCH", pat, "COUNT", cnt)) { ds4_buf_free(&v); goto out; }
            /* 回复 ["游标",[key…]]: 抠游标, 逐个 key 查 TYPE */
            const char *p = v.ptr + 2; snprintf(cursor, sizeof cursor, "%.*s", (int)strcspn(p, "\""), p);
            const char *arr = strchr(p, '['); if (arr) for (const char *k = arr + 1; *k == '"' && n < r->limit; ) {
                char *key = NULL; if (!json_string(&k, &key)) break;
                ds4_buf ty = {0}; char tn[32] = "?"; if (call(&c, &ty, 0, 2, "TYPE", key)) plain(&ty, tn, sizeof tn); ds4_buf_free(&ty);
                ds4_buf_printf(&keys, "%s · %s\n", key, tn); free(key); n++;
                if (*k == ',') k++;
            }
            ds4_buf_free(&v);
        } while (strcmp(cursor, "0") && n < r->limit);
        ds4_buf_printf(&o, "%ld keys matching %s (limit %ld)%s:\n", n, pat, r->limit, n >= r->limit ? "; there may be more" : ""); ds4_buf_append(&o, keys.ptr, keys.len); ds4_buf_free(&keys);
        done = true;
    } else if (!strcmp(what, "get")) {
        if (!r->key || !r->key[0]) { snprintf(c.err, sizeof c.err, "get needs key"); goto out; }
        done = do_get(&c, &o, r->key, r->limit);
    } else if (!strcmp(what, "info")) {
        ds4_buf v = {0}; done = call(&c, &v, 0, 2, "INFO", "default");
        if (done) { char *s = NULL; const char *p = v.ptr; if (json_string(&p, &s)) { ds4_buf_puts(&o, s); free(s); } }
        ds4_buf_free(&v);
    } else if (!strcmp(what, "command")) {
        char *argv[64]; const int argc = split(r->command ? r->command : "", argv, 64);
        if (!argc) { snprintf(c.err, sizeof c.err, "command is empty"); goto out; }
        if (!allowed(argv[0])) { snprintf(c.err, sizeof c.err, "%s is not allowed: this tool is read-only (GET/HGETALL/LRANGE/ZRANGE/SCAN/INFO and other reading commands only)", argv[0]); for (int i = 0; i < argc; i++) free(argv[i]); goto out; }
        if (!strcasecmp(argv[0], "CONFIG") && (argc < 2 || strcasecmp(argv[1], "GET"))) { snprintf(c.err, sizeof c.err, "only CONFIG GET is allowed"); for (int i = 0; i < argc; i++) free(argv[i]); goto out; }
        if (!strcasecmp(argv[0], "CLIENT") && (argc < 2 || (strcasecmp(argv[1], "LIST") && strcasecmp(argv[1], "INFO")))) { snprintf(c.err, sizeof c.err, "only CLIENT LIST / INFO are allowed"); for (int i = 0; i < argc; i++) free(argv[i]); goto out; }
        if (send_cmd(&c, argc, (const char **)argv)) { ds4_buf v = {0}; done = reply(&c, &v, 0, r->limit * 2); if (done) { ds4_buf_printf(&o, "%s →\n", argv[0]); ds4_buf_append(&o, v.ptr, v.len); ds4_buf_putc(&o, '\n'); } ds4_buf_free(&v); }
        for (int i = 0; i < argc; i++) free(argv[i]);
    } else snprintf(c.err, sizeof c.err, "what must be keys, get, info or command");
out:
    if (!done) { ds4_buf_free(&o); ds4_buf_printf(&o, "error: %s\n", c.err); } else *ok = true;
    rd_close(&c);
    return ds4_buf_take(&o);
}

bool redis_probe(const db_target *t, char *line, size_t n, char *err, size_t errn) {
    rd c; if (!rd_open(&c, t)) { snprintf(err, errn, "%s", c.err); return false; }
    ds4_buf v = {0}, k = {0};
    const bool ok = call(&c, &v, 0, 2, "INFO", "server") && call(&c, &k, 0, 1, "DBSIZE");
    if (ok) {
        char ver[32] = "?"; const char *p = strstr(v.ptr ? v.ptr : "", "redis_version:"); if (p) { snprintf(ver, sizeof ver, "%.*s", (int)strcspn(p + 14, "\\\r\n\""), p + 14); }
        snprintf(line, n, "Redis %s · db %s · %s keys", ver, t->db && t->db[0] ? t->db : "0", k.ptr ? k.ptr : "?");
    } else snprintf(err, errn, "%s", c.err);
    ds4_buf_free(&v); ds4_buf_free(&k); rd_close(&c);
    return ok;
}
