/* pg_wire.c — PostgreSQL 前后端协议 v3(简单查询那一半)+ SCRAM-SHA-256 / 明文口令鉴权(db.h)。
 * 只读两道锁: sql_guard 白名单 + 会话 SET default_transaction_read_only = on(服务器拒绝一切写, 连函数里的也拒); 外加 statement_timeout 10 s。
 * SCRAM 和 MongoDB 共用 mongo_scram.c: PostgreSQL 的 client-first 用户名留空(服务器按 StartupMessage 里的 user 算), 其余一字不差。
 * 不做 md5 鉴权(PostgreSQL 14 起默认 scram; 要用 md5 的老库报清楚让用户改 pg_hba), 不做 TLS。 */
#include "db.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct { int fd; char err[400]; } pg;

static void be32(uint8_t *p, uint32_t v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }
static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }
static uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] << 8 | p[1]); }

/* 发一条带类型字节的消息: type + int32(len 含自己) + body */
static bool send_msg(pg *c, char type, const void *body, size_t n) {
    uint8_t h[5]; h[0] = (uint8_t)type; be32(h + 1, (uint32_t)(n + 4));
    if (db_io_all(c->fd, h, 5, true) && db_io_all(c->fd, (void *)body, n, true)) return true;
    snprintf(c->err, sizeof c->err, "send failed (%s)", strerror(errno)); return false;
}
/* 收一条: 返回类型字节, body malloc(调用方 free); 0 = 失败 */
static char recv_msg(pg *c, uint8_t **body, size_t *n) {
    uint8_t h[5];
    if (!db_io_all(c->fd, h, 5, false)) { snprintf(c->err, sizeof c->err, "no reply (%s)", errno == EAGAIN || errno == EWOULDBLOCK ? "timed out" : "connection closed"); return 0; }
    const uint32_t len = rd32(h + 1);
    if (len < 4 || len > (64u << 20)) { snprintf(c->err, sizeof c->err, "malformed message"); return 0; }
    *n = len - 4; *body = malloc(*n + 1);
    if (!*body || !db_io_all(c->fd, *body, *n, false)) { free(*body); snprintf(c->err, sizeof c->err, "reply cut short"); return 0; }
    (*body)[*n] = 0;
    return (char)h[0];
}
static void err_msg(pg *c, const uint8_t *b, size_t n) {   /* ErrorResponse: 'S' severity 'C' code 'M' message … */
    const char *msg = NULL, *code = NULL;
    for (size_t i = 0; i < n && b[i]; ) { const char *v = (const char *)b + i + 1; if (b[i] == 'M') msg = v; else if (b[i] == 'C') code = v; i += 1 + strlen(v) + 1; }
    snprintf(c->err, sizeof c->err, "server: %s%s%s%s", msg ? msg : "error", code ? " (" : "", code ? code : "", code ? ")" : "");
}

static bool auth(pg *c, const db_target *t) {
    char cnonce[48]; uint8_t ssig[32]; bool scram = false;
    for (;;) {
        uint8_t *b; size_t n; const char ty = recv_msg(c, &b, &n);
        if (!ty) return false;
        if (ty == 'E') { err_msg(c, b, n); free(b); if (strstr(c->err, "28P01") || strstr(c->err, "28000")) snprintf(c->err, sizeof c->err, "authentication failed for user %s (wrong password, or pg_hba.conf does not allow this host)", t->user); return false; }
        if (ty == 'Z') { free(b); return true; }   /* ReadyForQuery */
        if (ty != 'R') { free(b); continue; }     /* S K N: 参数/后端 key/提示 */
        const uint32_t kind = n >= 4 ? rd32(b) : 99;
        if (kind == 0) { free(b); continue; }
        if (kind == 3) { if (!send_msg(c, 'p', t->pw ? t->pw : "", strlen(t->pw ? t->pw : "") + 1)) { free(b); return false; } free(b); continue; }   /* 明文(pg_hba trust/password) */
        if (kind == 5) { free(b); snprintf(c->err, sizeof c->err, "server wants md5 authentication; set password_encryption = scram-sha-256 and the pg_hba.conf method to scram-sha-256"); return false; }
        if (kind == 10) {   /* SASL: 机制表 */
            bool has = false; for (size_t i = 4; i < n && b[i]; i += strlen((char *)b + i) + 1) if (!strcmp((char *)b + i, "SCRAM-SHA-256")) has = true;
            free(b);
            if (!has) { snprintf(c->err, sizeof c->err, "server offers no SCRAM-SHA-256"); return false; }
            uint8_t rnd[SCRAM_NONCE_LEN]; FILE *f = fopen("/dev/urandom", "rb"); const bool got = f && fread(rnd, 1, sizeof rnd, f) == sizeof rnd; if (f) fclose(f);
            if (!got) { snprintf(c->err, sizeof c->err, "no random source"); return false; }
            scram_b64(rnd, sizeof rnd, cnonce);
            char first[128]; scram_client_first("", cnonce, first, sizeof first);
            ds4_buf m = {0}; ds4_buf_append(&m, "SCRAM-SHA-256", 14); uint8_t l[4]; be32(l, (uint32_t)strlen(first)); ds4_buf_append(&m, l, 4); ds4_buf_puts(&m, first);
            const bool ok = send_msg(c, 'p', m.ptr, m.len); ds4_buf_free(&m);
            if (!ok) return false;
            scram = true; continue;
        }
        if (kind == 11 && scram) {   /* server-first */
            char final_[512];
            const bool ok = scram_client_final("", t->pw ? t->pw : "", cnonce, (const char *)b + 4, final_, sizeof final_, ssig, c->err, sizeof c->err);
            free(b);
            if (!ok || !send_msg(c, 'p', final_, strlen(final_))) return false;
            continue;
        }
        if (kind == 12 && scram) { const bool ok = scram_server_ok(ssig, (const char *)b + 4); free(b); if (!ok) { snprintf(c->err, sizeof c->err, "server signature mismatch"); return false; } continue; }
        free(b); snprintf(c->err, sizeof c->err, "unsupported authentication method %u (only scram-sha-256 and password)", kind); return false;
    }
}

static bool pg_open(pg *c, const db_target *t) {
    c->err[0] = 0;
    if ((c->fd = db_tcp_connect(t->host, t->port > 0 ? t->port : 5432, c->err, sizeof c->err)) < 0) return false;
    ds4_buf s = {0}; uint8_t h[8]; be32(h, 0); be32(h + 4, 196608); ds4_buf_append(&s, h, 8);
    ds4_buf_append(&s, "user", 5); ds4_buf_append(&s, t->user ? t->user : "postgres", strlen(t->user ? t->user : "postgres") + 1);
    ds4_buf_append(&s, "database", 9); ds4_buf_append(&s, t->db ? t->db : "postgres", strlen(t->db ? t->db : "postgres") + 1);
    ds4_buf_append(&s, "client_encoding", 16); ds4_buf_append(&s, "UTF8", 5);
    ds4_buf_append(&s, "application_name", 17); ds4_buf_append(&s, "ds4-assistant", 14);
    ds4_buf_putc(&s, 0);
    be32((uint8_t *)s.ptr, (uint32_t)s.len);
    const bool sent = db_io_all(c->fd, s.ptr, s.len, true); ds4_buf_free(&s);
    if (!sent) { snprintf(c->err, sizeof c->err, "send failed"); close(c->fd); return false; }
    if (!auth(c, t)) { close(c->fd); c->fd = -1; return false; }
    return true;
}
static void pg_close(pg *c) { if (c->fd >= 0) { send_msg(c, 'X', "", 0); close(c->fd); } c->fd = -1; }

static bool numeric_oid(uint32_t o) { return o == 20 || o == 21 || o == 23 || o == 26 || o == 700 || o == 701 || o == 1700; }

/* 简单查询: 行进 sql_table(NULL = 丢弃); 一直收到 ReadyForQuery */
static bool query(pg *c, const char *sql, sql_table *tab, long max) {
    if (!send_msg(c, 'Q', sql, strlen(sql) + 1)) return false;
    uint32_t *oids = NULL; int ncols = 0; bool failed = false;
    for (;;) {
        uint8_t *b; size_t n; const char ty = recv_msg(c, &b, &n);
        if (!ty) { free(oids); return false; }
        if (ty == 'Z') { free(b); break; }
        if (ty == 'E') { err_msg(c, b, n); failed = true; }
        else if (ty == 'T' && tab) {
            ncols = rd16(b); free(oids); oids = calloc((size_t)ncols + 1, sizeof *oids);
            sql_table_cols(tab, ncols);
            size_t p = 2;
            for (int i = 0; i < ncols && p < n; i++) { const char *name = (const char *)b + p; p += strlen(name) + 1; if (p + 18 > n) break; oids[i] = rd32(b + p + 6); sql_table_col(tab, i, name); p += 18; }
        } else if (ty == 'D' && tab && (max < 0 || tab->rows < max)) {
            const int k = rd16(b); size_t p = 2;
            sql_table_row_begin(tab);
            for (int i = 0; i < k && i < ncols && p + 4 <= n; i++) {
                const int32_t len = (int32_t)rd32(b + p); p += 4;
                if (len < 0) { sql_table_cell(tab, i, NULL, true, false); continue; }
                if (p + (size_t)len > n) break;
                char *cell = malloc((size_t)len + 1); memcpy(cell, b + p, (size_t)len); cell[len] = 0; p += (size_t)len;
                if (oids[i] == 16) sql_table_cell(tab, i, cell[0] == 't' ? "true" : "false", false, true);
                else sql_table_cell(tab, i, cell, false, numeric_oid(oids[i]));
                free(cell);
            }
            sql_table_row_end(tab);
        }
        free(b);
    }
    free(oids);
    return !failed;
}
static void quote(ds4_buf *b, const char *s) { ds4_buf_putc(b, '\''); for (; *s; s++) { if (*s == '\'') ds4_buf_putc(b, '\''); ds4_buf_putc(b, *s); } ds4_buf_putc(b, '\''); }

char *pg_query(const db_target *t, const db_req *r, bool *ok) {
    *ok = false; pg c; ds4_buf o = {0};
    const char *what = r->what ? r->what : "";
    if (!pg_open(&c, t)) { ds4_buf_printf(&o, "error: %s\n", c.err); return ds4_buf_take(&o); }
    sql_table tab = {0}; bool done = false;
    if (!query(&c, "SET default_transaction_read_only = on; SET statement_timeout = 10000", NULL, 0)) goto out;
    if (!strcmp(what, "tables")) {
        done = query(&c, "SELECT n.nspname || '.' || c.relname AS \"table\", CASE c.relkind WHEN 'v' THEN 'view' WHEN 'm' THEN 'materialized view' ELSE 'table' END AS kind, GREATEST(c.reltuples, 0)::bigint AS approx_rows "
                        "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relkind IN ('r','v','m','p','f') AND n.nspname NOT IN ('pg_catalog','information_schema') ORDER BY 1", &tab, DB_LIMIT_MAX);
        if (done) ds4_buf_printf(&o, "%ld tables in %s (approx_rows is the planner's estimate; use sql with count(*) for an exact number):\n", tab.rows, t->db);
    } else if (!strcmp(what, "fields")) {
        if (!r->coll || !r->coll[0]) { snprintf(c.err, sizeof c.err, "fields needs collection (the table name, schema.table for tables outside public)"); goto out; }
        const char *dot = strchr(r->coll, '.'); char schema[128] = "public", name[256];
        if (dot) { snprintf(schema, sizeof schema, "%.*s", (int)(dot - r->coll < 127 ? dot - r->coll : 127), r->coll); snprintf(name, sizeof name, "%s", dot + 1); } else snprintf(name, sizeof name, "%s", r->coll);
        ds4_buf q = {0}; ds4_buf_puts(&q, "SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns WHERE table_schema = "); quote(&q, schema); ds4_buf_puts(&q, " AND table_name = "); quote(&q, name); ds4_buf_puts(&q, " ORDER BY ordinal_position");
        done = query(&c, q.ptr, &tab, DB_LIMIT_MAX); ds4_buf_free(&q);
        if (done && !tab.rows) { snprintf(c.err, sizeof c.err, "no table %s.%s (call what=tables to see the names)", schema, name); done = false; goto out; }
        if (done) ds4_buf_printf(&o, "%ld columns of %s.%s:\n", tab.rows, schema, name);
    } else if (!strcmp(what, "sql")) {
        const char *why = sql_readonly_check(r->sql);
        if (why) { snprintf(c.err, sizeof c.err, "%s", why); goto out; }
        char *w = sql_wrap_limit(r->sql, r->limit, "postgresql");
        done = query(&c, w, &tab, r->limit); free(w);
        if (done) ds4_buf_printf(&o, "%ld rows (limit %ld)%s:\n", tab.rows, r->limit, tab.rows == r->limit ? "; there may be more" : "");
    } else snprintf(c.err, sizeof c.err, "what must be tables, fields or sql");
out:
    if (done) { ds4_buf_append(&o, tab.out.ptr, tab.out.len); *ok = true; }
    else { ds4_buf_free(&o); ds4_buf_printf(&o, "error: %s\n", c.err); }
    sql_table_free(&tab); pg_close(&c);
    return ds4_buf_take(&o);
}

bool pg_probe(const db_target *t, char *line, size_t n, char *err, size_t errn) {
    pg c; if (!pg_open(&c, t)) { snprintf(err, errn, "%s", c.err); return false; }
    sql_table v = {0}, k = {0};
    const bool ok = query(&c, "SELECT split_part(version(), ' ', 2)", &v, 1) && query(&c, "SELECT count(*) FROM pg_class c JOIN pg_namespace nn ON nn.oid = c.relnamespace WHERE c.relkind IN ('r','v','m','p') AND nn.nspname NOT IN ('pg_catalog','information_schema')", &k, 1);
    if (ok) {   /* 两张单格表: 从 JSON 行里把值抠出来 */
        char ver[64] = "?", cnt[32] = "?";
        const char *p = v.out.ptr ? strchr(v.out.ptr, '\n') : NULL; if (p) { const char *q = strchr(p, ':'); if (q) { snprintf(ver, sizeof ver, "%s", q[1] == '"' ? q + 2 : q + 1); char *e = strpbrk(ver, "\"}"); if (e) *e = 0; } }
        p = k.out.ptr ? strchr(k.out.ptr, '\n') : NULL; if (p) { const char *q = strchr(p, ':'); if (q) { snprintf(cnt, sizeof cnt, "%s", q + 1); char *e = strchr(cnt, '}'); if (e) *e = 0; } }
        snprintf(line, n, "PostgreSQL %s · %s · %s tables", ver, t->db, cnt);
    } else snprintf(err, errn, "%s", c.err);
    sql_table_free(&v); sql_table_free(&k); pg_close(&c);
    return ok;
}
