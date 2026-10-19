/* mysql_wire.c — MySQL / MariaDB 客户端协议: 握手 v10 → HandshakeResponse41 → 鉴权(两种插件 + 插件切换 + sha2 全量鉴权)→ COM_QUERY 文本结果集(db.h)。
 * 只读两道锁: sql_guard 白名单 + SET SESSION TRANSACTION READ ONLY(InnoDB 拒绝一切写); 加 max_execution_time 10 s(MariaDB 叫 max_statement_time, 两个都试, 不认的忽略)。
 * 不设 CLIENT_DEPRECATE_EOF: 结果集按老样子以 EOF 包(0xFE, 长度 < 9)收尾, 两家都这么回, 解析最简单。不做 TLS, 不做压缩。 */
#include "db.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CAP_LONG_PASSWORD 1
#define CAP_LONG_FLAG 4
#define CAP_CONNECT_WITH_DB 8
#define CAP_PROTOCOL_41 0x200
#define CAP_TRANSACTIONS 0x2000
#define CAP_SECURE_CONNECTION 0x8000
#define CAP_MULTI_RESULTS 0x20000
#define CAP_PLUGIN_AUTH 0x80000
#define CAP_PLUGIN_AUTH_LENENC 0x200000

typedef struct { int fd; uint8_t seq; char err[400]; char version[64]; } my;

static bool send_pkt(my *c, const void *p, size_t n) {
    uint8_t h[4] = { (uint8_t)n, (uint8_t)(n >> 8), (uint8_t)(n >> 16), c->seq++ };
    if (db_io_all(c->fd, h, 4, true) && db_io_all(c->fd, (void *)p, n, true)) return true;
    snprintf(c->err, sizeof c->err, "send failed (%s)", strerror(errno)); return false;
}
static bool recv_pkt(my *c, uint8_t **p, size_t *n) {   /* 一个包(不拼 16 MB 分片: 我们限了行数, 单包够); 末尾补 0 方便当字符串 */
    uint8_t h[4];
    if (!db_io_all(c->fd, h, 4, false)) { snprintf(c->err, sizeof c->err, "no reply (%s)", errno == EAGAIN || errno == EWOULDBLOCK ? "timed out" : "connection closed"); return false; }
    *n = (size_t)h[0] | (size_t)h[1] << 8 | (size_t)h[2] << 16; c->seq = (uint8_t)(h[3] + 1);
    *p = malloc(*n + 1);
    if (!*p || !db_io_all(c->fd, *p, *n, false)) { free(*p); snprintf(c->err, sizeof c->err, "reply cut short"); return false; }
    (*p)[*n] = 0;
    return true;
}
static bool is_err(my *c, const uint8_t *p, size_t n) {   /* ERR 包: FF code(2) '#' sqlstate(5) msg */
    if (n < 3 || p[0] != 0xFF) return false;
    const int code = p[1] | p[2] << 8; const char *msg = (const char *)p + 3; if (n > 9 && *msg == '#') msg += 6;
    snprintf(c->err, sizeof c->err, "server: %s (code %d)", msg, code);
    return true;
}
static uint64_t lenenc(const uint8_t **p, const uint8_t *end, bool *null) {
    *null = false;
    if (*p >= end) return 0;
    const uint8_t b = *(*p)++;
    if (b < 0xFB) return b;
    if (b == 0xFB) { *null = true; return 0; }
    const int k = b == 0xFC ? 2 : b == 0xFD ? 3 : 8; uint64_t v = 0;
    if (*p + k > end) { *p = end; return 0; }
    for (int i = 0; i < k; i++) v |= (uint64_t)(*p)[i] << (8 * i);
    *p += k; return v;
}
static char *lenenc_str(const uint8_t **p, const uint8_t *end, bool *null) {
    const uint64_t n = lenenc(p, end, null);
    if (*null || *p + n > end) { if (!*null) *p = end; return NULL; }
    char *s = malloc((size_t)n + 1); memcpy(s, *p, (size_t)n); s[n] = 0; *p += n; return s;
}

/* 按插件名算口令应答 */
static size_t scramble(const char *plugin, const char *pw, const uint8_t nonce[20], uint8_t *out) {
    if (!pw || !pw[0]) return 0;
    if (!strcmp(plugin, "mysql_native_password")) { mysql_scramble_native(pw, nonce, out); return 20; }
    if (!strcmp(plugin, "caching_sha2_password")) { mysql_scramble_sha2(pw, nonce, out); return 32; }
    return (size_t)-1;
}

/* caching_sha2_password 快路不成: 服务器回 0x01 0x04 要全量鉴权 → 明文 TCP 上请公钥(0x02) → RSA-OAEP 加密 (pw‖0) XOR nonce 发回 */
static bool sha2_full(my *c, const char *pw, const uint8_t nonce[20]) {
    const uint8_t ask = 2; if (!send_pkt(c, &ask, 1)) return false;
    uint8_t *p; size_t n; if (!recv_pkt(c, &p, &n)) return false;
    if (is_err(c, p, n) || n < 2 || p[0] != 1) { free(p); if (!c->err[0]) snprintf(c->err, sizeof c->err, "server did not send its public key"); return false; }
    uint8_t msg[256]; const size_t mn = strlen(pw) + 1; if (mn > sizeof msg) { free(p); snprintf(c->err, sizeof c->err, "password too long"); return false; }
    memcpy(msg, pw, mn); for (size_t i = 0; i < mn; i++) msg[i] ^= nonce[i % 20];
    uint8_t enc[512]; size_t en;
    const bool ok = mysql_rsa_oaep((const char *)p + 1, msg, mn, enc, &en); free(p); memset(msg, 0, sizeof msg);
    if (!ok) { snprintf(c->err, sizeof c->err, "cannot encrypt the password with the server's RSA key"); return false; }
    return send_pkt(c, enc, en);
}

static bool my_open(my *c, const db_target *t) {
    c->err[0] = 0; c->seq = 0; c->version[0] = 0;
    if ((c->fd = db_tcp_connect(t->host, t->port > 0 ? t->port : 3306, c->err, sizeof c->err)) < 0) return false;
    uint8_t *p; size_t n;
    if (!recv_pkt(c, &p, &n)) { close(c->fd); return false; }
    if (is_err(c, p, n) || n < 40 || p[0] != 10) { free(p); if (!c->err[0]) snprintf(c->err, sizeof c->err, "not a MySQL handshake"); close(c->fd); return false; }
    const uint8_t *q = p + 1; snprintf(c->version, sizeof c->version, "%s", (const char *)q); q += strlen((const char *)q) + 1 + 4;
    uint8_t nonce[20] = {0}; memcpy(nonce, q, 8); q += 8 + 1;
    uint32_t caps = (uint32_t)(q[0] | q[1] << 8); q += 2 + 1 + 2; caps |= (uint32_t)(q[0] | q[1] << 8) << 16; q += 2;
    const int alen = *q; q += 1 + 10;
    const int part2 = alen > 8 ? alen - 8 : 13; memcpy(nonce + 8, q, 12); q += part2;
    char plugin[64] = "mysql_native_password"; if ((caps & CAP_PLUGIN_AUTH) && q < p + n) snprintf(plugin, sizeof plugin, "%s", (const char *)q);
    free(p);
    const char *pw = t->pw ? t->pw : "", *user = t->user ? t->user : "root";
    uint8_t ar[32]; size_t an = scramble(plugin, pw, nonce, ar);
    if (an == (size_t)-1) { an = 0; snprintf(plugin, sizeof plugin, "caching_sha2_password"); an = scramble(plugin, pw, nonce, ar); }   /* 服务器建议的插件不认识: 用 sha2 试, 服务器会切 */
    ds4_buf r = {0}; const uint32_t mycaps = CAP_LONG_PASSWORD | CAP_LONG_FLAG | CAP_PROTOCOL_41 | CAP_TRANSACTIONS | CAP_SECURE_CONNECTION | CAP_MULTI_RESULTS | CAP_PLUGIN_AUTH | CAP_PLUGIN_AUTH_LENENC | (t->db && t->db[0] ? CAP_CONNECT_WITH_DB : 0);
    uint8_t h[32] = {0}; for (int i = 0; i < 4; i++) h[i] = (uint8_t)(mycaps >> (8 * i)); h[4] = 0; h[5] = 0; h[6] = 0; h[7] = 1; h[8] = 45;   /* max packet 16 MB, charset utf8mb4 */
    ds4_buf_append(&r, h, 32); ds4_buf_append(&r, user, strlen(user) + 1); ds4_buf_putc(&r, (char)an); ds4_buf_append(&r, ar, an);
    if (t->db && t->db[0]) ds4_buf_append(&r, t->db, strlen(t->db) + 1);
    ds4_buf_append(&r, plugin, strlen(plugin) + 1);
    const bool sent = send_pkt(c, r.ptr, r.len); ds4_buf_free(&r);
    if (!sent) { close(c->fd); return false; }
    for (int round = 0; round < 4; round++) {
        if (!recv_pkt(c, &p, &n)) { close(c->fd); return false; }
        if (is_err(c, p, n)) { free(p); if (strstr(c->err, "code 1045")) snprintf(c->err, sizeof c->err, "authentication failed for user %s (code 1045: wrong password, or the user is not allowed from this host)", user); close(c->fd); return false; }
        if (n && p[0] == 0) { free(p); return true; }   /* OK */
        if (n && p[0] == 0xFE) {   /* AuthSwitchRequest: 插件名 + 新 nonce */
            snprintf(plugin, sizeof plugin, "%s", (const char *)p + 1); const uint8_t *d = p + 1 + strlen(plugin) + 1;
            if (d + 20 <= p + n) memcpy(nonce, d, 20);
            free(p);
            an = scramble(plugin, pw, nonce, ar);
            if (an == (size_t)-1) { snprintf(c->err, sizeof c->err, "server wants authentication plugin %s, which is not supported (use mysql_native_password or caching_sha2_password)", plugin); close(c->fd); return false; }
            if (!send_pkt(c, ar, an)) { close(c->fd); return false; }
            continue;
        }
        if (n >= 2 && p[0] == 1) {   /* caching_sha2: 03 快路成(下一包是 OK) / 04 要全量 */
            const uint8_t st = p[1]; free(p);
            if (st == 4 && !sha2_full(c, pw, nonce)) { close(c->fd); return false; }
            continue;
        }
        free(p); snprintf(c->err, sizeof c->err, "unexpected packet during authentication"); close(c->fd); return false;
    }
    snprintf(c->err, sizeof c->err, "authentication did not finish"); close(c->fd); return false;
}
static void my_close(my *c) { if (c->fd >= 0) { c->seq = 0; const uint8_t q = 1; send_pkt(c, &q, 1); close(c->fd); } c->fd = -1; }

static bool numeric_type(uint8_t t) { return t <= 5 || t == 8 || t == 9 || t == 13 || t == 0xF6; }

/* COM_QUERY; 结果行进 tab(NULL 丢弃); OK 包也算成功 */
static bool query(my *c, const char *sql, sql_table *tab, long max) {
    c->seq = 0; ds4_buf q = {0}; ds4_buf_putc(&q, 3); ds4_buf_puts(&q, sql);
    const bool sent = send_pkt(c, q.ptr, q.len); ds4_buf_free(&q);
    if (!sent) return false;
    uint8_t *p; size_t n;
    if (!recv_pkt(c, &p, &n)) return false;
    if (is_err(c, p, n)) { free(p); return false; }
    if (n && (p[0] == 0 || p[0] == 0xFE)) { free(p); return true; }
    const uint8_t *r = p; bool null; const int ncols = (int)lenenc(&r, p + n, &null); free(p);
    if (ncols <= 0 || ncols > 4096) { snprintf(c->err, sizeof c->err, "malformed result set"); return false; }
    uint8_t *types = calloc((size_t)ncols, 1);
    if (tab) sql_table_cols(tab, ncols);
    for (int i = 0; i < ncols; i++) {   /* 列定义: 6 个 lenenc 串, 然后 0x0c, charset(2) len(4) type(1) flags(2) dec(1) */
        if (!recv_pkt(c, &p, &n)) { free(types); return false; }
        r = p; char *name = NULL;
        for (int k = 0; k < 6; k++) { char *s = lenenc_str(&r, p + n, &null); if (k == 4) name = s; else free(s); }
        if (r + 8 <= p + n) types[i] = r[7];
        if (tab) sql_table_col(tab, i, name ? name : "?");
        free(name); free(p);
    }
    if (!recv_pkt(c, &p, &n)) { free(types); return false; }   /* EOF after columns */
    free(p);
    bool failed = false;
    for (;;) {
        if (!recv_pkt(c, &p, &n)) { free(types); return false; }
        if (n && p[0] == 0xFE && n < 9) { free(p); break; }
        if (is_err(c, p, n)) { free(p); failed = true; break; }
        if (tab && (max < 0 || tab->rows < max)) {
            r = p; sql_table_row_begin(tab);
            for (int i = 0; i < ncols; i++) { char *s = lenenc_str(&r, p + n, &null); sql_table_cell(tab, i, s, null, numeric_type(types[i])); free(s); }
            sql_table_row_end(tab);
        }
        free(p);
    }
    free(types);
    return !failed;
}
static void quote(ds4_buf *b, const char *s) { ds4_buf_putc(b, '\''); for (; *s; s++) { if (*s == '\'' || *s == '\\') ds4_buf_putc(b, '\\'); ds4_buf_putc(b, *s); } ds4_buf_putc(b, '\''); }

char *mysql_query(const db_target *t, const db_req *r, bool *ok) {
    *ok = false; my c; ds4_buf o = {0};
    const char *what = r->what ? r->what : "";
    if (!my_open(&c, t)) { ds4_buf_printf(&o, "error: %s\n", c.err); return ds4_buf_take(&o); }
    sql_table tab = {0}; bool done = false;
    if (!query(&c, "SET SESSION TRANSACTION READ ONLY", NULL, 0)) goto out;
    if (!query(&c, "SET SESSION max_execution_time = 10000", NULL, 0)) query(&c, "SET SESSION max_statement_time = 10", NULL, 0);   /* MySQL / MariaDB 各一个名字 */
    if (!strcmp(what, "tables")) {
        done = query(&c, "SELECT table_name AS `table`, table_type AS kind, table_rows AS approx_rows FROM information_schema.tables WHERE table_schema = DATABASE() ORDER BY 1", &tab, DB_LIMIT_MAX);
        if (done) ds4_buf_printf(&o, "%ld tables in %s (approx_rows is an estimate; use sql with count(*) for an exact number):\n", tab.rows, t->db ? t->db : "");
    } else if (!strcmp(what, "fields")) {
        if (!r->coll || !r->coll[0]) { snprintf(c.err, sizeof c.err, "fields needs collection (the table name)"); goto out; }
        ds4_buf q = {0}; ds4_buf_puts(&q, "SELECT column_name, column_type, is_nullable, column_key, column_default FROM information_schema.columns WHERE table_schema = DATABASE() AND table_name = "); quote(&q, r->coll); ds4_buf_puts(&q, " ORDER BY ordinal_position");
        done = query(&c, q.ptr, &tab, DB_LIMIT_MAX); ds4_buf_free(&q);
        if (done && !tab.rows) { snprintf(c.err, sizeof c.err, "no table %s (call what=tables to see the names)", r->coll); done = false; goto out; }
        if (done) ds4_buf_printf(&o, "%ld columns of %s:\n", tab.rows, r->coll);
    } else if (!strcmp(what, "sql")) {
        const char *why = sql_readonly_check(r->sql);
        if (why) { snprintf(c.err, sizeof c.err, "%s", why); goto out; }
        char *w = sql_wrap_limit(r->sql, r->limit, "mysql");
        done = query(&c, w, &tab, r->limit); free(w);
        if (done) ds4_buf_printf(&o, "%ld rows (limit %ld)%s:\n", tab.rows, r->limit, tab.rows == r->limit ? "; there may be more" : "");
    } else snprintf(c.err, sizeof c.err, "what must be tables, fields or sql");
out:
    if (done) { ds4_buf_append(&o, tab.out.ptr, tab.out.len); *ok = true; }
    else { ds4_buf_free(&o); ds4_buf_printf(&o, "error: %s\n", c.err); }
    sql_table_free(&tab); my_close(&c);
    return ds4_buf_take(&o);
}

bool mysql_probe(const db_target *t, char *line, size_t n, char *err, size_t errn) {
    my c; if (!my_open(&c, t)) { snprintf(err, errn, "%s", c.err); return false; }
    sql_table k = {0};
    const bool ok = query(&c, "SELECT count(*) FROM information_schema.tables WHERE table_schema = DATABASE()", &k, 1);
    if (ok) {
        char cnt[32] = "?"; const char *p = k.out.ptr ? strchr(k.out.ptr, '\n') : NULL;
        if (p) { const char *q = strchr(p, ':'); if (q) { snprintf(cnt, sizeof cnt, "%s", q + 1); char *e = strchr(cnt, '}'); if (e) *e = 0; } }
        snprintf(line, n, "%s %s · %s · %s tables", strstr(c.version, "MariaDB") ? "MariaDB" : "MySQL", c.version, t->db ? t->db : "(no database)", cnt);
    } else snprintf(err, errn, "%s", c.err);
    sql_table_free(&k); my_close(&c);
    return ok;
}
