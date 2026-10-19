/* sql_guard.c — 三家 SQL(PostgreSQL / MySQL / SQLite)共用: 只读语句白名单 + 包 LIMIT + 结果表转文本(db.h)。
 * 白名单是第一道; 第二道在各协议文件里(PostgreSQL/MySQL 开只读事务, SQLite 用 -readonly 开文件), 服务器自己再拒一次写。
 * 为什么包 SELECT * FROM (<sql>) AS q LIMIT n 而不是检查模型有没有写 LIMIT: 服务器端限, 一条 SELECT * FROM news_items 不会先把 2.5 万行发过来再由我们丢。
 * SHOW / EXPLAIN / DESCRIBE 不能当子查询, 原样发(它们的输出本来就小)。 */
#include "db.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 去掉开头的空白与注释(-- 到行尾, 以及块注释), 返回第一个词的位置 */
static const char *skip_lead(const char *s) {
    for (;;) {
        while (*s && isspace((unsigned char)*s)) s++;
        if (s[0] == '-' && s[1] == '-') { s = strchr(s, '\n'); if (!s) return ""; continue; }
        if (s[0] == '/' && s[1] == '*') { const char *e = strstr(s + 2, "*/"); if (!e) return ""; s = e + 2; continue; }
        if (*s == '(') { s++; continue; }   /* (SELECT …) UNION … 这种 */
        return s;
    }
}
static bool word_is(const char *s, const char *w) {
    const size_t n = strlen(w);
    for (size_t i = 0; i < n; i++) if (!s[i] || toupper((unsigned char)s[i]) != w[i]) return false;
    return !s[n] || (!isalnum((unsigned char)s[n]) && s[n] != '_');
}

const char *sql_readonly_check(const char *sql) {
    if (!sql || !sql[0]) return "sql is empty";
    if (strlen(sql) > DB_SQL_MAX) return "sql is too long";
    const char *s = skip_lead(sql);
    if (!*s) return "sql is empty";
    static const char *OKW[] = { "SELECT", "WITH", "SHOW", "EXPLAIN", "DESCRIBE", "DESC", "VALUES", "TABLE" };   /* 不放 PRAGMA: writable_schema 之类能改状态; 表结构走 SELECT … FROM pragma_table_info() */
    bool ok = false;
    for (size_t i = 0; i < sizeof OKW / sizeof OKW[0] && !ok; i++) ok = word_is(s, OKW[i]);
    if (!ok) return "only read-only statements are allowed: SELECT, WITH … SELECT, SHOW, EXPLAIN, DESCRIBE";
    /* 多条语句: 字符串和注释外的分号。模型写 "SELECT 1; DROP TABLE x" 这招白名单看不见第二条 */
    char q = 0;
    for (const char *p = s; *p; p++) {
        if (q) { if (*p == q) { if (p[1] == q) p++; else q = 0; } continue; }
        if (*p == '\'' || *p == '"' || *p == '`') { q = *p; continue; }
        if (p[0] == '-' && p[1] == '-') { p = strchr(p, '\n'); if (!p) break; continue; }
        if (p[0] == '/' && p[1] == '*') { const char *e = strstr(p + 2, "*/"); if (!e) break; p = e + 1; continue; }
        if (*p == ';') { const char *r = p + 1; while (*r && isspace((unsigned char)*r)) r++; if (*r) return "one statement at a time (no ';' between statements)"; }
    }
    /* WITH / EXPLAIN 后面藏写语句: PostgreSQL 的 WITH x AS (DELETE … RETURNING *) SELECT …, MySQL 的 EXPLAIN ANALYZE UPDATE …(真执行)。
     * 字符串外任何深度出现这些词都拒(is_delete 这种带下划线的不算词) —— 真正的第二道锁是只读事务, 这里只是让模型早点看到原因 */
    if (word_is(s, "WITH") || word_is(s, "EXPLAIN")) {
        static const char *BAD[] = { "INSERT", "UPDATE", "DELETE", "MERGE", "CREATE", "DROP", "ALTER", "TRUNCATE", "GRANT", "COPY", "CALL", "LOCK" };
        q = 0;
        for (const char *p = s; *p; p++) {
            if (q) { if (*p == q) q = 0; continue; }
            if (*p == '\'' || *p == '"' || *p == '`') { q = *p; continue; }
            if (p == s || (!isalnum((unsigned char)p[-1]) && p[-1] != '_'))
                for (size_t i = 0; i < sizeof BAD / sizeof BAD[0]; i++) if (word_is(p, BAD[i])) return "only read-only statements are allowed: a writing statement is not allowed after WITH or EXPLAIN";
        }
    }
    return NULL;
}

char *sql_wrap_limit(const char *sql, long limit, const char *dialect) {
    const char *s = skip_lead(sql);
    ds4_buf b = {0};
    char body[DB_SQL_MAX + 1]; snprintf(body, sizeof body, "%s", sql);
    size_t n = strlen(body); while (n && (isspace((unsigned char)body[n - 1]) || body[n - 1] == ';')) body[--n] = 0;   /* 末尾分号包进子查询是语法错 */
    if (word_is(s, "SELECT") || word_is(s, "WITH") || word_is(s, "VALUES") || word_is(s, "TABLE") || *s == '(')
        ds4_buf_printf(&b, "SELECT * FROM (%s) AS q LIMIT %ld", body, limit);
    else ds4_buf_puts(&b, body);
    (void)dialect;
    return ds4_buf_take(&b);
}

void sql_table_cols(sql_table *t, int n) { t->ncols = n; t->cols = calloc((size_t)(n > 0 ? n : 1), sizeof(char *)); t->rows = 0; }
void sql_table_col(sql_table *t, int i, const char *name) { if (i >= 0 && i < t->ncols) { free(t->cols[i]); t->cols[i] = strdup(name ? name : "?"); } }
void sql_table_row_begin(sql_table *t) {
    if (!t->rows) { for (int i = 0; i < t->ncols; i++) ds4_buf_printf(&t->out, "%s%s", i ? "\t" : "", t->cols[i] ? t->cols[i] : "?"); ds4_buf_putc(&t->out, '\n'); }
    ds4_buf_putc(&t->out, '{');
}
void sql_table_cell(sql_table *t, int i, const char *text, bool is_null, bool numeric) {
    if (i) ds4_buf_putc(&t->out, ',');
    ds4_json_escape(&t->out, i < t->ncols && t->cols[i] ? t->cols[i] : "?"); ds4_buf_putc(&t->out, ':');
    if (is_null || !text) { ds4_buf_puts(&t->out, "null"); return; }
    if (numeric) {   /* 数值列原文直出; 但服务器给的 NaN/Infinity 不是 JSON */
        const bool plain = text[0] && strspn(text, "+-0123456789.eE") == strlen(text);
        if (plain) ds4_buf_puts(&t->out, text); else ds4_json_escape(&t->out, text);
        return;
    }
    ds4_json_escape(&t->out, text);
}
void sql_table_row_end(sql_table *t) { ds4_buf_puts(&t->out, "}\n"); t->rows++; }
void sql_table_free(sql_table *t) { for (int i = 0; i < t->ncols; i++) free(t->cols[i]); free(t->cols); ds4_buf_free(&t->out); memset(t, 0, sizeof *t); }
