/* db.c — 种类表 + 分发(db.h)。页面下拉、表单字段、默认端口、query_db 的 what 枚举都从 DB_KINDS 来。
 * 加一种库: 这里加一行 + 一份协议文件; 页面(studio.html 连接器卡)和工具(tool_db.c)一行不改。 */
#include "db.h"
#include <stdio.h>
#include <string.h>

const db_kind DB_KINDS[] = {
    { "mongodb", "MongoDB", 27017, "host,port,db,user,password,auth_db", "collections,fields,find,count,distinct,aggregate",
      "MongoDB: what=collections lists collections with document counts; fields samples a collection's keys and types; find/count/distinct take a MongoDB filter as JSON (use {\"$oid\":…} and {\"$date\":…} exactly as they appear in results); aggregate takes pipeline as a JSON array of stages." },
    { "postgresql", "PostgreSQL", 5432, "host,port,db,user,password", "tables,fields,sql",
      "PostgreSQL: what=tables lists tables with row estimates; fields lists a table's columns and types; sql runs one read-only SELECT (standard PostgreSQL SQL; schema-qualify tables outside public)." },
    { "mysql", "MySQL / MariaDB", 3306, "host,port,db,user,password", "tables,fields,sql",
      "MySQL/MariaDB: what=tables lists tables with row estimates; fields lists a table's columns and types; sql runs one read-only SELECT (MySQL dialect)." },
    { "redis", "Redis", 6379, "host,port,db,user,password", "keys,get,info,command",
      "Redis: what=keys scans keys matching pattern (default *); get reads one key whatever its type (string, hash, list, set, zset, stream); info gives server stats; command runs one read-only command given as text (e.g. 'HGETALL user:1', 'ZRANGE top 0 10 WITHSCORES')." },
    { "sqlite", "SQLite file", 0, "path", "tables,fields,sql",
      "SQLite: a database file on this machine; what=tables, fields and sql work as for PostgreSQL (SQLite dialect)." },
};
const int DB_NKINDS = (int)(sizeof DB_KINDS / sizeof DB_KINDS[0]);

const db_kind *db_kind_find(const char *kind) {
    if (!kind) return NULL;
    for (int i = 0; i < DB_NKINDS; i++) if (!strcmp(DB_KINDS[i].kind, kind)) return &DB_KINDS[i];
    return NULL;
}

static char *errf(const char *msg) { ds4_buf b = {0}; ds4_buf_printf(&b, "error: %s\n", msg); return ds4_buf_take(&b); }

char *db_query(const db_target *t, const db_req *r, bool *ok) {
    *ok = false;
    const db_kind *k = db_kind_find(t->kind);
    if (!k) return errf("unknown connector kind");
    if (!strcmp(k->kind, "mongodb")) {
        const mongo_target mt = { t->host, t->port, t->db, t->auth_db, t->user, t->pw };
        const mongo_req mr = { r->what, r->coll, r->filter, r->projection, r->sort, r->field, r->pipeline, r->limit };
        return mongo_query(&mt, &mr, ok);
    }
    if (!strcmp(k->kind, "postgresql")) return pg_query(t, r, ok);
    if (!strcmp(k->kind, "mysql")) return mysql_query(t, r, ok);
    if (!strcmp(k->kind, "redis")) return redis_query(t, r, ok);
    return errf("this kind is queried through the sandbox, not here");   /* sqlite: tool_db.c */
}

bool db_probe(const db_target *t, char *line, size_t n, char *err, size_t errn) {
    const db_kind *k = db_kind_find(t->kind);
    if (!k) { snprintf(err, errn, "unknown connector kind"); return false; }
    if (!strcmp(k->kind, "mongodb")) {
        const mongo_target mt = { t->host, t->port, t->db, t->auth_db, t->user, t->pw };
        char ver[32], dbs[512]; int ncoll;
        if (!mongo_probe(&mt, ver, sizeof ver, &ncoll, dbs, sizeof dbs, err, errn)) return false;
        if (ncoll) snprintf(line, n, "MongoDB %s · %s · %d collections", ver[0] ? ver : "?", t->db, ncoll);
        else snprintf(line, n, "MongoDB %s · database %s has no collections (check the name; the server has: %s)", ver[0] ? ver : "?", t->db, dbs[0] ? dbs : "nothing visible to this user");
        return true;
    }
    if (!strcmp(k->kind, "postgresql")) return pg_probe(t, line, n, err, errn);
    if (!strcmp(k->kind, "mysql")) return mysql_probe(t, line, n, err, errn);
    if (!strcmp(k->kind, "redis")) return redis_probe(t, line, n, err, errn);
    snprintf(err, errn, "this kind is probed through the sandbox, not here");
    return false;
}
