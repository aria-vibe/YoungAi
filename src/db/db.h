/* db.h — 连接器支持的数据库种类表 + 统一的查询入口(方案稿 docs/assistant_connectors.md)。
 * 一种库一份协议文件(mongo_*.c / pg_wire.c / mysql_*.c / redis.c), 这里只做分发; 页面的下拉、表单字段、默认端口都从 DB_KINDS 来, 加一种库页面不用改。
 * 全部纯 C 自己写线协议: 不链任何客户端库(spark 上没有, 本项目零依赖); SQLite 没有线协议, 走沙盒里工具箱的 sqlite3(tool_db.c 管)。
 * 只读在每份协议文件里各自保证(代码里只有读命令 / 只读事务 / 命令白名单), 不靠提示词。 */
#ifndef DS4_DB_H
#define DS4_DB_H
#include "mongo.h"

#define DB_LIMIT_DEFAULT 50
#define DB_LIMIT_MAX 500       /* 一次最多给这么多条: 25495 条 news_items 一把全拉就是几十 MB 的上下文 */
#define DB_SQL_MAX 8192        /* 模型写的一条 SQL 的长度上限 */

typedef struct {
    const char *kind, *label;    /* 页面下拉的值与显示名 */
    int port;                    /* 默认端口; 0 = 没有(sqlite) */
    const char *fields;          /* 表单要填的字段(逗号表): host port db user password auth_db path */
    const char *whats;           /* 这种库的 what 枚举(逗号表), 给 query_db 的 schema */
    const char *hint;            /* 工具描述里一句: 这种库怎么查 */
} db_kind;
extern const db_kind DB_KINDS[];
extern const int DB_NKINDS;
const db_kind *db_kind_find(const char *kind);

typedef struct { const char *kind, *host, *db, *user, *pw, *auth_db, *path; int port; } db_target;   /* pw 只在握手那一刻读 */
typedef struct {
    const char *what;            /* 各种库自己的枚举, 见 DB_KINDS.whats */
    const char *coll;            /* 集合 / 表 */
    const char *filter, *projection, *sort, *field, *pipeline;   /* mongo */
    const char *sql;             /* SQL 三家 */
    const char *key, *pattern, *command;   /* redis */
    long limit;
} db_req;

/* 执行; 返回 malloc 文本("error: …" 或 摘要行 + 结果行), *ok 成败。sqlite 不走这里(要沙盒, 在 tool_db.c) */
char *db_query(const db_target *t, const db_req *r, bool *ok);
/* 测试连接: 一行人话 "MongoDB 7.0.14 · quant_trading · 13 collections" 或 err */
bool db_probe(const db_target *t, char *line, size_t n, char *err, size_t errn);

/* ---- sql_guard.c: 三家 SQL 共用 ---- */
/* 只放行只读语句: SELECT / WITH … SELECT / SHOW / DESCRIBE|DESC / EXPLAIN / VALUES / TABLE; 多条(分号)拒; 返回 NULL = 放行, 否则一句原因 */
const char *sql_readonly_check(const char *sql);
/* 把 SELECT 包成 SELECT * FROM (<sql>) AS q LIMIT n(SHOW/EXPLAIN 这类不能包的原样); out 为 malloc */
char *sql_wrap_limit(const char *sql, long limit, const char *dialect);
/* 结果表 → 文本: 第一行列名(tab 分隔), 之后一行一条 JSON 对象 {col: value}; 调用方逐行喂 */
typedef struct { ds4_buf out; int ncols; char **cols; long rows; } sql_table;
void sql_table_cols(sql_table *t, int n);                     /* 开 n 列 */
void sql_table_col(sql_table *t, int i, const char *name);
void sql_table_row_begin(sql_table *t);
void sql_table_cell(sql_table *t, int i, const char *text, bool is_null, bool numeric);   /* text 原文; numeric = 不加引号 */
void sql_table_row_end(sql_table *t);
void sql_table_free(sql_table *t);

/* ---- pg_wire.c / mysql_wire.c / redis.c ---- */
char *pg_query(const db_target *t, const db_req *r, bool *ok);
bool pg_probe(const db_target *t, char *line, size_t n, char *err, size_t errn);
char *mysql_query(const db_target *t, const db_req *r, bool *ok);
bool mysql_probe(const db_target *t, char *line, size_t n, char *err, size_t errn);
char *redis_query(const db_target *t, const db_req *r, bool *ok);
bool redis_probe(const db_target *t, char *line, size_t n, char *err, size_t errn);

/* ---- mysql_auth.c ---- */
void mysql_sha1(const void *data, size_t n, uint8_t out[20]);
void mysql_scramble_native(const char *pw, const uint8_t nonce[20], uint8_t out[20]);
void mysql_scramble_sha2(const char *pw, const uint8_t nonce[20], uint8_t out[32]);
bool mysql_rsa_oaep(const char *pem, const uint8_t *msg, size_t mn, uint8_t *out, size_t *on);   /* out ≥ 512 */
uint64_t mysql_bn_powmod_small(uint64_t m, uint32_t e, uint64_t n);   /* 单测验大数 */

/* ---- 共用的 TCP 小工具(db_net.c) ---- */
int db_tcp_connect(const char *host, int port, char *err, size_t errn);   /* 带 5 s 连接超时 + 20 s 收发超时; 失败 -1 */
bool db_io_all(int fd, void *buf, size_t n, bool write_);

#endif
