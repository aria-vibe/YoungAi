/* mongo.h — MongoDB 线协议客户端(纯 C, 方案稿 docs/assistant_connectors.md §5): TCP + OP_MSG + SCRAM-SHA-256 + 只读命令。
 * 不链 libmongoc、不调 mongosh: spark 上没有, 本项目零依赖; 密码学原语全在 src/common/ds4_sha256。
 * 只做明文 TCP(本机/局域网), 不做 TLS / +srv / 压缩 / SCRAM-SHA-1(MongoDB 4.0+ 默认 SHA-256)。 */
#ifndef DS4_MONGO_H
#define DS4_MONGO_H
#include "bson.h"

#define MONGO_CONNECT_S 5        /* 连不上 5 秒就算了: 本机/局域网一毫秒就通, 等更久只是让用户干等 */
#define MONGO_IO_S 20            /* 单次收发: 要比 maxTimeMS(10 s)长, 让服务器先报"超时"而不是我们先断线 */
#define MONGO_MAX_MSG (48u << 20)   /* 服务器的 maxMessageSizeBytes; 超过就是坏包 */
#define MONGO_MAX_TIME_MS 10000  /* 每条查询命令都带: 一条烂的 aggregate 不能把交易 agent 的库卡住 */
#define MONGO_BATCH 101          /* 服务器默认首批; 用 limit 更小的那个 */

typedef struct {
    int fd;
    int32_t req_id;
    char err[320];
} mongo_conn;

typedef struct { const char *host; int port; const char *db, *auth_db, *user, *pw; } mongo_target;   /* pw 只在 saslStart 那一刻读, 不留副本 */

/* ---- mongo_wire.c ---- */
bool mongo_connect(mongo_conn *c, const mongo_target *t);   /* TCP + hello + (有账号)SCRAM; 失败 c->err */
void mongo_close(mongo_conn *c);
/* 发一条命令(cmd = 完整 BSON 文档, 必须含 $db), 收回复文档(malloc 到 *reply, 长度 *n)。只管收发; 回复里 ok 由 mongo_ok 看 */
bool mongo_command(mongo_conn *c, const uint8_t *cmd, size_t cmdn, uint8_t **reply, size_t *n);
bool mongo_ok(mongo_conn *c, const uint8_t *reply, size_t n);   /* ok:1 → true; 否则 err = "server: <errmsg> (code N)" */

/* ---- mongo_scram.c(RFC 5802/7677; 金标 RFC 7677 §3 向量) ---- */
#define SCRAM_NONCE_LEN 24   /* 随机字节数, base64 后 32 字符 */
bool scram_client_first(const char *user, const char *cnonce, char *out, size_t n);   /* "n,,n=<user>,r=<cnonce>"; user 里的 , = 按 RFC 转义 */
/* 算 client-final 与期望的 server signature; server_first = 服务器回的 "r=…,s=…,i=…" */
bool scram_client_final(const char *user, const char *pw, const char *cnonce, const char *server_first, char *final_out, size_t fn, uint8_t server_sig[32], char *err, size_t errn);
bool scram_server_ok(const uint8_t server_sig[32], const char *server_final);   /* "v=<base64>" 逐字节对 */
size_t scram_b64(const uint8_t *p, size_t n, char *out);   /* 标准 base64 带 = 填充(SCRAM 用这种; 推送的 base64url 不带); out ≥ 4*((n+2)/3)+1 */

/* ---- mongo_query.c: 只读命令 ---- */
typedef struct {
    const char *what;        /* collections | fields | find | count | distinct | aggregate */
    const char *coll, *filter, *projection, *sort, *field, *pipeline;   /* JSON 原文; 没有 = NULL */
    long limit;              /* find/aggregate 的条数上限(调用方已钳) */
} mongo_req;
/* 模型给的 JSON 里有没有会写库或跑服务器端 JS 的键($out $merge $where $function $accumulator); 有 → 返回那个键名 */
const char *mongo_forbidden(const char *json);
/* 执行; 返回 malloc 文本(成功: 第一行摘要 + 一行一条 JSON; 失败: "error: …"), *ok 告诉调用方成败 */
char *mongo_query(const mongo_target *t, const mongo_req *q, bool *ok);
/* 测试连接: version(buildInfo)、集合数、库名列表(库不存在时给用户看有哪些) */
bool mongo_probe(const mongo_target *t, char *version, size_t vn, int *ncoll, char *dbs, size_t dn, char *err, size_t errn);

#endif
