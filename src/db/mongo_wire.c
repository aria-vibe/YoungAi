/* mongo_wire.c — TCP 连接 + OP_MSG(opcode 2013)收发 + hello + SCRAM-SHA-256 握手(mongo.h)。
 * 一次工具调用连一次、用完关(本机/局域网握手几毫秒, 连接池不值得写); 每条命令同步一问一答。
 * 回包先看 16 字节头: 长度 < 21 或 > 48 MB、opCode 不是 2013、responseTo 对不上 → 当坏包断线, 不按它的长度分内存。
 * 密码只在 saslStart 那一刻从调用方的缓冲读, 这里不复制、不落日志; err 里永远不带它。 */
#include "db.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define OP_MSG 2013

static void le32(uint8_t *p, int32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)((uint32_t)v >> (8 * i)); }
static int32_t rd32(const uint8_t *p) { return (int32_t)((uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24); }

bool mongo_command(mongo_conn *c, const uint8_t *cmd, size_t cmdn, uint8_t **reply, size_t *n) {
    *reply = NULL; *n = 0;
    if (c->fd < 0) { snprintf(c->err, sizeof c->err, "not connected"); return false; }
    const int32_t id = ++c->req_id;
    uint8_t hdr[21];   /* messageLength, requestID, responseTo, opCode, flagBits, section kind 0 */
    le32(hdr, (int32_t)(21 + cmdn)); le32(hdr + 4, id); le32(hdr + 8, 0); le32(hdr + 12, OP_MSG); le32(hdr + 16, 0); hdr[20] = 0;
    if (!db_io_all(c->fd, hdr, 21, true) || !db_io_all(c->fd, (void *)cmd, cmdn, true)) { snprintf(c->err, sizeof c->err, "send failed (%s)", strerror(errno)); return false; }
    uint8_t rh[16];
    if (!db_io_all(c->fd, rh, 16, false)) { snprintf(c->err, sizeof c->err, "no reply (%s)", errno == EAGAIN || errno == EWOULDBLOCK ? "timed out" : strerror(errno)); return false; }
    const int32_t len = rd32(rh), resp_to = rd32(rh + 8), op = rd32(rh + 12);
    if (len < 21 || (size_t)len > MONGO_MAX_MSG || op != OP_MSG || resp_to != id) { snprintf(c->err, sizeof c->err, "malformed reply (len %d op %d)", len, op); return false; }
    uint8_t *body = malloc((size_t)len - 16);
    if (!body || !db_io_all(c->fd, body, (size_t)len - 16, false)) { free(body); snprintf(c->err, sizeof c->err, "reply cut short"); return false; }
    if (body[4] != 0 || !bson_doclen(body + 5, (size_t)len - 21)) { free(body); snprintf(c->err, sizeof c->err, "malformed reply body"); return false; }   /* 只认 kind-0 section; 服务器回复从不带 kind-1 */
    *n = bson_doclen(body + 5, (size_t)len - 21);
    memmove(body, body + 5, *n);
    *reply = body;
    return true;
}

bool mongo_ok(mongo_conn *c, const uint8_t *reply, size_t n) {
    uint8_t t; const uint8_t *v; size_t vn;
    if (bson_find(reply, n, "ok", &t, &v, &vn) && bson_dbl(t, v) == 1.0) return true;
    const char *msg = bson_find(reply, n, "errmsg", &t, &v, &vn) ? bson_string(t, v) : NULL;
    const long code = bson_find(reply, n, "code", &t, &v, &vn) ? (long)bson_num(t, v) : 0;
    snprintf(c->err, sizeof c->err, "server: %s (code %ld)", msg ? msg : "command failed", code);
    return false;
}

static bool run(mongo_conn *c, ds4_buf *cmd, uint8_t **reply, size_t *n) {   /* 发 + 收 + 看 ok; cmd 用完释放 */
    const bool sent = mongo_command(c, (const uint8_t *)cmd->ptr, cmd->len, reply, n);
    ds4_buf_free(cmd);
    if (!sent) return false;
    if (mongo_ok(c, *reply, *n)) return true;
    free(*reply); *reply = NULL;
    return false;
}

/* SCRAM: saslStart(client-first) → saslContinue(client-final) → 校验 v=; skipEmptyExchange 省掉第三回合(服务器不支持就再发一个空 payload) */
static bool auth(mongo_conn *c, const mongo_target *t) {
    uint8_t rnd[SCRAM_NONCE_LEN]; char cnonce[48];
    FILE *f = fopen("/dev/urandom", "rb"); const bool got = f && fread(rnd, 1, sizeof rnd, f) == sizeof rnd; if (f) fclose(f);
    if (!got) { snprintf(c->err, sizeof c->err, "no random source"); return false; }
    scram_b64(rnd, sizeof rnd, cnonce);
    char first[512]; if (!scram_client_first(t->user, cnonce, first, sizeof first)) { snprintf(c->err, sizeof c->err, "user name too long"); return false; }
    const char *adb = t->auth_db && t->auth_db[0] ? t->auth_db : "admin";
    ds4_buf cmd = {0}; size_t s = bson_begin(&cmd);
    bson_int32(&cmd, "saslStart", 1); bson_str(&cmd, "mechanism", "SCRAM-SHA-256"); bson_bin(&cmd, "payload", first, strlen(first)); bson_bool(&cmd, "autoAuthorize", true);
    size_t o = bson_sub(&cmd, "options", false); bson_bool(&cmd, "skipEmptyExchange", true); bson_end(&cmd, o);
    bson_str(&cmd, "$db", adb); bson_end(&cmd, s);
    uint8_t *rep; size_t rn;
    if (!run(c, &cmd, &rep, &rn)) { if (strstr(c->err, "code 18")) snprintf(c->err, sizeof c->err, "authentication failed for user %s on %s (code 18; the user may live in another authSource)", t->user, adb); return false; }
    uint8_t ty; const uint8_t *v; size_t vn; char server_first[512] = ""; int64_t conv = 0;
    if (bson_find(rep, rn, "payload", &ty, &v, &vn) && ty == 5) { const int32_t l = rd32(v); if (l > 0 && (size_t)l < sizeof server_first) { memcpy(server_first, v + 5, (size_t)l); server_first[l] = 0; } }
    if (bson_find(rep, rn, "conversationId", &ty, &v, &vn)) conv = bson_num(ty, v);
    free(rep);
    char final_[512]; uint8_t ssig[32];
    if (!scram_client_final(t->user, t->pw ? t->pw : "", cnonce, server_first, final_, sizeof final_, ssig, c->err, sizeof c->err)) return false;
    for (int round = 0; round < 2; round++) {
        s = bson_begin(&cmd);
        bson_int32(&cmd, "saslContinue", 1); bson_int32(&cmd, "conversationId", (int32_t)conv); bson_bin(&cmd, "payload", round ? "" : final_, round ? 0 : strlen(final_)); bson_str(&cmd, "$db", adb); bson_end(&cmd, s);
        if (!run(c, &cmd, &rep, &rn)) { if (strstr(c->err, "code 18")) snprintf(c->err, sizeof c->err, "authentication failed for user %s on %s (code 18; wrong password, or the user lives in another authSource)", t->user, adb); return false; }
        char server_final[256] = "";
        if (bson_find(rep, rn, "payload", &ty, &v, &vn) && ty == 5) { const int32_t l = rd32(v); if (l > 0 && (size_t)l < sizeof server_final) { memcpy(server_final, v + 5, (size_t)l); server_final[l] = 0; } }
        const bool done = bson_find(rep, rn, "done", &ty, &v, &vn) && bson_num(ty, v);
        free(rep);
        if (round == 0 && !scram_server_ok(ssig, server_final)) { snprintf(c->err, sizeof c->err, "server signature mismatch (not the server that knows this password?)"); return false; }
        if (done) return true;
    }
    snprintf(c->err, sizeof c->err, "SCRAM did not finish");
    return false;
}

bool mongo_connect(mongo_conn *c, const mongo_target *t) {
    c->fd = -1; c->req_id = 0; c->err[0] = 0;
    if (!t->host || !t->host[0]) { snprintf(c->err, sizeof c->err, "no host"); return false; }
    if (strstr(t->host, "://") || !strncmp(t->host, "mongodb+srv", 11)) { snprintf(c->err, sizeof c->err, "TLS and +srv are not supported; use host:port on the local network"); return false; }
    if ((c->fd = db_tcp_connect(t->host, t->port > 0 ? t->port : 27017, c->err, sizeof c->err)) < 0) return false;
    ds4_buf cmd = {0}; const size_t s = bson_begin(&cmd);
    bson_int32(&cmd, "hello", 1); bson_str(&cmd, "$db", "admin");
    if (t->user && t->user[0]) { char su[256]; snprintf(su, sizeof su, "%s.%s", t->auth_db && t->auth_db[0] ? t->auth_db : "admin", t->user); bson_str(&cmd, "saslSupportedMechs", su); }
    bson_end(&cmd, s);
    uint8_t *rep; size_t rn;
    if (!run(c, &cmd, &rep, &rn)) { mongo_close(c); return false; }
    uint8_t ty; const uint8_t *v; size_t vn;
    const bool wire_ok = bson_find(rep, rn, "maxWireVersion", &ty, &v, &vn) && bson_num(ty, v) >= 6;   /* OP_MSG 从 3.6(wire 6)起 */
    bool sha256 = true;
    if (t->user && t->user[0] && bson_find(rep, rn, "saslSupportedMechs", &ty, &v, &vn) && ty == 4) {   /* 服务器列了这个用户能用的机制: 没有 SHA-256 就明说, 别让用户猜密码错了 */
        bson_iter it; uint8_t et; const char *ek; const uint8_t *ev; size_t en; sha256 = false;
        if (bson_iter_init(&it, v, vn)) while (bson_next(&it, &et, &ek, &ev, &en)) if (bson_string(et, ev) && !strcmp(bson_string(et, ev), "SCRAM-SHA-256")) sha256 = true;
    }
    free(rep);
    if (!wire_ok) { snprintf(c->err, sizeof c->err, "server too old (needs MongoDB 3.6 or newer)"); mongo_close(c); return false; }
    if (!sha256) { snprintf(c->err, sizeof c->err, "server offers only SCRAM-SHA-1 for this user; enable SCRAM-SHA-256 on the server (MongoDB 4.0+)"); mongo_close(c); return false; }
    if (t->user && t->user[0] && !auth(c, t)) { mongo_close(c); return false; }
    return true;
}

void mongo_close(mongo_conn *c) { if (c->fd >= 0) close(c->fd); c->fd = -1; }
