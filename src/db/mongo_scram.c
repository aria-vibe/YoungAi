/* mongo_scram.c — SCRAM-SHA-256(RFC 5802 流程, RFC 7677 的 SHA-256 参数)客户端那半边, 纯算术, 不碰网络: 单测直接拿 RFC 7677 §3 的向量逐字节对。
 * MongoDB 的 SCRAM-SHA-256 用的是 SASLprep 后的原始密码(不像 SHA-1 那样先 MD5 一遍); 密码只许可见 ASCII(凭证接口本来就只收这种), SASLprep 对它是恒等。
 * 密钥材料(SaltedPassword / ClientKey / StoredKey)算完立刻清零: 这几样拿到就等于拿到密码。 */
#include "db.h"
#include "../common/ds4_sha256.h"
#include "../push/ds4_push.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t scram_b64(const uint8_t *p, size_t n, char *out) {
    static const char A[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t v = (uint32_t)p[i] << 16 | (i + 1 < n ? (uint32_t)p[i + 1] << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
        out[o++] = A[v >> 18]; out[o++] = A[(v >> 12) & 63];
        out[o++] = i + 1 < n ? A[(v >> 6) & 63] : '=';
        out[o++] = i + 2 < n ? A[v & 63] : '=';
    }
    out[o] = 0;
    return o;
}

/* 用户名里的 , 和 = 按 RFC 5802 §5.1 写成 =2C =3D(不然服务器把它当字段分隔) */
bool scram_client_first(const char *user, const char *cnonce, char *out, size_t n) {
    size_t o = (size_t)snprintf(out, n, "n,,n=");
    for (const char *p = user; *p; p++) {
        const char *rep = *p == ',' ? "=2C" : *p == '=' ? "=3D" : NULL;
        if (o + 4 >= n) return false;
        if (rep) { memcpy(out + o, rep, 3); o += 3; } else out[o++] = *p;
    }
    return snprintf(out + o, n - o, ",r=%s", cnonce) < (int)(n - o);
}

/* server-first 里取 "x=" 字段的值(到下一个逗号) */
static bool field(const char *msg, char key, char *out, size_t n) {
    for (const char *p = msg; p && *p; p = strchr(p, ','), p = p ? p + 1 : NULL) {
        if (p[0] == key && p[1] == '=') {
            const char *e = strchr(p, ','); const size_t len = e ? (size_t)(e - p - 2) : strlen(p + 2);
            if (len + 1 > n) return false;
            memcpy(out, p + 2, len); out[len] = 0;
            return true;
        }
    }
    return false;
}

bool scram_client_final(const char *user, const char *pw, const char *cnonce, const char *server_first, char *final_out, size_t fn, uint8_t server_sig[32], char *err, size_t errn) {
    char nonce[256], salt_b64[256], iter_s[32];
    if (!field(server_first, 'r', nonce, sizeof nonce) || !field(server_first, 's', salt_b64, sizeof salt_b64) || !field(server_first, 'i', iter_s, sizeof iter_s)) { snprintf(err, errn, "malformed SCRAM server-first message"); return false; }
    if (strncmp(nonce, cnonce, strlen(cnonce))) { snprintf(err, errn, "SCRAM server nonce does not start with ours"); return false; }   /* 不对就是中间人或错包 */
    const long iter = strtol(iter_s, NULL, 10);
    if (iter < 4096 || iter > 10000000) { snprintf(err, errn, "SCRAM iteration count %ld out of range", iter); return false; }   /* RFC 要求 ≥ 4096; 上限防服务器让我们算几分钟 */
    uint8_t salt[128]; size_t saltn = 0;
    if (!ds4_b64url_dec(salt_b64, salt, sizeof salt, &saltn) || !saltn) { snprintf(err, errn, "SCRAM salt is not base64"); return false; }

    char bare[512]; if (!scram_client_first(user, cnonce, bare, sizeof bare)) { snprintf(err, errn, "user name too long"); return false; }
    char wo_proof[320]; snprintf(wo_proof, sizeof wo_proof, "c=biws,r=%s", nonce);   /* biws = base64("n,,"): 不做 channel binding */
    ds4_buf auth = {0};   /* AuthMessage = client-first-bare , server-first , client-final-without-proof */
    ds4_buf_puts(&auth, bare + 3); ds4_buf_putc(&auth, ','); ds4_buf_puts(&auth, server_first); ds4_buf_putc(&auth, ','); ds4_buf_puts(&auth, wo_proof);

    uint8_t salted[32], ckey[32], stored[32], csig[32], skey[32], proof[32];
    ds4_pbkdf2_sha256(pw, strlen(pw), salt, saltn, (uint32_t)iter, salted, 32);
    ds4_hmac_sha256(salted, 32, "Client Key", 10, ckey);
    ds4_sha256(ckey, 32, stored);
    ds4_hmac_sha256(stored, 32, auth.ptr, auth.len, csig);
    for (int i = 0; i < 32; i++) proof[i] = ckey[i] ^ csig[i];
    ds4_hmac_sha256(salted, 32, "Server Key", 10, skey);
    ds4_hmac_sha256(skey, 32, auth.ptr, auth.len, server_sig);
    ds4_buf_free(&auth);
    memset(salted, 0, 32); memset(ckey, 0, 32); memset(stored, 0, 32); memset(skey, 0, 32);

    char proof_b64[48]; scram_b64(proof, 32, proof_b64);
    memset(proof, 0, 32);
    return snprintf(final_out, fn, "%s,p=%s", wo_proof, proof_b64) < (int)fn;
}

bool scram_server_ok(const uint8_t server_sig[32], const char *server_final) {
    char v[128]; uint8_t got[64]; size_t n = 0;
    if (!field(server_final, 'v', v, sizeof v) || !ds4_b64url_dec(v, got, sizeof got, &n) || n != 32) return false;
    int diff = 0; for (int i = 0; i < 32; i++) diff |= got[i] ^ server_sig[i];
    return diff == 0;
}
