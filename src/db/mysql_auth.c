/* mysql_auth.c — MySQL / MariaDB 两种口令算法 + 为 caching_sha2_password 全量鉴权准备的 RSA-OAEP(纯 C 大数, 只做公钥加密这一件事)。
 *   mysql_native_password(MariaDB 默认, MySQL ≤ 8.3 可选): SHA1(pw) XOR SHA1(nonce ‖ SHA1(SHA1(pw)))        —— 要 SHA-1, 全仓只有这里用, 就放这里
 *   caching_sha2_password(MySQL 8 默认):  SHA256(pw) XOR SHA256(SHA256(SHA256(pw)) ‖ nonce) 先试快路(服务器缓存里有这个用户才成);
 *     不成就"全量鉴权": 明文 TCP 上服务器要我们用它的 RSA 公钥把 (pw ‖ 0) XOR nonce 加密了发 —— 所以要 RSA 公钥加密(PKCS#1 OAEP, SHA-1)。
 * RSA 只有 c = m^e mod n, e = 65537: 大数乘 + 移位减法取模, 2048 位一次加密毫秒级, 不值得写蒙哥马利。
 * 金标: tests/asst/t_asst_db.c(SHA-1 FIPS 向量、native 算法对 MySQL 文档例子、RSA 对小数手算)。 */
#include "db.h"
#include "../common/ds4_sha256.h"
#include "../push/ds4_push.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- SHA-1(FIPS 180-4) ---- */
static uint32_t rol(uint32_t x, int n) { return x << n | x >> (32 - n); }
static void sha1_block(uint32_t h[5], const uint8_t *p) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4 * i] << 24 | (uint32_t)p[4 * i + 1] << 16 | (uint32_t)p[4 * i + 2] << 8 | p[4 * i + 3];
    for (int i = 16; i < 80; i++) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) { f = (b & c) | (~b & d); k = 0x5A827999; } else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; } else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; } else { f = b ^ c ^ d; k = 0xCA62C1D6; }
        const uint32_t t = rol(a, 5) + f + e + k + w[i]; e = d; d = c; c = rol(b, 30); b = a; a = t;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
}
void mysql_sha1(const void *data, size_t n, uint8_t out[20]) {
    uint32_t h[5] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };
    const uint8_t *p = data; size_t i = 0;
    for (; i + 64 <= n; i += 64) sha1_block(h, p + i);
    uint8_t last[128] = {0}; const size_t rem = n - i; memcpy(last, p + i, rem); last[rem] = 0x80;
    const size_t blk = rem < 56 ? 64 : 128; const uint64_t bits = (uint64_t)n * 8;
    for (int k = 0; k < 8; k++) last[blk - 1 - k] = (uint8_t)(bits >> (8 * k));
    sha1_block(h, last); if (blk == 128) sha1_block(h, last + 64);
    for (int k = 0; k < 5; k++) { out[4 * k] = h[k] >> 24; out[4 * k + 1] = h[k] >> 16; out[4 * k + 2] = h[k] >> 8; out[4 * k + 3] = (uint8_t)h[k]; }
}

/* ---- 两种 scramble ---- */
void mysql_scramble_native(const char *pw, const uint8_t nonce[20], uint8_t out[20]) {
    uint8_t s1[20], s2[20], buf[40];
    mysql_sha1(pw, strlen(pw), s1); mysql_sha1(s1, 20, s2);
    memcpy(buf, nonce, 20); memcpy(buf + 20, s2, 20); mysql_sha1(buf, 40, out);
    for (int i = 0; i < 20; i++) out[i] ^= s1[i];
    memset(s1, 0, 20);
}
void mysql_scramble_sha2(const char *pw, const uint8_t nonce[20], uint8_t out[32]) {
    uint8_t s1[32], s2[32], buf[52];
    ds4_sha256(pw, strlen(pw), s1); ds4_sha256(s1, 32, s2);
    memcpy(buf, s2, 32); memcpy(buf + 32, nonce, 20); ds4_sha256(buf, 52, out);
    for (int i = 0; i < 32; i++) out[i] ^= s1[i];
    memset(s1, 0, 32);
}

/* ---- 大数(大端字节 → 32 位 limb 小端) ---- */
#define BN_MAX 136   /* 4352 位: 够 4096 位的模 + 余量 */
typedef struct { uint32_t v[BN_MAX]; int n; } bn;
static void bn_from(bn *a, const uint8_t *p, size_t len) { memset(a, 0, sizeof *a); for (size_t i = 0; i < len && i / 4 < BN_MAX; i++) a->v[i / 4] |= (uint32_t)p[len - 1 - i] << (8 * (i % 4)); a->n = BN_MAX; while (a->n > 1 && !a->v[a->n - 1]) a->n--; }
static int bn_cmp(const bn *a, const bn *b) { for (int i = BN_MAX - 1; i >= 0; i--) if (a->v[i] != b->v[i]) return a->v[i] < b->v[i] ? -1 : 1; return 0; }
static void bn_sub(bn *a, const bn *b) { uint64_t br = 0; for (int i = 0; i < BN_MAX; i++) { const uint64_t d = (uint64_t)a->v[i] - b->v[i] - br; a->v[i] = (uint32_t)d; br = d >> 63; } }
static int bn_bits(const bn *a) { for (int i = BN_MAX - 1; i >= 0; i--) if (a->v[i]) { int b = 32 * i; uint32_t x = a->v[i]; while (x) { b++; x >>= 1; } return b; } return 0; }
static void bn_shl1(bn *a) { uint32_t c = 0; for (int i = 0; i < BN_MAX; i++) { const uint32_t nc = a->v[i] >> 31; a->v[i] = a->v[i] << 1 | c; c = nc; } }
/* r = (a*b) mod m: 逐位从高到低 r = 2r (+a), 每步减模 —— 乘与模一步做, 不用双倍宽的中间积 */
static void bn_mulmod(bn *r, const bn *a, const bn *b, const bn *m) {
    bn t; memset(&t, 0, sizeof t);
    for (int i = bn_bits(b) - 1; i >= 0; i--) {
        bn_shl1(&t); if (bn_cmp(&t, m) >= 0) bn_sub(&t, m);
        if (b->v[i / 32] >> (i % 32) & 1) { uint64_t c = 0; for (int k = 0; k < BN_MAX; k++) { c += (uint64_t)t.v[k] + a->v[k]; t.v[k] = (uint32_t)c; c >>= 32; } if (bn_cmp(&t, m) >= 0) bn_sub(&t, m); }
    }
    *r = t;
}
static void bn_powmod(bn *r, const bn *base, uint32_t e, const bn *m) {
    bn acc; memset(&acc, 0, sizeof acc); acc.v[0] = 1; bn b = *base;
    for (; e; e >>= 1) { if (e & 1) bn_mulmod(&acc, &acc, &b, m); bn_mulmod(&b, &b, &b, m); }
    *r = acc;
}

/* ---- DER: SubjectPublicKeyInfo → n, e(只认 RSA, 结构固定: SEQ{SEQ{OID,NULL}, BITSTR{SEQ{INT n, INT e}}}) ---- */
static bool der_tl(const uint8_t **p, const uint8_t *end, uint8_t tag, size_t *len) {
    if (*p + 2 > end || **p != tag) return false;
    (*p)++; size_t l = **p; (*p)++;
    if (l & 0x80) { const int k = l & 0x7F; if (k > 4 || *p + k > end) return false; l = 0; for (int i = 0; i < k; i++) l = l << 8 | *(*p)++; }
    if (*p + l > end) return false;
    *len = l; return true;
}
static bool rsa_from_pem(const char *pem, uint8_t *n, size_t *nn, uint32_t *e) {
    const char *b = strstr(pem, "-----BEGIN"); if (!b) return false; b = strchr(b + 10, '\n'); if (!b) return false;
    const char *z = strstr(b, "-----END"); if (!z) return false;
    char *clean = malloc((size_t)(z - b) + 1); size_t k = 0;
    for (const char *c = b; c < z; c++) if (*c != '\n' && *c != '\r' && *c != ' ') clean[k++] = *c;
    clean[k] = 0;
    uint8_t der[1200]; size_t dn = 0;
    const bool ok = ds4_b64url_dec(clean, der, sizeof der, &dn); free(clean);
    if (!ok) return false;
    const uint8_t *p = der, *end = der + dn; size_t l;
    if (!der_tl(&p, end, 0x30, &l)) return false;
    if (!der_tl(&p, end, 0x30, &l)) return false; p += l;       /* 算法标识 */
    if (!der_tl(&p, end, 0x03, &l)) return false; p++;           /* BIT STRING, 跳过 unused-bits 字节 */
    if (!der_tl(&p, end, 0x30, &l)) return false;
    if (!der_tl(&p, end, 0x02, &l)) return false;
    while (l && !*p) { p++; l--; }                               /* 正数前导 0 */
    if (l > 512) return false;
    memcpy(n, p, l); *nn = l; p += l;
    if (!der_tl(&p, end, 0x02, &l) || l > 4) return false;
    *e = 0; for (size_t i = 0; i < l; i++) *e = *e << 8 | p[i];
    return true;
}

/* RSAES-OAEP(SHA-1, MGF1-SHA1, 空标签; MySQL 用 RSA_PKCS1_OAEP_PADDING 就是这组) 加密 msg → out(长度 = 模长) */
static void mgf1(const uint8_t *seed, size_t sn, uint8_t *mask, size_t mn) {   /* seed 可以是整个 DB(模长 - 21 字节), 不只是 20 字节的 seed */
    uint8_t buf[512 + 4], h[20];
    if (sn > 512) return;
    for (uint32_t c = 0; mn; c++) { memcpy(buf, seed, sn); buf[sn] = c >> 24; buf[sn + 1] = c >> 16; buf[sn + 2] = c >> 8; buf[sn + 3] = (uint8_t)c; mysql_sha1(buf, sn + 4, h); const size_t k = mn < 20 ? mn : 20; memcpy(mask, h, k); mask += k; mn -= k; }
}
bool mysql_rsa_oaep(const char *pem, const uint8_t *msg, size_t mn, uint8_t *out, size_t *on) {
    uint8_t n[512]; size_t nn; uint32_t e;
    if (!rsa_from_pem(pem, n, &nn, &e) || nn < 64 || mn + 42 > nn) return false;   /* 消息最长 k - 2hLen - 2 */
    uint8_t em[512]; memset(em, 0, nn);
    uint8_t *seed = em + 1, *db = em + 21; const size_t dbn = nn - 21;
    mysql_sha1("", 0, db);                                  /* lHash */
    db[dbn - mn - 1] = 1; memcpy(db + dbn - mn, msg, mn);  /* PS 全 0 ‖ 01 ‖ M */
    FILE *f = fopen("/dev/urandom", "rb"); const bool got = f && fread(seed, 1, 20, f) == 20; if (f) fclose(f);
    if (!got) return false;
    uint8_t mask[512]; mgf1(seed, 20, mask, dbn); for (size_t i = 0; i < dbn; i++) db[i] ^= mask[i];
    mgf1(db, dbn, mask, 20); for (int i = 0; i < 20; i++) seed[i] ^= mask[i];
    bn m, mod, c; bn_from(&m, em, nn); bn_from(&mod, n, nn);
    bn_powmod(&c, &m, e, &mod);
    for (size_t i = 0; i < nn; i++) out[nn - 1 - i] = (uint8_t)(c.v[i / 4] >> (8 * (i % 4)));
    *on = nn;
    return true;
}
/* 单测用: 小数验算 m^e mod n */
uint64_t mysql_bn_powmod_small(uint64_t m, uint32_t e, uint64_t n) {
    uint8_t mb[8], nb[8]; for (int i = 0; i < 8; i++) { mb[7 - i] = (uint8_t)(m >> (8 * i)); nb[7 - i] = (uint8_t)(n >> (8 * i)); }
    bn a, mod, r; bn_from(&a, mb, 8); bn_from(&mod, nb, 8); bn_powmod(&r, &a, e, &mod);
    return (uint64_t)r.v[1] << 32 | r.v[0];
}
