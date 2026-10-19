/* bson.c — BSON 编解码(接口见 bson.h)。金标: tests/asst/t_asst_db.c 用规范文档里的字节序列逐字节对。
 * 解码每一步都先查长度再读: 回包是网络上来的字节, 一个错的长度字段不能让我们读越界; 坏文档一律当"到此为止", 不 abort。
 * 浮点输出先试 %.15g 能否原样读回, 不能才 %.17g: 0.1 写成 0.10000000000000001 模型会照抄进回答。 */
#include "bson.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put32(ds4_buf *b, int32_t v) { uint8_t x[4]; for (int i = 0; i < 4; i++) x[i] = (uint8_t)((uint32_t)v >> (8 * i)); ds4_buf_append(b, x, 4); }
static void put64(ds4_buf *b, int64_t v) { uint8_t x[8]; for (int i = 0; i < 8; i++) x[i] = (uint8_t)((uint64_t)v >> (8 * i)); ds4_buf_append(b, x, 8); }
static int32_t get32(const uint8_t *p) { return (int32_t)((uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24); }
static int64_t get64(const uint8_t *p) { uint64_t v = 0; for (int i = 7; i >= 0; i--) v = v << 8 | p[i]; return (int64_t)v; }
static void elem(ds4_buf *b, uint8_t type, const char *key) { ds4_buf_putc(b, (char)type); ds4_buf_append(b, key, strlen(key) + 1); }

size_t bson_begin(ds4_buf *b) { const size_t s = b->len; put32(b, 0); return s; }
void bson_end(ds4_buf *b, size_t start) {
    ds4_buf_putc(b, 0);
    const int32_t n = (int32_t)(b->len - start);
    for (int i = 0; i < 4; i++) b->ptr[start + i] = (char)((uint32_t)n >> (8 * i));
}
size_t bson_sub(ds4_buf *b, const char *key, bool array) { elem(b, array ? 4 : 3, key); return bson_begin(b); }
void bson_int32(ds4_buf *b, const char *key, int32_t v) { elem(b, 0x10, key); put32(b, v); }
void bson_int64(ds4_buf *b, const char *key, int64_t v) { elem(b, 0x12, key); put64(b, v); }
void bson_double(ds4_buf *b, const char *key, double v) { elem(b, 1, key); uint64_t u; memcpy(&u, &v, 8); put64(b, (int64_t)u); }
void bson_str(ds4_buf *b, const char *key, const char *s) { elem(b, 2, key); put32(b, (int32_t)strlen(s) + 1); ds4_buf_append(b, s, strlen(s) + 1); }
void bson_bool(ds4_buf *b, const char *key, bool v) { elem(b, 8, key); ds4_buf_putc(b, v ? 1 : 0); }
void bson_null(ds4_buf *b, const char *key) { elem(b, 0x0A, key); }
void bson_bin(ds4_buf *b, const char *key, const void *p, size_t n) { elem(b, 5, key); put32(b, (int32_t)n); ds4_buf_putc(b, 0); ds4_buf_append(b, p, n); }
static void bson_date(ds4_buf *b, const char *key, int64_t ms) { elem(b, 9, key); put64(b, ms); }

/* ---- 日期(Howard Hinnant 的 civil 算法, 不碰 timegm/时区) ---- */
static int64_t days_from_civil(int y, int m, int d) {
    y -= m <= 2;
    const int era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = (unsigned)(y - era * 400), doy = (unsigned)((153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1), doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097LL + (int64_t)doe - 719468;
}
static void civil_from_days(int64_t z, int *y, int *m, int *d) {
    z += 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = (unsigned)(z - era * 146097), yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365, doy = doe - (365 * yoe + yoe / 4 - yoe / 100), mp = (5 * doy + 2) / 153;
    *d = (int)(doy - (153 * mp + 2) / 5 + 1); *m = (int)(mp < 10 ? mp + 3 : mp - 9); *y = (int)(yoe + era * 400 + (*m <= 2));
}
bool bson_iso_to_ms(const char *s, int64_t *ms) {
    int Y, M, D, h = 0, mi = 0, n = 0; double sec = 0;
    if (sscanf(s, "%4d-%2d-%2d%n", &Y, &M, &D, &n) < 3 || n < 8 || M < 1 || M > 12 || D < 1 || D > 31) return false;
    const char *p = s + n;
    if (*p == 'T' || *p == ' ') {
        p++; if (sscanf(p, "%2d:%2d%n", &h, &mi, &n) < 2) return false; p += n;
        if (*p == ':') { p++; char *e; sec = strtod(p, &e); if (e == p) return false; p = e; }
    }
    long off = 0;
    if (*p == 'Z') p++;
    else if (*p == '+' || *p == '-') {
        const int sg = *p == '-' ? -1 : 1; int oh = 0, om = 0; p++;
        if (sscanf(p, "%2d%n", &oh, &n) < 1) return false; p += n;
        if (*p == ':') p++;
        if (sscanf(p, "%2d%n", &om, &n) >= 1) p += n;
        off = sg * (oh * 3600L + om * 60L);
    }
    if (*p) return false;
    *ms = ((days_from_civil(Y, M, D) * 86400 + h * 3600 + mi * 60) - off) * 1000 + (int64_t)(sec * 1000 + 0.5);
    return true;
}
void bson_ms_to_iso(int64_t ms, char *out, size_t n) {
    int64_t s = ms / 1000, frac = ms % 1000; if (frac < 0) { frac += 1000; s--; }
    int64_t days = s / 86400, rem = s % 86400; if (rem < 0) { rem += 86400; days--; }
    int y, m, d; civil_from_days(days, &y, &m, &d);
    if (frac) snprintf(out, n, "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ", y, m, d, (int)(rem / 3600), (int)(rem % 3600 / 60), (int)(rem % 60), (int)frac);
    else snprintf(out, n, "%04d-%02d-%02dT%02d:%02d:%02dZ", y, m, d, (int)(rem / 3600), (int)(rem % 3600 / 60), (int)(rem % 60));
}

/* ---- JSON → BSON ---- */
static int hexv(char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; }
static bool fail(char *err, size_t errn, const char *what) { snprintf(err, errn, "%s", what); return false; }

/* {"$oid":…} / {"$date":…} / {"$numberLong":…}: 只认恰好一个 key 的对象; 认不出(格式不对)返回 false 且不消费, 调用方按普通文档编 */
static bool special(ds4_buf *b, const char *key, const char **json) {
    const char *p = *json; p++; json_ws(&p);
    char *k = NULL; if (*p != '"' || !json_string(&p, &k)) return false;
    json_ws(&p); if (*p != ':') { free(k); return false; }
    p++; json_ws(&p);
    bool ok = false;
    if (!strcmp(k, "$oid")) {
        char *s = NULL;
        if (json_string(&p, &s)) {
            uint8_t oid[12]; ok = strlen(s) == 24;
            for (int i = 0; ok && i < 12; i++) { const int hi = hexv(s[2 * i]), lo = hexv(s[2 * i + 1]); if (hi < 0 || lo < 0) ok = false; else oid[i] = (uint8_t)(hi * 16 + lo); }
            if (ok) { elem(b, 7, key); ds4_buf_append(b, oid, 12); }
            free(s);
        }
    } else if (!strcmp(k, "$date")) {
        int64_t ms = 0; double d;
        if (*p == '"') { char *s = NULL; if (json_string(&p, &s)) { ok = bson_iso_to_ms(s, &ms); free(s); } }
        else if (json_number(&p, &d)) { ms = (int64_t)d; ok = true; }
        if (ok) bson_date(b, key, ms);
    } else if (!strcmp(k, "$numberLong")) {
        char *s = NULL;
        if (json_string(&p, &s)) { char *e; const long long v = strtoll(s, &e, 10); ok = e != s && !*e; if (ok) bson_int64(b, key, v); free(s); }
    }
    free(k);
    if (!ok) return false;
    json_ws(&p);
    if (*p != '}') return false;   /* 元素已经写进 b 了, 但带别的 key 的 {"$date":…,"x":1} 不是扩展形式; 调用方整条放弃(err) */
    *json = p + 1;
    return true;
}

static bool jv(ds4_buf *b, const char *key, const char **json, int depth, char *err, size_t errn) {
    if (depth > BSON_MAX_DEPTH) return fail(err, errn, "JSON nested too deep");
    const char *p = *json; json_ws(&p);
    if (*p == '{') {
        const char *q = p + 1; json_ws(&q);
        if (*q == '"' && q[1] == '$') { const char *save = p; if (special(b, key, &p)) { *json = p; return true; } p = save; }
        const size_t s = bson_sub(b, key, false);
        p++; json_ws(&p);
        if (*p == '}') p++;
        else for (;;) {
            char *k = NULL; if (!json_string(&p, &k)) return fail(err, errn, "expected a key");
            json_ws(&p); if (*p != ':') { free(k); return fail(err, errn, "expected ':'"); }
            p++;
            const bool ok = jv(b, k, &p, depth + 1, err, errn); free(k);
            if (!ok) return false;
            json_ws(&p);
            if (*p == ',') { p++; json_ws(&p); continue; }
            if (*p == '}') { p++; break; }
            return fail(err, errn, "expected ',' or '}'");
        }
        bson_end(b, s); *json = p; return true;
    }
    if (*p == '[') {
        const size_t s = bson_sub(b, key, true);
        p++; json_ws(&p);
        if (*p == ']') p++;
        else for (int i = 0;; i++) {
            char ik[16]; snprintf(ik, sizeof ik, "%d", i);
            if (!jv(b, ik, &p, depth + 1, err, errn)) return false;
            json_ws(&p);
            if (*p == ',') { p++; continue; }
            if (*p == ']') { p++; break; }
            return fail(err, errn, "expected ',' or ']'");
        }
        bson_end(b, s); *json = p; return true;
    }
    if (*p == '"') { char *s = NULL; if (!json_string(&p, &s)) return fail(err, errn, "bad string"); bson_str(b, key, s); free(s); *json = p; return true; }
    if (*p == 't' || *p == 'f') { bool v; if (!json_bool(&p, &v)) return fail(err, errn, "bad literal"); bson_bool(b, key, v); *json = p; return true; }
    if (*p == 'n') { if (!json_lit(&p, "null")) return fail(err, errn, "bad literal"); bson_null(b, key); *json = p; return true; }
    double d; if (!json_number(&p, &d)) return fail(err, errn, "bad value");
    const bool integral = d == (double)(int64_t)d;
    if (integral && d >= -2147483648.0 && d <= 2147483647.0) bson_int32(b, key, (int32_t)d);
    else if (integral && d > -9007199254740992.0 && d < 9007199254740992.0) bson_int64(b, key, (int64_t)d);
    else bson_double(b, key, d);
    *json = p; return true;
}
bool bson_json_value(ds4_buf *b, const char *key, const char **json, char *err, size_t errn) { return jv(b, key, json, 0, err, errn); }

bool bson_json_fields(ds4_buf *b, const char *json, char *err, size_t errn) {
    const char *p = json; json_ws(&p);
    if (*p != '{') return fail(err, errn, "expected a JSON object");
    p++; json_ws(&p);
    if (*p == '}') return true;
    for (;;) {
        char *k = NULL; if (!json_string(&p, &k)) return fail(err, errn, "expected a key");
        json_ws(&p); if (*p != ':') { free(k); return fail(err, errn, "expected ':'"); }
        p++;
        const bool ok = jv(b, k, &p, 1, err, errn); free(k);
        if (!ok) return false;
        json_ws(&p);
        if (*p == ',') { p++; json_ws(&p); continue; }
        if (*p == '}') return true;
        return fail(err, errn, "expected ',' or '}'");
    }
}

/* ---- 解码 ---- */
size_t bson_doclen(const uint8_t *doc, size_t avail) {
    if (avail < 5) return 0;
    const int32_t n = get32(doc);
    return n >= 5 && (size_t)n <= avail && doc[n - 1] == 0 ? (size_t)n : 0;
}
bool bson_iter_init(bson_iter *it, const uint8_t *doc, size_t avail) {
    const size_t n = bson_doclen(doc, avail);
    if (!n) return false;
    it->p = doc + 4; it->end = doc + n - 1;
    return true;
}
/* 值占多少字节; 坏的(长度越界 / 字符串没结尾)给 false */
static bool vsize(uint8_t t, const uint8_t *p, const uint8_t *end, size_t *n) {
    const size_t avail = (size_t)(end - p);
    switch (t) {
    case 0x01: case 0x09: case 0x11: case 0x12: *n = 8; break;
    case 0x02: case 0x0D: case 0x0E: { if (avail < 4) return false; const int32_t l = get32(p); if (l < 1 || (size_t)l > avail - 4 || p[4 + l - 1]) return false; *n = 4 + (size_t)l; break; }
    case 0x03: case 0x04: case 0x0F: { if (avail < 5) return false; const int32_t l = get32(p); if (l < 5 || (size_t)l > avail) return false; *n = (size_t)l; break; }
    case 0x05: { if (avail < 5) return false; const int32_t l = get32(p); if (l < 0 || (size_t)l > avail - 5) return false; *n = 5 + (size_t)l; break; }
    case 0x06: case 0x0A: case 0xFF: case 0x7F: *n = 0; break;
    case 0x07: *n = 12; break;
    case 0x08: *n = 1; break;
    case 0x0B: { const uint8_t *a = memchr(p, 0, avail); if (!a) return false; const uint8_t *c = memchr(a + 1, 0, (size_t)(end - a - 1)); if (!c) return false; *n = (size_t)(c + 1 - p); break; }
    case 0x0C: { if (avail < 4) return false; const int32_t l = get32(p); if (l < 1 || (size_t)l > avail - 16) return false; *n = 4 + (size_t)l + 12; break; }
    case 0x10: *n = 4; break;
    case 0x13: *n = 16; break;
    default: return false;
    }
    return *n <= avail;
}
bool bson_next(bson_iter *it, uint8_t *type, const char **key, const uint8_t **val, size_t *vlen) {
    if (it->p >= it->end) return false;
    const uint8_t t = *it->p;
    if (!t) return false;
    const uint8_t *k = it->p + 1, *z = memchr(k, 0, (size_t)(it->end - k));
    if (!z) return false;
    size_t n;
    if (!vsize(t, z + 1, it->end, &n)) { it->p = it->end; return false; }
    *type = t; *key = (const char *)k; *val = z + 1; *vlen = n;
    it->p = z + 1 + n;
    return true;
}
bool bson_find(const uint8_t *doc, size_t avail, const char *key, uint8_t *type, const uint8_t **val, size_t *vlen) {
    bson_iter it; const char *k;
    if (!bson_iter_init(&it, doc, avail)) return false;
    while (bson_next(&it, type, &k, val, vlen)) if (!strcmp(k, key)) return true;
    return false;
}
int64_t bson_num(uint8_t type, const uint8_t *val) {
    switch (type) { case 0x10: return get32(val); case 0x12: case 0x09: return get64(val); case 0x08: return val[0]; case 0x01: return (int64_t)bson_dbl(type, val); default: return 0; }
}
double bson_dbl(uint8_t type, const uint8_t *val) {
    if (type == 0x01) { const uint64_t u = (uint64_t)get64(val); double d; memcpy(&d, &u, 8); return d; }
    return (double)bson_num(type, val);
}
const char *bson_string(uint8_t type, const uint8_t *val) { return type == 0x02 ? (const char *)val + 4 : NULL; }

static void hex(ds4_buf *o, const uint8_t *p, size_t n) { static const char H[] = "0123456789abcdef"; for (size_t i = 0; i < n; i++) { ds4_buf_putc(o, H[p[i] >> 4]); ds4_buf_putc(o, H[p[i] & 15]); } }
static void dbl_json(ds4_buf *o, double d) {
    uint64_t u; memcpy(&u, &d, 8);
    if (((u >> 52) & 0x7FF) == 0x7FF) { ds4_buf_puts(o, "null"); return; }   /* NaN/Inf 不是 JSON; 位判断不走 isnan(全仓 -ffast-math 下会被折掉) */
    char s[32]; snprintf(s, sizeof s, "%.15g", d);
    if (strtod(s, NULL) != d) snprintf(s, sizeof s, "%.17g", d);
    if (!strpbrk(s, ".eE")) { const size_t n = strlen(s); if (n + 2 < sizeof s) memcpy(s + n, ".0", 3); }   /* 保留"这是个浮点"的信息: 2.0 写成 2 模型会当整数 */
    ds4_buf_puts(o, s);
}
static void doc_json(const uint8_t *doc, size_t avail, ds4_buf *o, bool array, int depth);
static void val_json(uint8_t t, const uint8_t *v, size_t n, ds4_buf *o, int depth) {
    char tmp[64];
    switch (t) {
    case 0x01: dbl_json(o, bson_dbl(t, v)); break;
    case 0x02: case 0x0E: ds4_json_escape(o, (const char *)v + 4); break;
    case 0x0D: ds4_buf_puts(o, "{\"$code\":"); ds4_json_escape(o, (const char *)v + 4); ds4_buf_putc(o, '}'); break;
    case 0x03: doc_json(v, n, o, false, depth + 1); break;
    case 0x04: doc_json(v, n, o, true, depth + 1); break;
    case 0x05: { const int32_t l = get32(v); ds4_buf_printf(o, "{\"$binary\":{\"bytes\":%d,\"subtype\":%d,\"hex\":\"", l, v[4]); hex(o, v + 5, l < 32 ? (size_t)l : 32); ds4_buf_puts(o, l > 32 ? "…\"}}" : "\"}}"); break; }
    case 0x07: ds4_buf_puts(o, "{\"$oid\":\""); hex(o, v, 12); ds4_buf_puts(o, "\"}"); break;
    case 0x08: ds4_buf_puts(o, v[0] ? "true" : "false"); break;
    case 0x09: bson_ms_to_iso(get64(v), tmp, sizeof tmp); ds4_buf_printf(o, "{\"$date\":\"%s\"}", tmp); break;
    case 0x0B: { const char *re = (const char *)v, *opt = re + strlen(re) + 1; ds4_buf_puts(o, "{\"$regex\":"); ds4_json_escape(o, re); ds4_buf_puts(o, ",\"$options\":"); ds4_json_escape(o, opt); ds4_buf_putc(o, '}'); break; }
    case 0x10: ds4_buf_printf(o, "%d", get32(v)); break;
    case 0x11: ds4_buf_printf(o, "{\"$timestamp\":{\"t\":%u,\"i\":%u}}", (unsigned)get32(v + 4), (unsigned)get32(v)); break;
    case 0x12: ds4_buf_printf(o, "%lld", (long long)get64(v)); break;
    case 0x13: ds4_buf_puts(o, "{\"$numberDecimal\":\"0x"); hex(o, v, 16); ds4_buf_puts(o, "\"}"); break;   /* decimal128 不展开(没见过谁用它存; 要用再补) */
    case 0xFF: ds4_buf_puts(o, "{\"$minKey\":1}"); break;
    case 0x7F: ds4_buf_puts(o, "{\"$maxKey\":1}"); break;
    default: ds4_buf_puts(o, "null"); break;   /* null / undefined / dbpointer / code_w_scope */
    }
}
static void doc_json(const uint8_t *doc, size_t avail, ds4_buf *o, bool array, int depth) {
    bson_iter it; uint8_t t; const char *k; const uint8_t *v; size_t n; int i = 0;
    ds4_buf_putc(o, array ? '[' : '{');
    if (depth <= BSON_MAX_DEPTH && bson_iter_init(&it, doc, avail)) while (bson_next(&it, &t, &k, &v, &n)) {
        if (i++) ds4_buf_putc(o, ',');
        if (!array) { ds4_json_escape(o, k); ds4_buf_putc(o, ':'); }
        val_json(t, v, n, o, depth);
    }
    ds4_buf_putc(o, array ? ']' : '}');
}
void bson_to_json(const uint8_t *doc, size_t avail, ds4_buf *out) { doc_json(doc, avail, out, false, 0); }
