/* t_asst_db.c — 连接器与 src/db 的离线单测(方案稿 docs/assistant_connectors.md §8)。
 * 金标: BSON 用 bsonspec.org 规范文档里的两条字节序列; SCRAM-SHA-256 用 RFC 7677 §3 的完整向量; SHA-1 用 FIPS 180 的 "abc"; RSA 大数用手算的小数。
 * 真库不在这里(要本机有库): ./ds4_asst_test --tool query_db '{"connector":"quant","what":"collections"}'; 末尾那段 live 只在 127.0.0.1:27017 有 Mongo 时跑, 否则 SKIP。
 * 由 t_asst_sbx.c 的 main 调, 在 asst_test_logic 之后(助理根目录已建好)。 */
#include "../../src/assistant/asst_internal.h"
#include "../../src/db/db.h"
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

static int pass, fail;
#define CHECK(c, ...) do { if (c) pass++; else { fail++; printf("  FAIL db: "); printf(__VA_ARGS__); printf(" (%s:%d)\n", __FILE__, __LINE__); } } while (0)

static bool bytes_eq(const ds4_buf *b, const uint8_t *want, size_t n) { return b->len == n && !memcmp(b->ptr, want, n); }

static void test_bson(void) {
    /* 规范示例 1: {"hello": "world"} */
    static const uint8_t ex1[] = { 0x16,0,0,0, 0x02, 'h','e','l','l','o',0, 0x06,0,0,0, 'w','o','r','l','d',0, 0 };
    ds4_buf b = {0}; size_t s = bson_begin(&b); bson_str(&b, "hello", "world"); bson_end(&b, s);
    CHECK(bytes_eq(&b, ex1, sizeof ex1), "spec example 1 bytes"); ds4_buf_free(&b);
    /* 规范示例 2: {"BSON": ["awesome", 5.05, 1986]} —— 经 JSON 编进去 */
    static const uint8_t ex2[] = { 0x31,0,0,0, 0x04,'B','S','O','N',0, 0x26,0,0,0, 0x02,'0',0, 0x08,0,0,0,'a','w','e','s','o','m','e',0, 0x01,'1',0, 0x33,0x33,0x33,0x33,0x33,0x33,0x14,0x40, 0x10,'2',0, 0xc2,0x07,0,0, 0, 0 };
    char err[128]; const char *j = "[\"awesome\", 5.05, 1986]";
    s = bson_begin(&b); CHECK(bson_json_value(&b, "BSON", &j, err, sizeof err), "json array encodes: %s", err); bson_end(&b, s);
    CHECK(bytes_eq(&b, ex2, sizeof ex2), "spec example 2 bytes (len %zu)", b.len);
    ds4_buf out = {0}; bson_to_json((const uint8_t *)b.ptr, b.len, &out);
    CHECK(!strcmp(out.ptr, "{\"BSON\":[\"awesome\",5.05,1986]}"), "decode back: %s", out.ptr); ds4_buf_free(&out); ds4_buf_free(&b);
    /* 往返: 扩展形式与各类型 */
    const char *doc = "{\"_id\":{\"$oid\":\"66f1a2b3c4d5e6f708192a3b\"},\"when\":{\"$date\":\"2026-10-18T09:30:00+08:00\"},\"big\":{\"$numberLong\":\"9007199254740993\"},\"n\":-7,\"f\":2.0,\"ok\":true,\"none\":null,\"s\":\"中文 \\\"q\\\"\",\"sub\":{\"a\":[1,[2,3]],\"b\":{}}}";
    s = bson_begin(&b); CHECK(bson_json_fields(&b, doc, err, sizeof err), "fields encode: %s", err); bson_end(&b, s);
    bson_to_json((const uint8_t *)b.ptr, b.len, &out);
    CHECK(!strcmp(out.ptr, "{\"_id\":{\"$oid\":\"66f1a2b3c4d5e6f708192a3b\"},\"when\":{\"$date\":\"2026-10-18T01:30:00Z\"},\"big\":9007199254740993,\"n\":-7,\"f\":2,\"ok\":true,\"none\":null,\"s\":\"中文 \\\"q\\\"\",\"sub\":{\"a\":[1,[2,3]],\"b\":{}}}"), "round trip (2.0 becomes int 2 on purpose): %s", out.ptr);
    uint8_t t; const uint8_t *v; size_t vn;
    CHECK(bson_find((const uint8_t *)b.ptr, b.len, "n", &t, &v, &vn) && t == 0x10 && bson_num(t, v) == -7, "find int32");
    CHECK(bson_find((const uint8_t *)b.ptr, b.len, "big", &t, &v, &vn) && t == 0x12 && bson_num(t, v) == 9007199254740993LL, "int64 via $numberLong");
    CHECK(bson_find((const uint8_t *)b.ptr, b.len, "when", &t, &v, &vn) && t == 0x09, "date type");
    ds4_buf_free(&out);
    /* 坏文档: 长度字段超出 */
    uint8_t bad[8] = { 0x40, 0, 0, 0, 0x10, 'a', 0, 0 }; bson_iter it;
    CHECK(!bson_iter_init(&it, bad, sizeof bad) && !bson_doclen(bad, sizeof bad), "oversized length rejected");
    b.ptr[b.len - 1] = 1; CHECK(!bson_doclen((const uint8_t *)b.ptr, b.len), "missing terminator rejected"); ds4_buf_free(&b);
    /* 扩展形式带多余 key / 坏 oid → 当普通文档或报错 */
    j = "{\"$oid\":\"zz\"}"; s = bson_begin(&b); CHECK(bson_json_value(&b, "x", &j, err, sizeof err), "bad $oid falls back to a plain document"); bson_end(&b, s); ds4_buf_free(&b);
    j = "{\"a\":"; s = bson_begin(&b); CHECK(!bson_json_value(&b, "x", &j, err, sizeof err) && err[0], "truncated JSON reports: %s", err); ds4_buf_free(&b);
    int64_t ms; CHECK(bson_iso_to_ms("2026-10-18", &ms) && ms == 1792281600000LL, "date only → midnight UTC (%lld)", (long long)ms);
    CHECK(bson_iso_to_ms("1970-01-01T00:00:01.5Z", &ms) && ms == 1500, "fraction seconds");
    CHECK(!bson_iso_to_ms("18/10/2026", &ms) && !bson_iso_to_ms("2026-13-01", &ms), "bad dates rejected");
    char iso[40]; bson_ms_to_iso(-1000, iso, sizeof iso); CHECK(!strcmp(iso, "1969-12-31T23:59:59Z"), "negative ms: %s", iso);
}

static void test_scram(void) {   /* RFC 7677 §3 */
    char first[256]; CHECK(scram_client_first("user", "rOprNGfwEbeRWgbNEkqO", first, sizeof first) && !strcmp(first, "n,,n=user,r=rOprNGfwEbeRWgbNEkqO"), "client-first: %s", first);
    char fin[512]; uint8_t sig[32]; char err[128];
    CHECK(scram_client_final("user", "pencil", "rOprNGfwEbeRWgbNEkqO", "r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0,s=W22ZaJ0SNY7soEsUEjb6gQ==,i=4096", fin, sizeof fin, sig, err, sizeof err), "client-final computes: %s", err);
    CHECK(!strcmp(fin, "c=biws,r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0,p=dHzbZapWIk4jUhN+Ute9ytag9zjfMHgsqmmiz7AndVQ="), "client-final proof: %s", fin);
    CHECK(scram_server_ok(sig, "v=6rriTRBi23WpRR/wtup+mMhUZUn/dB5nLTJRsjl95G4="), "server signature verifies");
    CHECK(!scram_server_ok(sig, "v=6rriTRBi23WpRS/wtup+mMhUZUn/dB5nLTJRsjl95G4="), "wrong server signature rejected");
    CHECK(!scram_client_final("user", "pencil", "rOprNGfwEbeRWgbNEkqO", "r=XXXX,s=W22ZaJ0SNY7soEsUEjb6gQ==,i=4096", fin, sizeof fin, sig, err, sizeof err) && strstr(err, "nonce"), "foreign nonce rejected: %s", err);
    CHECK(scram_client_first("a,b=c", "n", first, sizeof first) && !strcmp(first, "n,,n=a=2Cb=3Dc,r=n"), "user name escaping: %s", first);
    char b64[16]; scram_b64((const uint8_t *)"ab", 2, b64); CHECK(!strcmp(b64, "YWI="), "base64 padding: %s", b64);
}

static void test_mysql_auth(void) {
    uint8_t h[20]; mysql_sha1("abc", 3, h);
    static const uint8_t want[20] = { 0xa9,0x99,0x3e,0x36,0x47,0x06,0x81,0x6a,0xba,0x3e,0x25,0x71,0x78,0x50,0xc2,0x6c,0x9c,0xd0,0xd8,0x9d };
    CHECK(!memcmp(h, want, 20), "sha1(abc)");
    char longmsg[1000]; memset(longmsg, 'a', sizeof longmsg); mysql_sha1(longmsg, 1000, h);   /* 跨块 + 两块填充 */
    static const uint8_t want2[20] = { 0x29,0x1e,0x9a,0x6c,0x66,0x99,0x49,0x49,0xb5,0x7b,0xa5,0xe6,0x50,0x36,0x1e,0x98,0xfc,0x36,0xb1,0xba };
    CHECK(!memcmp(h, want2, 20), "sha1(1000 x a)");
    uint8_t nonce[20]; for (int i = 0; i < 20; i++) nonce[i] = (uint8_t)i;
    uint8_t s1[20], s2[20]; mysql_scramble_native("secret", nonce, s1); mysql_scramble_native("secret", nonce, s2);
    uint8_t ck[20], ck2[20], buf[40]; mysql_sha1("secret", 6, ck); mysql_sha1(ck, 20, ck2); memcpy(buf, nonce, 20); memcpy(buf + 20, ck2, 20); mysql_sha1(buf, 40, h);
    for (int i = 0; i < 20; i++) h[i] ^= ck[i];
    CHECK(!memcmp(s1, s2, 20) && !memcmp(s1, h, 20), "native scramble = SHA1(pw) XOR SHA1(nonce ‖ SHA1(SHA1(pw)))");
    uint8_t s3[32]; mysql_scramble_sha2("secret", nonce, s3); CHECK(memcmp(s3, s1, 20) != 0, "sha2 scramble differs");
    CHECK(mysql_bn_powmod_small(4, 13, 497) == 445, "4^13 mod 497 = 445");
    /* 大数对小模: 2^65537 mod 1000000007 手算(python pow) = 947173645 */
    CHECK(mysql_bn_powmod_small(2, 65537, 1000000007ULL) == 947173645ULL, "2^65537 mod 1e9+7 (%llu)", (unsigned long long)mysql_bn_powmod_small(2, 65537, 1000000007ULL));
    /* 一把真公钥(2048 位, 测试专用)能解出来并加密出模长的密文 */
    const char *pem = "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAu1SU1LfVLPHCozMxH2Mo\n4lgOEePzNm0tRgeLezV6ffAt0gunVTLw7onLRnrq0/IzW7yWR7QkrmBL7jTKEn5u\n+qKhbwKfBstIs+bMY2Zkp18gnTxKLxoS2tFczGkPLPgizskuemMghRniWaoLcyeh\nkd3qqGElvW/VDL5AaWTg0nLVkjRo9z+40RQzuVaE8AkAFmxZzow3x+VJYKdjykkJ\n0iT9wCS0DRTXu269V264Vf/3jvredZiKRkgwlL9xNAwxXFg0x/XFw005UWVRIkdg\ncKWTjpBP2dPwVZ4WWC+9aGVd+Gyn1o0CLelf4rEjGoXbAAEgAqeGUxrcIlbjXfbc\nmwIDAQAB\n-----END PUBLIC KEY-----\n";
    uint8_t enc[512]; size_t en = 0;
    CHECK(mysql_rsa_oaep(pem, (const uint8_t *)"pw\0", 3, enc, &en) && en == 256, "RSA-OAEP with a 2048-bit key gives 256 bytes (%zu)", en);
    uint8_t enc2[512]; size_t en2; mysql_rsa_oaep(pem, (const uint8_t *)"pw\0", 3, enc2, &en2);
    CHECK(memcmp(enc, enc2, 256) != 0, "OAEP is randomised");
}

static void test_guards(void) {
    CHECK(!sql_readonly_check("SELECT * FROM t") && !sql_readonly_check("  -- note\n with x as (select 1) select * from x") && !sql_readonly_check("SHOW TABLES") && !sql_readonly_check("EXPLAIN SELECT 1") && !sql_readonly_check("(SELECT 1) UNION (SELECT 2)"), "read-only statements pass");
    CHECK(sql_readonly_check("DELETE FROM t") && sql_readonly_check("INSERT INTO t VALUES (1)") && sql_readonly_check("UPDATE t SET a=1") && sql_readonly_check("DROP TABLE t") && sql_readonly_check("CREATE TABLE t(a)"), "writes rejected");
    CHECK(sql_readonly_check("SELECT 1; DROP TABLE t"), "two statements rejected");
    CHECK(!sql_readonly_check("SELECT 1;") && !sql_readonly_check("SELECT ';' AS s; "), "trailing ';' and ';' in a string are fine");
    CHECK(sql_readonly_check("WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d") && sql_readonly_check("EXPLAIN ANALYZE UPDATE t SET a = 1"), "writing CTE / EXPLAIN ANALYZE rejected");
    CHECK(!sql_readonly_check("SELECT is_delete, update_time FROM t WHERE note = 'delete me'"), "words inside identifiers and strings are not statements");
    CHECK(sql_readonly_check("") && sql_readonly_check("   "), "empty rejected");
    char *w = sql_wrap_limit("SELECT a FROM t ;", 7, "mysql"); CHECK(!strcmp(w, "SELECT * FROM (SELECT a FROM t) AS q LIMIT 7"), "wrap: %s", w); free(w);
    w = sql_wrap_limit("SHOW TABLES", 7, "mysql"); CHECK(!strcmp(w, "SHOW TABLES"), "SHOW not wrapped: %s", w); free(w);
    CHECK(!mongo_forbidden("{\"a\":{\"$gt\":1},\"note\":\"$out\"}") && !mongo_forbidden(NULL) && !mongo_forbidden("[{\"$match\":{}},{\"$group\":{\"_id\":\"$sym\"}}]"), "normal filters/pipelines pass");
    const char *bad = mongo_forbidden("[{\"$match\":{}},{\"$out\":\"x\"}]"); CHECK(bad && !strcmp(bad, "$out"), "$out stage caught: %s", bad ? bad : "(null)");
    bad = mongo_forbidden("{\"x\":{\"$in\":[1]},\"$where\":\"this.a > 1\"}"); CHECK(bad && !strcmp(bad, "$where"), "$where caught");
    bad = mongo_forbidden("[{\"$group\":{\"_id\":null,\"v\":{\"$accumulator\":{}}}}]"); CHECK(bad && !strcmp(bad, "$accumulator"), "nested $accumulator caught");
    const db_kind *k = db_kind_find("mysql"); CHECK(k && k->port == 3306 && strstr(k->whats, "sql") && !db_kind_find("oracle"), "kinds table");
}

static void test_store(void) {
    char err[256]; ds4_buf b = {0};
    asst_tctx ctx = { "", "", NULL, false };
    CHECK(!asst_conn_build_tool(&ctx, &b), "no connectors → no query_db"); ds4_buf_free(&b);
    CHECK(!asst_conn_op("{\"op\":\"save\",\"name\":\"Bad Name\",\"kind\":\"mongodb\",\"host\":\"x\"}", err, sizeof err) && strstr(err, "lowercase"), "name rule: %s", err);
    CHECK(!asst_conn_op("{\"op\":\"save\",\"name\":\"q\",\"kind\":\"oracle\",\"host\":\"x\"}", err, sizeof err) && strstr(err, "kind"), "unknown kind: %s", err);
    CHECK(!asst_conn_op("{\"op\":\"save\",\"name\":\"q\",\"kind\":\"mongodb\"}", err, sizeof err) && strstr(err, "host"), "host required: %s", err);
    CHECK(asst_conn_op("{\"op\":\"save\",\"name\":\"quant-x\",\"kind\":\"mongodb\",\"host\":\"127.0.0.1\",\"db\":\"quant_trading\",\"user\":\"cfo\",\"password\":\"s3cret\"}", err, sizeof err), "save: %s", err);
    asst_conn c; CHECK(asst_conn_load("quant-x", &c) && c.port == 27017 && c.enabled && !strcmp(c.db, "quant_trading") && !strcmp(c.auth_db, ""), "loaded with default port");
    char pw[257]; CHECK(asst_conn_password("quant-x", pw, sizeof pw) && !strcmp(pw, "s3cret"), "password in secrets/conn_quant_x");
    char sp[ASST_PATH + 64]; snprintf(sp, sizeof sp, "%s/secrets/conn_quant_x", asst_root); struct stat st; CHECK(!stat(sp, &st) && (st.st_mode & 0777) == 0600, "secret file 0600");
    asst_conns_json(&b);
    CHECK(strstr(b.ptr, "\"name\":\"quant-x\"") && strstr(b.ptr, "\"has_password\":true") && !strstr(b.ptr, "s3cret") && !strstr(b.ptr, "\"password\"") && strstr(b.ptr, "\"kinds\":[{\"kind\":\"mongodb\"") && strstr(b.ptr, "\"label\":\"MySQL / MariaDB\""), "list has no password, has kinds: %s", b.ptr); ds4_buf_free(&b);
    CHECK(asst_conn_op("{\"op\":\"save\",\"name\":\"quant-x\",\"kind\":\"mongodb\",\"host\":\"192.168.2.203\",\"db\":\"quant_trading\",\"user\":\"cfo\"}", err, sizeof err) && asst_conn_password("quant-x", pw, sizeof pw) && !strcmp(pw, "s3cret"), "save without password keeps the old one");
    CHECK(asst_conn_build_tool(&ctx, &b) && strstr(b.ptr, "\"name\":\"query_db\"") && strstr(b.ptr, "\"enum\":[\"quant-x\"]") && strstr(b.ptr, "quant-x: MongoDB at 192.168.2.203:27017, database quant_trading") && strstr(b.ptr, "\"collections\",\"fields\",\"find\"") && !strstr(b.ptr, "\"tables\"") && !strstr(b.ptr, "s3cret"), "tool schema: %.300s", b.ptr); ds4_buf_free(&b);
    char *tools = asst_tools_json(&ctx); CHECK(strstr(tools, "\"query_db\""), "query_db in the tool list"); free(tools);
    CHECK(asst_conn_op("{\"op\":\"enable\",\"name\":\"quant-x\",\"on\":false}", err, sizeof err) && !asst_conn_enabled("quant-x"), "switch off");
    CHECK(!asst_conn_build_tool(&ctx, &b), "all off → no query_db"); ds4_buf_free(&b);
    int ok = 1; char *r = asst_db_exec("s", "{\"connector\":\"quant-x\",\"what\":\"collections\"}", &ok);
    CHECK(!ok && strstr(r, "turned off"), "off connector refuses: %s", r); free(r);
    r = asst_db_exec("s", "{\"connector\":\"nope\",\"what\":\"collections\"}", &ok); CHECK(!ok && strstr(r, "no such connector"), "unknown connector"); free(r);
    CHECK(asst_conn_op("{\"op\":\"enable\",\"name\":\"quant-x\",\"on\":true}", err, sizeof err) && asst_conn_enabled("quant-x"), "switch on");
    r = asst_db_exec("s", "{\"connector\":\"quant-x\",\"what\":\"sql\",\"sql\":\"SELECT 1\"}", &ok); CHECK(!ok && strstr(r, "what must be one of: collections"), "what checked per kind: %s", r); free(r);
    r = asst_db_exec("s", "{\"connector\":\"quant-x\",\"what\":\"find\",\"collection\":\"t\",\"pipeline\":[{\"$out\":\"x\"}]}", &ok); CHECK(!ok && strstr(r, "$out is not allowed"), "forbidden caught before connecting: %s", r); free(r);
    /* 守门: 放行, cls=db; 执行后置 taint_priv */
    char sid[ASST_ID_LEN + 1]; asst_session_new(sid, "db", "hi", 0);
    asst_gres g; asst_guard(sid, "query_db", "{\"connector\":\"quant-x\",\"what\":\"collections\"}", &g);
    CHECK(g.dec == ASST_ALLOW && !strcmp(g.cls, "db") && !strcmp(g.target, "quant-x"), "guard allows, cls db (%d %s)", g.dec, g.cls);
    char *val = NULL; CHECK(!asst_meta_get(sid, "taint_priv", &val) || !val || !strstr(val, "true"), "no taint before (%s)", val ? val : "-"); free(val); val = NULL;
    asst_guard_mark(sid, "db", NULL);
    CHECK(asst_meta_get(sid, "taint_priv", &val) && val && strstr(val, "true"), "taint_priv after reading the database"); free(val);
    asst_guard(sid, "web_search", "{\"query\":\"x\"}", &g); CHECK(g.dec == ASST_ASK && g.once_only, "going online afterwards asks (once only)");
    /* sqlite 连接器: path 必填并解析 */
    CHECK(!asst_conn_op("{\"op\":\"save\",\"name\":\"lite\",\"kind\":\"sqlite\"}", err, sizeof err) && strstr(err, "path"), "sqlite needs path: %s", err);
    CHECK(asst_conn_op("{\"op\":\"save\",\"name\":\"lite\",\"kind\":\"sqlite\",\"path\":\"data/x.db\"}", err, sizeof err) && asst_conn_load("lite", &c) && c.path[0] == '/' && strstr(c.path, "/data/x.db"), "sqlite path resolved: %s", c.path);
    CHECK(asst_conn_build_tool(&ctx, &b) && strstr(b.ptr, "lite: SQLite file, file ") && strstr(b.ptr, "\"tables\"") && strstr(b.ptr, "\"collections\""), "two kinds → what union"); ds4_buf_free(&b);
    /* 删除: json 进 trash, 密码没了 */
    CHECK(asst_conn_op("{\"op\":\"delete\",\"name\":\"quant-x\"}", err, sizeof err) && !asst_conn_load("quant-x", &c) && !asst_conn_password("quant-x", pw, sizeof pw), "delete removes json and secret");
    char tp[ASST_PATH + 64]; snprintf(tp, sizeof tp, "%s/trash/connectors", asst_root); int n = 0; { DIR *d = opendir(tp); struct dirent *e; while (d && (e = readdir(d))) if (strstr(e->d_name, "quant-x-")) n++; if (d) closedir(d); }
    CHECK(n == 1, "json in trash/connectors (%d)", n);
    CHECK(asst_conn_op("{\"op\":\"delete\",\"name\":\"lite\"}", err, sizeof err), "cleanup");
    memset(pw, 0, sizeof pw);
}

/* 本机 127.0.0.1:27017 有 Mongo 才跑: 握手 + listCollections; 没有就 SKIP(不 FAIL) */
static void test_live(void) {
    char err[256]; const int fd = db_tcp_connect("127.0.0.1", 27017, err, sizeof err);
    if (fd < 0) { printf("  SKIP db live: %s\n", err); return; }
    close(fd);
    const db_target t = { "mongodb", "127.0.0.1", "admin", "", "", "", "", 27017 };
    char line[256]; const bool ok = db_probe(&t, line, sizeof line, err, sizeof err);
    CHECK(ok && strstr(line, "MongoDB "), "live probe: %s", ok ? line : err);
    const db_req r = { .what = "collections", .limit = 5 }; bool qok = false;
    char *o = db_query(&t, &r, &qok); CHECK(qok && strstr(o, "collections in admin"), "live collections: %.120s", o); free(o);
    const db_req r2 = { .what = "aggregate", .coll = "system.version", .pipeline = "[{\"$out\":\"x\"}]", .limit = 5 };
    o = db_query(&t, &r2, &qok); CHECK(!qok && strstr(o, "$out is not allowed"), "live $out refused"); free(o);
    const db_target bad = { "mongodb", "127.0.0.1", "admin", "nobody", "wrong", "admin", "", 27017 };
    CHECK(!db_probe(&bad, line, sizeof line, err, sizeof err) && strstr(err, "authentication failed"), "live wrong user: %s", err);
}

int asst_test_db(const char *base) {
    (void)base; pass = fail = 0;
    test_bson(); test_scram(); test_mysql_auth(); test_guards(); test_store(); test_live();
    printf("asst db: %d passed, %d failed\n", pass, fail);
    return fail;
}
