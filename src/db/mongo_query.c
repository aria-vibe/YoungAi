/* mongo_query.c — 只读查询(mongo.h): collections / fields / find / count / distinct / aggregate 六种 what, 加 getMore 游标与 killCursors。
 * ★只读是这里保证的: 文件里只有这几个读命令的代码, 没有 insert/update/delete/drop —— 模型想写也没有手。
 * 发请求前扫模型给的 JSON 键: $out / $merge 会写库, $where / $function / $accumulator 在服务器上跑 JavaScript, 都拒。
 * 每条命令带 maxTimeMS: 一条烂的 aggregate 不能把交易 agent 的库卡住。
 * 结果: 第一行摘要, 之后一行一条 JSON(扩展形式, bson.h); 调用方(tool_db.c)负责截给模型与落盘。 */
#include "db.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIELDS_SAMPLE 20   /* fields 抽样条数: 够看出键名与类型, 又不至于把大文档拉一堆 */
#define FIELDS_MAX 200     /* 一个集合顶层键的上限 */
#define SAMPLE_CHARS 80    /* 样例值截到这么长 */

static const char *FORBIDDEN[] = { "$out", "$merge", "$where", "$function", "$accumulator" };

/* 递归走 JSON 文本, 只看对象的键(值里的字符串不算: {"note":"$out"} 是数据) */
static const char *scan(const char **p, int depth) {
    json_ws(p);
    if (depth > BSON_MAX_DEPTH) return "nested too deep";
    if (**p == '{') {
        (*p)++; json_ws(p);
        if (**p == '}') { (*p)++; return NULL; }
        for (;;) {
            char *k = NULL; if (!json_string(p, &k)) return NULL;
            for (size_t i = 0; i < sizeof FORBIDDEN / sizeof FORBIDDEN[0]; i++) if (!strcmp(k, FORBIDDEN[i])) { free(k); return FORBIDDEN[i]; }
            free(k);
            json_ws(p); if (**p != ':') return NULL; (*p)++;
            const char *hit = scan(p, depth + 1); if (hit) return hit;
            json_ws(p);
            if (**p == ',') { (*p)++; continue; }
            if (**p == '}') { (*p)++; return NULL; }
            return NULL;
        }
    }
    if (**p == '[') {
        (*p)++; json_ws(p);
        if (**p == ']') { (*p)++; return NULL; }
        for (;;) {
            const char *hit = scan(p, depth + 1); if (hit) return hit;
            json_ws(p);
            if (**p == ',') { (*p)++; continue; }
            if (**p == ']') { (*p)++; return NULL; }
            return NULL;
        }
    }
    json_skip_value(p);
    return NULL;
}
const char *mongo_forbidden(const char *json) { if (!json) return NULL; const char *p = json; return scan(&p, 0); }

typedef struct { mongo_conn c; const mongo_target *t; ds4_buf out; char err[400]; } qctx;

static bool add_json(ds4_buf *cmd, const char *key, const char *json, bool array, qctx *q) {   /* 模型给的 JSON 原文编成 key 下的文档/数组; 没给就不加 */
    if (!json || !json[0]) return true;
    const char *p = json; json_ws(&p);
    if ((array && *p != '[') || (!array && *p != '{')) { snprintf(q->err, sizeof q->err, "%s must be a JSON %s", key, array ? "array" : "object"); return false; }
    char why[128];
    if (!bson_json_value(cmd, key, &p, why, sizeof why)) { snprintf(q->err, sizeof q->err, "%s: %s", key, why); return false; }
    return true;
}

/* 跑一条命令, 回 reply(调用方 free); 失败把 c.err 搬进 q->err */
static bool cmd_run(qctx *q, ds4_buf *cmd, uint8_t **rep, size_t *rn) {
    const bool sent = mongo_command(&q->c, (const uint8_t *)cmd->ptr, cmd->len, rep, rn);
    ds4_buf_free(cmd);
    if (sent && mongo_ok(&q->c, *rep, *rn)) return true;
    if (sent) { free(*rep); *rep = NULL; }
    snprintf(q->err, sizeof q->err, "%s", q->c.err);
    return false;
}

/* cursor 回复: 取 id 与 firstBatch/nextBatch, 每条文档交给 each; 返回这批的条数 */
typedef void (*doc_fn)(qctx *q, const uint8_t *doc, size_t n, void *arg);
static long batch(qctx *q, const uint8_t *rep, size_t rn, int64_t *cursor_id, long max, doc_fn each, void *arg) {
    uint8_t t; const uint8_t *cur, *arr; size_t cn, an;
    if (!bson_find(rep, rn, "cursor", &t, &cur, &cn) || t != 3) return 0;
    if (bson_find(cur, cn, "id", &t, &arr, &an)) *cursor_id = bson_num(t, arr);
    if (!bson_find(cur, cn, "firstBatch", &t, &arr, &an) && !bson_find(cur, cn, "nextBatch", &t, &arr, &an)) return 0;
    bson_iter it; const char *k; const uint8_t *v; size_t vn; long got = 0;
    if (bson_iter_init(&it, arr, an)) while (got < max && bson_next(&it, &t, &k, &v, &vn)) if (t == 3) { each(q, v, vn, arg); got++; }
    return got;
}
static void kill_cursor(qctx *q, int64_t id, const char *coll) {   /* 没拉完就到 limit 了: 让服务器放掉游标, 不然它占着 10 分钟 */
    ds4_buf cmd = {0}; const size_t s = bson_begin(&cmd);
    bson_str(&cmd, "killCursors", coll); const size_t a = bson_sub(&cmd, "cursors", true); bson_int64(&cmd, "0", id); bson_end(&cmd, a); bson_str(&cmd, "$db", q->t->db); bson_end(&cmd, s);
    uint8_t *rep; size_t rn; if (mongo_command(&q->c, (const uint8_t *)cmd.ptr, cmd.len, &rep, &rn)) free(rep);
    ds4_buf_free(&cmd);
}
/* 首批 + getMore 直到 max 条或游标耗尽 */
static bool drain(qctx *q, ds4_buf *cmd, const char *coll, long max, doc_fn each, void *arg, long *total) {
    uint8_t *rep; size_t rn; int64_t id = 0;
    if (!cmd_run(q, cmd, &rep, &rn)) return false;
    *total = batch(q, rep, rn, &id, max, each, arg); free(rep);
    while (id && *total < max) {
        ds4_buf gm = {0}; const size_t s = bson_begin(&gm);
        bson_int64(&gm, "getMore", id); bson_str(&gm, "collection", coll); bson_int32(&gm, "batchSize", (int32_t)(max - *total < MONGO_BATCH ? max - *total : MONGO_BATCH)); bson_int32(&gm, "maxTimeMS", MONGO_MAX_TIME_MS); bson_str(&gm, "$db", q->t->db); bson_end(&gm, s);
        if (!cmd_run(q, &gm, &rep, &rn)) return false;
        id = 0; *total += batch(q, rep, rn, &id, max - *total, each, arg); free(rep);
    }
    if (id) kill_cursor(q, id, coll);
    return true;
}

static void emit_line(qctx *q, const uint8_t *doc, size_t n, void *arg) { (void)q; bson_to_json(doc, n, arg); ds4_buf_putc(arg, '\n'); }   /* arg = 行缓冲(摘要行要等条数知道了才能写在前面) */

/* collections: listCollections → 名字, 再逐个 count */
static void coll_name(qctx *q, const uint8_t *doc, size_t n, void *arg) {
    ds4_buf *names = arg; uint8_t t; const uint8_t *v; size_t vn;
    const char *nm = bson_find(doc, n, "name", &t, &v, &vn) ? bson_string(t, v) : NULL;
    const char *ty = bson_find(doc, n, "type", &t, &v, &vn) ? bson_string(t, v) : "collection";
    if (nm) { ds4_buf_puts(names, nm); ds4_buf_putc(names, ty && !strcmp(ty, "view") ? '\t' : '\n'); }   /* \t 标视图(不 count, count 在视图上要跑整条聚合) */
    (void)q;
}
static bool list_collections(qctx *q, ds4_buf *names) {
    ds4_buf cmd = {0}; const size_t s = bson_begin(&cmd);
    bson_int32(&cmd, "listCollections", 1); bson_bool(&cmd, "nameOnly", true); bson_str(&cmd, "$db", q->t->db); bson_end(&cmd, s);
    long total;
    return drain(q, &cmd, "$cmd.listCollections", 10000, coll_name, names, &total);
}
static bool do_collections(qctx *q) {
    ds4_buf names = {0};
    if (!list_collections(q, &names)) return false;
    int k = 0; ds4_buf body = {0};
    for (char *p = names.ptr; p && *p; ) {
        char *e = strpbrk(p, "\n\t"); const char sep = e ? *e : '\n'; if (e) *e = 0;
        if (sep == '\t') ds4_buf_printf(&body, "%s (view)\n", p);
        else {
            ds4_buf cmd = {0}; const size_t s = bson_begin(&cmd);
            bson_str(&cmd, "count", p); bson_int32(&cmd, "maxTimeMS", MONGO_MAX_TIME_MS); bson_str(&cmd, "$db", q->t->db); bson_end(&cmd, s);
            uint8_t *rep; size_t rn; uint8_t t; const uint8_t *v; size_t vn;
            if (!cmd_run(q, &cmd, &rep, &rn)) { ds4_buf_free(&names); ds4_buf_free(&body); return false; }
            ds4_buf_printf(&body, "%s · %lld documents\n", p, (long long)(bson_find(rep, rn, "n", &t, &v, &vn) ? bson_num(t, v) : 0)); free(rep);
        }
        k++; p = e ? e + 1 : NULL;
    }
    ds4_buf_free(&names);
    if (!k) ds4_buf_printf(&q->out, "database %s has no collections (check the database name)\n", q->t->db);
    else { ds4_buf_printf(&q->out, "%d collections in %s (name · documents):\n", k, q->t->db); ds4_buf_append(&q->out, body.ptr, body.len); }
    ds4_buf_free(&body);
    return true;
}

/* fields: 抽样合并顶层键: 名字 · 类型 · 第一次见到的样例值 */
typedef struct { char key[128]; char type[16]; char sample[SAMPLE_CHARS * 2 + 8]; int seen; } fld;
typedef struct { fld *f; int n; } flds;
static const char *tname(uint8_t t) { switch (t) { case 1: return "double"; case 2: return "string"; case 3: return "object"; case 4: return "array"; case 5: return "binary"; case 7: return "ObjectId"; case 8: return "bool"; case 9: return "date"; case 0x0A: return "null"; case 0x10: return "int"; case 0x12: return "long"; case 0x13: return "decimal"; default: return "other"; } }
static void collect(qctx *q, const uint8_t *doc, size_t n, void *arg) {
    flds *F = arg; bson_iter it; uint8_t t; const char *k; const uint8_t *v; size_t vn; (void)q;
    if (!bson_iter_init(&it, doc, n)) return;
    while (bson_next(&it, &t, &k, &v, &vn)) {
        int i = 0; while (i < F->n && strcmp(F->f[i].key, k)) i++;
        if (i == F->n) {
            if (F->n == FIELDS_MAX) continue;
            fld *f = &F->f[F->n++]; memset(f, 0, sizeof *f); snprintf(f->key, sizeof f->key, "%s", k); snprintf(f->type, sizeof f->type, "%s", tname(t));
            /* 样例: 把这个元素单独编回 JSON 最省事 —— 造一个只含它的文档 {"x":…} 走 bson_to_json, 再剥掉外壳 */
            ds4_buf d = {0}; const size_t st = bson_begin(&d); ds4_buf_putc(&d, (char)t); ds4_buf_append(&d, "x", 2); ds4_buf_append(&d, v, vn); bson_end(&d, st);
            ds4_buf j = {0}; bson_to_json((const uint8_t *)d.ptr, d.len, &j);
            const char *val = j.ptr + 5;
            size_t len = j.len > 6 ? j.len - 6 : 0;
            if (len > SAMPLE_CHARS) { len = SAMPLE_CHARS; while (len && ((unsigned char)val[len] & 0xC0) == 0x80) len--; snprintf(f->sample, sizeof f->sample, "%.*s…", (int)len, val); }
            else snprintf(f->sample, sizeof f->sample, "%.*s", (int)len, val);
            ds4_buf_free(&d); ds4_buf_free(&j);
        }
        F->f[i].seen++;
    }
}
static bool do_fields(qctx *q, const mongo_req *r) {
    ds4_buf cmd = {0}; const size_t s = bson_begin(&cmd);
    bson_str(&cmd, "find", r->coll); bson_int32(&cmd, "limit", FIELDS_SAMPLE); bson_int32(&cmd, "maxTimeMS", MONGO_MAX_TIME_MS); bson_str(&cmd, "$db", q->t->db); bson_end(&cmd, s);
    flds F = { calloc(FIELDS_MAX, sizeof(fld)), 0 }; long total = 0;
    const bool ok = F.f && drain(q, &cmd, r->coll, FIELDS_SAMPLE, collect, &F, &total);
    if (ok) {
        if (!total) ds4_buf_printf(&q->out, "collection %s is empty or does not exist\n", r->coll);
        else {
            ds4_buf_printf(&q->out, "%d fields seen in %ld sampled documents of %s (field · type · in N of sample · example):\n", F.n, total, r->coll);
            for (int i = 0; i < F.n; i++) ds4_buf_printf(&q->out, "%s · %s · %d/%ld · %s\n", F.f[i].key, F.f[i].type, F.f[i].seen, total, F.f[i].sample);
        }
    }
    free(F.f);
    return ok;
}

static bool do_find(qctx *q, const mongo_req *r) {
    ds4_buf cmd = {0}; const size_t s = bson_begin(&cmd);
    bson_str(&cmd, "find", r->coll);
    if (!add_json(&cmd, "filter", r->filter, false, q) || !add_json(&cmd, "projection", r->projection, false, q) || !add_json(&cmd, "sort", r->sort, false, q)) { ds4_buf_free(&cmd); return false; }
    bson_int32(&cmd, "limit", (int32_t)r->limit); bson_int32(&cmd, "batchSize", (int32_t)(r->limit < MONGO_BATCH ? r->limit : MONGO_BATCH)); bson_int32(&cmd, "maxTimeMS", MONGO_MAX_TIME_MS); bson_str(&cmd, "$db", q->t->db); bson_end(&cmd, s);
    ds4_buf lines = {0}; long total = 0;
    const bool ok = drain(q, &cmd, r->coll, r->limit, emit_line, &lines, &total);
    if (ok) { ds4_buf_printf(&q->out, "%ld documents from %s.%s (limit %ld)%s\n", total, q->t->db, r->coll, r->limit, total == r->limit ? "; there may be more: raise limit or narrow the filter" : ""); ds4_buf_append(&q->out, lines.ptr, lines.len); }
    ds4_buf_free(&lines);
    return ok;
}
static bool do_aggregate(qctx *q, const mongo_req *r) {
    if (!r->pipeline || !r->pipeline[0]) { snprintf(q->err, sizeof q->err, "aggregate needs pipeline (a JSON array of stages)"); return false; }
    ds4_buf cmd = {0}; const size_t s = bson_begin(&cmd);
    bson_str(&cmd, "aggregate", r->coll);
    if (!add_json(&cmd, "pipeline", r->pipeline, true, q)) { ds4_buf_free(&cmd); return false; }
    const size_t c = bson_sub(&cmd, "cursor", false); bson_int32(&cmd, "batchSize", (int32_t)(r->limit < MONGO_BATCH ? r->limit : MONGO_BATCH)); bson_end(&cmd, c);
    bson_int32(&cmd, "maxTimeMS", MONGO_MAX_TIME_MS); bson_str(&cmd, "$db", q->t->db); bson_end(&cmd, s);
    ds4_buf lines = {0}; long total = 0;
    const bool ok = drain(q, &cmd, r->coll, r->limit, emit_line, &lines, &total);
    if (ok) { ds4_buf_printf(&q->out, "%ld results from aggregate on %s.%s (limit %ld)%s\n", total, q->t->db, r->coll, r->limit, total == r->limit ? "; there may be more: add a $limit stage or raise limit" : ""); ds4_buf_append(&q->out, lines.ptr, lines.len); }
    ds4_buf_free(&lines);
    return ok;
}
static bool do_count(qctx *q, const mongo_req *r) {
    ds4_buf cmd = {0}; const size_t s = bson_begin(&cmd);
    bson_str(&cmd, "count", r->coll);
    if (!add_json(&cmd, "query", r->filter, false, q)) { ds4_buf_free(&cmd); return false; }
    bson_int32(&cmd, "maxTimeMS", MONGO_MAX_TIME_MS); bson_str(&cmd, "$db", q->t->db); bson_end(&cmd, s);
    uint8_t *rep; size_t rn; uint8_t t; const uint8_t *v; size_t vn;
    if (!cmd_run(q, &cmd, &rep, &rn)) return false;
    ds4_buf_printf(&q->out, "%lld documents in %s.%s match%s\n", (long long)(bson_find(rep, rn, "n", &t, &v, &vn) ? bson_num(t, v) : 0), q->t->db, r->coll, r->filter && r->filter[0] ? " the filter" : " (no filter)");
    free(rep);
    return true;
}
static bool do_distinct(qctx *q, const mongo_req *r) {
    if (!r->field || !r->field[0]) { snprintf(q->err, sizeof q->err, "distinct needs field"); return false; }
    ds4_buf cmd = {0}; const size_t s = bson_begin(&cmd);
    bson_str(&cmd, "distinct", r->coll); bson_str(&cmd, "key", r->field);
    if (!add_json(&cmd, "query", r->filter, false, q)) { ds4_buf_free(&cmd); return false; }
    bson_int32(&cmd, "maxTimeMS", MONGO_MAX_TIME_MS); bson_str(&cmd, "$db", q->t->db); bson_end(&cmd, s);
    uint8_t *rep; size_t rn; uint8_t t; const uint8_t *v; size_t vn;
    if (!cmd_run(q, &cmd, &rep, &rn)) return false;
    if (bson_find(rep, rn, "values", &t, &v, &vn) && t == 4) {
        ds4_buf j = {0}; ds4_buf d = {0}; const size_t st = bson_begin(&d); ds4_buf_putc(&d, 4); ds4_buf_append(&d, "v", 2); ds4_buf_append(&d, v, vn); bson_end(&d, st);
        bson_to_json((const uint8_t *)d.ptr, d.len, &j);
        bson_iter it; const char *k; const uint8_t *ev; size_t en; long cnt = 0; if (bson_iter_init(&it, v, vn)) while (bson_next(&it, &t, &k, &ev, &en)) cnt++;
        ds4_buf_printf(&q->out, "%ld distinct values of %s in %s.%s:\n%.*s\n", cnt, r->field, q->t->db, r->coll, (int)(j.len > 6 ? j.len - 6 : 0), j.ptr + 5);
        ds4_buf_free(&j); ds4_buf_free(&d);
    }
    free(rep);
    return true;
}

char *mongo_query(const mongo_target *t, const mongo_req *r, bool *ok) {
    *ok = false;
    qctx q = { .t = t }; q.err[0] = 0;
    const char *what = r->what ? r->what : "";
    const bool need_coll = strcmp(what, "collections") != 0;
    if (need_coll && (!r->coll || !r->coll[0])) { ds4_buf b = {0}; ds4_buf_printf(&b, "error: %s needs collection (call what=collections first)\n", what); return ds4_buf_take(&b); }
    const char *bad = mongo_forbidden(r->filter); if (!bad) bad = mongo_forbidden(r->pipeline); if (!bad) bad = mongo_forbidden(r->projection);
    if (bad) { ds4_buf b = {0}; ds4_buf_printf(&b, "error: %s is not allowed: this tool is read-only and does not run server-side JavaScript\n", bad); return ds4_buf_take(&b); }
    if (!mongo_connect(&q.c, t)) { ds4_buf b = {0}; ds4_buf_printf(&b, "error: %s\n", q.c.err); return ds4_buf_take(&b); }
    bool done;
    if (!strcmp(what, "collections")) done = do_collections(&q);
    else if (!strcmp(what, "fields")) done = do_fields(&q, r);
    else if (!strcmp(what, "find")) done = do_find(&q, r);
    else if (!strcmp(what, "count")) done = do_count(&q, r);
    else if (!strcmp(what, "distinct")) done = do_distinct(&q, r);
    else if (!strcmp(what, "aggregate")) done = do_aggregate(&q, r);
    else { snprintf(q.err, sizeof q.err, "what must be collections, fields, find, count, distinct or aggregate"); done = false; }
    mongo_close(&q.c);
    if (!done) { ds4_buf_free(&q.out); ds4_buf b = {0}; ds4_buf_printf(&b, "error: %s\n", q.err); return ds4_buf_take(&b); }
    *ok = true;
    return ds4_buf_take(&q.out);
}

bool mongo_probe(const mongo_target *t, char *version, size_t vn, int *ncoll, char *dbs, size_t dn, char *err, size_t errn) {
    qctx q = { .t = t }; *ncoll = 0; version[0] = 0; dbs[0] = 0;
    if (!mongo_connect(&q.c, t)) { snprintf(err, errn, "%s", q.c.err); return false; }
    ds4_buf cmd = {0}; size_t s = bson_begin(&cmd); bson_int32(&cmd, "buildInfo", 1); bson_str(&cmd, "$db", "admin"); bson_end(&cmd, s);
    uint8_t *rep; size_t rn; uint8_t ty; const uint8_t *v; size_t vl;
    if (cmd_run(&q, &cmd, &rep, &rn)) { if (bson_find(rep, rn, "version", &ty, &v, &vl) && bson_string(ty, v)) snprintf(version, vn, "%s", bson_string(ty, v)); free(rep); }
    ds4_buf names = {0};
    const bool ok = list_collections(&q, &names);
    if (ok) for (const char *p = names.ptr; p && *p; p = strpbrk(p, "\n\t"), p = p ? p + 1 : NULL) (*ncoll)++;
    ds4_buf_free(&names);
    if (ok && !*ncoll) {   /* 库名可能写错: 列一下服务器上有哪些库(没权限就空着) */
        s = bson_begin(&cmd); bson_int32(&cmd, "listDatabases", 1); bson_bool(&cmd, "nameOnly", true); bson_str(&cmd, "$db", "admin"); bson_end(&cmd, s);
        if (cmd_run(&q, &cmd, &rep, &rn)) {
            if (bson_find(rep, rn, "databases", &ty, &v, &vl) && ty == 4) {
                bson_iter it; const char *k; const uint8_t *ev; size_t en; size_t o = 0;
                if (bson_iter_init(&it, v, vl)) while (bson_next(&it, &ty, &k, &ev, &en)) {
                    uint8_t nt; const uint8_t *nv; size_t nn;
                    if (bson_find(ev, en, "name", &nt, &nv, &nn) && bson_string(nt, nv) && o + strlen(bson_string(nt, nv)) + 3 < dn) o += (size_t)snprintf(dbs + o, dn - o, "%s%s", o ? ", " : "", bson_string(nt, nv));
                }
            }
            free(rep);
        }
    }
    if (!ok) snprintf(err, errn, "%s", q.err);
    mongo_close(&q.c);
    return ok;
}
