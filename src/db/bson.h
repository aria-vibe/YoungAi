/* bson.h — BSON(MongoDB 的二进制文档格式, bsonspec.org)编解码 + JSON 互转。纯 C, 零依赖(方案稿 docs/assistant_connectors.md §5)。
 * 编码往 ds4_buf 里追加; 解码在原字节上走迭代器, 不拷贝、不建树。
 * JSON → BSON 认三个扩展对象: {"$oid":"24 位十六进制"} → ObjectId, {"$date":"ISO 8601" | 毫秒数} → UTC 日期, {"$numberLong":"…"} → int64;
 * BSON → JSON 把 ObjectId / 日期也写成这两种形式 —— 模型从结果里原样抄进下一次 filter 就能匹配, 写成裸字符串就对不上 _id 了。 */
#ifndef DS4_BSON_H
#define DS4_BSON_H
#include "../common/ds4_json.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define BSON_MAX_DEPTH 64   /* 模型给的 filter 可以嵌套, 封顶防栈爆(服务器自己的上限是 100) */

/* ---- 编码 ---- */
size_t bson_begin(ds4_buf *b);                               /* 开一个文档: 占 4 字节长度, 返回其偏移 */
void bson_end(ds4_buf *b, size_t start);                     /* 写 0x00 结尾并回填长度 */
size_t bson_sub(ds4_buf *b, const char *key, bool array);    /* key 下开嵌套文档/数组, 用 bson_end 收 */
void bson_int32(ds4_buf *b, const char *key, int32_t v);
void bson_int64(ds4_buf *b, const char *key, int64_t v);
void bson_double(ds4_buf *b, const char *key, double v);
void bson_str(ds4_buf *b, const char *key, const char *s);
void bson_bool(ds4_buf *b, const char *key, bool v);
void bson_null(ds4_buf *b, const char *key);
void bson_bin(ds4_buf *b, const char *key, const void *p, size_t n);   /* 二进制 subtype 0(SCRAM 的 payload) */
/* 一个 JSON 值编成 key 下的元素; *json 前移到值之后。对象 → 文档, 数组 → 数组, 整数 → int32/int64, 其余 → double/字符串/bool/null */
bool bson_json_value(ds4_buf *b, const char *key, const char **json, char *err, size_t errn);
/* 整个 JSON 对象的字段并入当前打开的文档(不另起一层): 把模型给的 filter 平铺进 find 命令用 */
bool bson_json_fields(ds4_buf *b, const char *json, char *err, size_t errn);

/* ---- 解码 ---- */
typedef struct { const uint8_t *p, *end; } bson_iter;
size_t bson_doclen(const uint8_t *doc, size_t avail);        /* 文档声明的长度, 不合法(短于 5、超出 avail、末尾不是 0)给 0 */
bool bson_iter_init(bson_iter *it, const uint8_t *doc, size_t avail);
bool bson_next(bson_iter *it, uint8_t *type, const char **key, const uint8_t **val, size_t *vlen);   /* 走到结尾或遇坏元素返回 false */
bool bson_find(const uint8_t *doc, size_t avail, const char *key, uint8_t *type, const uint8_t **val, size_t *vlen);   /* 顶层找 key */
int64_t bson_num(uint8_t type, const uint8_t *val);          /* int32/int64/double/bool → 整数(其它 0) */
double bson_dbl(uint8_t type, const uint8_t *val);
const char *bson_string(uint8_t type, const uint8_t *val);   /* 0x02 字符串(NUL 结尾); 其它类型 NULL */
void bson_to_json(const uint8_t *doc, size_t avail, ds4_buf *out);   /* 文档 → 一行 JSON */

/* 日期: 毫秒 ↔ ISO 8601(UTC); 给页面/测试也用 */
bool bson_iso_to_ms(const char *s, int64_t *ms);             /* "2026-10-18", "2026-10-18T09:30:00Z", "…+08:00", 带小数秒 */
void bson_ms_to_iso(int64_t ms, char *out, size_t n);

#endif
