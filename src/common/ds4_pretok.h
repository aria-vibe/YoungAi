/* ds4_pretok.h — DeepSeek V4/V4.1 BPE 预切分(tokenizer.json 的 pre_tokenizer), 纯主机 C, 无引擎依赖(2026-10-20)。
 *
 * 官方 pre_tokenizer 是三道 Split(Isolated)串起来: ① \p{N}{1,3}  ② [一-龥぀-ゟ゠-ヿ]+
 *   ③ [ASCII 标点][A-Za-z]+ | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+ |  ?[\p{P}\p{S}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
 * 先按①②把数字串/CJK 串切出来当独立片, 剩下的段再按③从左到右择一匹配; ③里 (?!\S) 的"后面"只看本段 —— 段尾就算没有
 * 非空白。这一刀切错了 BPE 合并就不一样: "。\n" 官方一个 token(876), 切成两片就是 "。"+"\n" 两个, 模型看到的是训练里没见过的序列。
 * 用法: ds4_pretok st = {0}; for (pos = 0; pos < len; pos = end) { end = ds4_pretok_next(&st, s, len, pos); 片 = [pos, end) }
 *   st 只缓存当前段的末尾, 顺序扫一遍是 O(n); 换一段文本就重新清零。
 * 类别表 ds4_unicode_tab.h 是生成件(gguf-tools/scripts/unicode_pretok_gen.py); 金标用例 tests/unit/pretok_cases.h 同一脚本生成。 */
#ifndef DS4_PRETOK_H
#define DS4_PRETOK_H
#include <stddef.h>
#include <stdint.h>

enum { DS4_UC_L = 1, DS4_UC_M = 2, DS4_UC_N = 4, DS4_UC_PS = 8 };   /* 字母 / 组合标记 / 数字 / 标点或符号 */
int ds4_uc_class(uint32_t cp);                                     /* 位或; 其他类别(空白/控制/未分配)= 0 */

typedef struct { size_t seg_end; } ds4_pretok;
size_t ds4_pretok_next(ds4_pretok *st, const char *s, size_t len, size_t pos);   /* 返回 pos 起那一片的末尾(> pos) */

#endif
