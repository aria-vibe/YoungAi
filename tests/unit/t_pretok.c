/* t_pretok.c — BPE 预切分对官方 tokenizer.json 预切分器的金标(离线, 无模型; 2026-10-20)。
 * 用例与切片边界由 gguf-tools/scripts/unicode_pretok_gen.py 从官方 tokenizers 生成(pretok_cases.h), 这里逐条比片长。
 * 为什么要它: 10-20 实撞 —— 中文标点被切开, 11k 的真实提示有 27 处 token 与官方不同; 这类错只有对官方切法逐片比才抓得到。 */
#include "ds4_pretok.h"
#include "pretok_cases.h"
#include <stdio.h>
#include <string.h>

int unit_pretok(void) {
    int fail = 0;
    const size_t n = sizeof(pretok_cases) / sizeof(pretok_cases[0]);
    for (size_t i = 0; i < n; i++) {
        const char *s = pretok_cases[i].text;
        const size_t len = strlen(s);
        ds4_pretok st = {0};
        size_t pos = 0; int k = 0; int bad = 0;
        while (pos < len) {
            size_t end = ds4_pretok_next(&st, s, len, pos);
            if (end <= pos) { bad = 1; break; }
            if (pretok_cases[i].lens[k] != (int)(end - pos)) { bad = 1; break; }
            pos = end; k++;
        }
        if (!bad && pretok_cases[i].lens[k] != 0) bad = 1;
        if (bad) {
            fail++;
            fprintf(stderr, "FAIL pretok 用例 %zu \"%s\": 第 %d 片不对(期望 %d)\n", i, s, k, pretok_cases[i].lens[k]);
            ds4_pretok st2 = {0};
            for (size_t p = 0; p < len;) { size_t e = ds4_pretok_next(&st2, s, len, p); fprintf(stderr, "  片 [%.*s] %zu\n", (int)(e - p), s + p, e - p); if (e <= p) break; p = e; }
        }
    }
    /* 类别函数本身的几个锚点: 中文句号/全角逗号/右引号是标点, 箭头/表情是符号, 全角数字是数字, é 是字母, 组合重音是标记 */
    struct { uint32_t cp; int cls; } anchors[] = {
        {0x3002, DS4_UC_PS}, {0xFF0C, DS4_UC_PS}, {0x201D, DS4_UC_PS}, {0x2192, DS4_UC_PS}, {0x1F600, DS4_UC_PS},
        {0xFF11, DS4_UC_N}, {0xE9, DS4_UC_L}, {0x301, DS4_UC_M}, {0x4E2D, DS4_UC_L}, {0x3000, 0}, {'a', DS4_UC_L}, {'7', DS4_UC_N}, {'.', DS4_UC_PS},
    };
    for (size_t i = 0; i < sizeof(anchors) / sizeof(anchors[0]); i++)
        if (ds4_uc_class(anchors[i].cp) != anchors[i].cls) { fail++; fprintf(stderr, "FAIL uc_class U+%04X = %d, 期望 %d\n", anchors[i].cp, ds4_uc_class(anchors[i].cp), anchors[i].cls); }
    printf("pretok: %zu 条官方切法用例 + %zu 个类别锚点, %d 失败\n", n, sizeof(anchors) / sizeof(anchors[0]), fail);
    return fail;
}
