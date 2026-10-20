/* ds4_pretok.c — 官方 tokenizer.json 预切分的 C 实现; 规则与为什么见 ds4_pretok.h。 */
#include "ds4_pretok.h"
#include "ds4_unicode_tab.h"
#include <stdbool.h>

static bool in_tab(const uint32_t (*t)[2], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < t[mid][0]) hi = mid; else if (cp > t[mid][1]) lo = mid + 1; else return true;
    }
    return false;
}
#define TAB(t) t, sizeof(t) / sizeof(t[0])

int ds4_uc_class(uint32_t cp) {
    if (cp < 0x80) {   /* ASCII 走直判, 表只在非 ASCII 才查 */
        if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return DS4_UC_L;
        if (cp >= '0' && cp <= '9') return DS4_UC_N;
        if ((cp >= '!' && cp <= '/') || (cp >= ':' && cp <= '@') || (cp >= '[' && cp <= '`') || (cp >= '{' && cp <= '~')) return DS4_UC_PS;
        return 0;
    }
    int c = 0;
    if (in_tab(TAB(uc_tab_L), cp)) c |= DS4_UC_L;
    if (in_tab(TAB(uc_tab_M), cp)) c |= DS4_UC_M;
    if (in_tab(TAB(uc_tab_N), cp)) c |= DS4_UC_N;
    if (in_tab(TAB(uc_tab_PS), cp)) c |= DS4_UC_PS;
    return c;
}

/* 正则 \s = Unicode White_Space(regex crate 的定义), 不只 ASCII: 全角空格 U+3000 / NBSP U+00A0 也算 */
static bool uc_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}
static bool uc_newline(uint32_t cp) { return cp == '\n' || cp == '\r'; }
static bool uc_cjk(uint32_t cp) {   /* 规则②的三段: CJK 统一表意文字 / 平假名 / 片假名 */
    return (cp >= 0x4E00 && cp <= 0x9FA5) || (cp >= 0x3040 && cp <= 0x309F) || (cp >= 0x30A0 && cp <= 0x30FF);
}

/* 解一个 UTF-8 码位; 坏字节当单字节码位(与 BPE 字节回退同步, 不会卡死) */
static uint32_t peek(const char *s, size_t len, size_t pos, size_t *next) {
    const uint8_t c0 = (uint8_t)s[pos];
    size_t n = c0 < 0x80 ? 1 : (c0 >> 5) == 6 ? 2 : (c0 >> 4) == 14 ? 3 : (c0 >> 3) == 30 ? 4 : 1;
    if (pos + n > len) n = 1;
    for (size_t i = 1; i < n; i++) if (((uint8_t)s[pos + i] & 0xC0) != 0x80) { n = 1; break; }
    *next = pos + n;
    if (n == 1) return c0;
    uint32_t cp = c0 & (0x7F >> n);
    for (size_t i = 1; i < n; i++) cp = (cp << 6) | ((uint8_t)s[pos + i] & 0x3F);
    return cp;
}

/* 规则③在段 [pos, end) 上从 pos 起择一匹配; 匹配不到返回 pos(调用方把它并进"空隙片") */
static size_t rule3_match(const char *s, size_t end, size_t pos) {
    size_t nx, p;
    uint32_t c = peek(s, end, pos, &nx);
    int cls = ds4_uc_class(c);
    /* A: ASCII 标点 + [A-Za-z]+ ("'m" "'t") */
    if (c < 0x80 && (cls & DS4_UC_PS) && nx < end && (uint8_t)s[nx] < 0x80 && ds4_uc_class((uint8_t)s[nx]) == DS4_UC_L) {
        p = nx;
        while (p < end && (uint8_t)s[p] < 0x80 && ds4_uc_class((uint8_t)s[p]) == DS4_UC_L) p++;
        return p;
    }
    /* B: 可选一个前缀字符(非换行、非字母/标点/符号 —— 空格、制表、全角空格、控制字符都行) + [字母|标记]+ */
    size_t lm_from = (cls & (DS4_UC_L | DS4_UC_M)) ? pos : (!uc_newline(c) && !(cls & (DS4_UC_L | DS4_UC_PS))) ? nx : (size_t)-1;
    if (lm_from != (size_t)-1 && lm_from < end) {
        size_t q = lm_from, qn;
        while (q < end && (ds4_uc_class(peek(s, end, q, &qn)) & (DS4_UC_L | DS4_UC_M))) q = qn;
        if (q > lm_from) return q;
    }
    /* C: 可选一个空格 + [标点|符号]+ + 换行* */
    p = (c == ' ') ? nx : pos;
    {
        size_t q = p, qn;
        while (q < end && (ds4_uc_class(peek(s, end, q, &qn)) & DS4_UC_PS)) q = qn;
        if (q > p) {
            while (q < end && uc_newline((uint8_t)s[q])) q++;
            return q;
        }
    }
    /* D/E/F: 空白串 —— 含换行就切到最后一个换行; 否则后面有非空白就少留一个字符给它(单个空白直接整个吃掉) */
    if (uc_space(c)) {
        size_t q = pos, qn, last_nl = 0, nsp = 0, prev = pos;
        while (q < end) {
            uint32_t w = peek(s, end, q, &qn);
            if (!uc_space(w)) break;
            if (uc_newline(w)) last_nl = qn;
            prev = q; q = qn; nsp++;
        }
        if (last_nl) return last_nl;
        if (q < end && nsp > 1) return prev;
        return q;
    }
    return pos;
}

size_t ds4_pretok_next(ds4_pretok *st, const char *s, size_t len, size_t pos) {
    size_t nx;
    if (pos >= st->seg_end) {   /* 进入新段: 数字片(≤3 个数字) / CJK 串 / 到下一个数字或 CJK 为止的普通段 */
        uint32_t c = peek(s, len, pos, &nx);
        if (ds4_uc_class(c) & DS4_UC_N) {
            size_t q = nx, qn; int k = 1;
            while (q < len && k < 3 && (ds4_uc_class(peek(s, len, q, &qn)) & DS4_UC_N)) { q = qn; k++; }
            st->seg_end = q;
            return q;
        }
        if (uc_cjk(c)) {
            size_t q = nx, qn;
            while (q < len && uc_cjk(peek(s, len, q, &qn))) q = qn;
            st->seg_end = q;
            return q;
        }
        size_t q = nx, qn;
        while (q < len) {
            uint32_t w = peek(s, len, q, &qn);
            if ((ds4_uc_class(w) & DS4_UC_N) || uc_cjk(w)) break;
            q = qn;
        }
        st->seg_end = q;
    }
    size_t m = rule3_match(s, st->seg_end, pos);
    if (m > pos) return m;
    /* 空隙: 规则③谁都不匹配的字符连成一片, 直到下一个匹配开始或段尾 */
    size_t q = pos;
    peek(s, st->seg_end, q, &q);
    while (q < st->seg_end && rule3_match(s, st->seg_end, q) == q) peek(s, st->seg_end, q, &q);
    return q;
}
