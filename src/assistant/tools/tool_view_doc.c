/* tool_view_doc.c — view_file 看 docx / pptx / xlsx: 读 zip 中央目录找正文部件, inflate(src/office 的 office_inflate, 不抄第二份), 去掉 XML 标签给正文。
 * 只给文字: 版式、图、图表看不到, 结果里说明了。为什么不走 soffice 渲染: 两台机器上 soffice 首次起要 5 s 还要可写 HOME, 而模型核的多半是"写进去的内容对不对",
 *   文字就够; 要看版式用户在文件卡里看。
 * xlsx: 单元格按 sharedStrings 还原字符串, 公式只给缓存值(<v>), 一行一行 "A1=…" 列出; docx 按 </w:p> 分段, pptx 按 slide 分页、</a:p> 分段。
 * 出错会怎样: 不是 zip / 中央目录坏 / 部件缺 → NULL(调用方报"不是能读的 Office 包"); inflate 失败的部件跳过。 */
#include "../asst_verify.h"
#include "../../office/ds4_office.h"
#include <strings.h>

static uint32_t le32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t le16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }

/* 中央目录里按名找一个部件并解出来(malloc, NUL 结尾); 没有 / 坏 → NULL */
static char *part(const uint8_t *z, size_t n, const char *name, size_t *outn) {
    if (n < 22) return NULL;
    size_t eocd = n - 22; const size_t stop = n > 66000 ? n - 66000 : 0;
    while (eocd > stop && le32(z + eocd) != 0x06054b50) eocd--;
    if (le32(z + eocd) != 0x06054b50) return NULL;
    uint32_t cnt = le16(z + eocd + 10), off = le32(z + eocd + 16);
    for (uint32_t i = 0; i < cnt && off + 46 <= n; i++) {
        const uint8_t *h = z + off;
        if (le32(h) != 0x02014b50) return NULL;
        const uint16_t method = le16(h + 10), nlen = le16(h + 28), elen = le16(h + 30), clen = le16(h + 32);
        const uint32_t csize = le32(h + 20), usize = le32(h + 24), lho = le32(h + 42);
        if (off + 46 + nlen > n) return NULL;
        if (nlen == strlen(name) && !memcmp(h + 46, name, nlen)) {
            if (lho + 30 > n || le32(z + lho) != 0x04034b50) return NULL;
            const size_t data = lho + 30 + le16(z + lho + 26) + le16(z + lho + 28);
            if (data + csize > n) return NULL;
            uint8_t *out = NULL; size_t on = 0;
            if (method == 0) { out = malloc(csize + 1); memcpy(out, z + data, csize); on = csize; }
            else if (method == 8) { if (!office_inflate(z + data, csize, &out, &on)) return NULL; out = realloc(out, on + 1); }
            else return NULL;
            (void)usize; out[on] = 0; *outn = on;
            return (char *)out;
        }
        off += 46u + nlen + elen + clen;
    }
    return NULL;
}

static void entity(ds4_buf *b, const char **p) {   /* &amp; &lt; &gt; &quot; &apos; &#NN; 其余原样 */
    static const struct { const char *e; char c; } T[] = { { "&amp;", '&' }, { "&lt;", '<' }, { "&gt;", '>' }, { "&quot;", '"' }, { "&apos;", '\'' } };
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++) if (!strncmp(*p, T[i].e, strlen(T[i].e))) { ds4_buf_putc(b, T[i].c); *p += strlen(T[i].e); return; }
    if ((*p)[1] == '#') { const char *q = *p + 2; long v = strtol(q, (char **)&q, 10); if (*q == ';' && v > 0 && v < 128) { ds4_buf_putc(b, (char)v); *p = q + 1; return; } }
    ds4_buf_putc(b, '&'); (*p)++;
}

/* XML → 文字: 段落闭合处换行(紧跟着单元格闭合时不换, 不然表格一格一行), 单元格闭合处制表, 行闭合处换行; 其余标签丢掉 */
static void strip(const char *xml, const char *para_end, const char *cell_end, const char *row_end, ds4_buf *b, size_t max) {
    bool nl = false;
    for (const char *p = xml; *p && b->len < max; ) {
        if (*p == '<') {
            if (para_end && !strncmp(p, para_end, strlen(para_end))) nl = true;
            else if (cell_end && !strncmp(p, cell_end, strlen(cell_end))) { ds4_buf_putc(b, '\t'); nl = false; }
            else if (row_end && !strncmp(p, row_end, strlen(row_end))) { ds4_buf_putc(b, '\n'); nl = false; }
            const char *e = strchr(p, '>'); if (!e) break; p = e + 1;
        } else {
            if (nl) { ds4_buf_putc(b, '\n'); nl = false; }
            if (*p == '&') entity(b, &p); else ds4_buf_putc(b, *p++);
        }
    }
    if (nl) ds4_buf_putc(b, '\n');
}

static void xlsx(const uint8_t *z, size_t n, ds4_buf *b, size_t max) {
    size_t sn = 0; char *ss = part(z, n, "xl/sharedStrings.xml", &sn);
    char **str = NULL; int ns = 0;
    for (const char *p = ss; p && (p = strstr(p, "<si>")); ) {   /* 每个 <si> 里所有 <t> 拼成一条 */
        const char *e = strstr(p, "</si>"); if (!e) break;
        ds4_buf one = {0};
        for (const char *t = p; (t = strstr(t, "<t")) && t < e; ) { const char *c = strchr(t, '>'); if (!c) break; const char *te = strstr(c, "</t>"); if (!te || te > e) break; char tmp[4096]; snprintf(tmp, sizeof tmp, "%.*s", (int)(te - c - 1), c + 1); strip(tmp, NULL, NULL, NULL, &one, 4096); t = te + 4; }
        str = realloc(str, (size_t)(ns + 1) * sizeof *str); str[ns++] = one.ptr ? one.ptr : strdup("");
        p = e + 5;
    }
    free(ss);
    for (int s = 1; s <= 50 && b->len < max; s++) {
        char name[48]; snprintf(name, sizeof name, "xl/worksheets/sheet%d.xml", s);
        size_t xn = 0; char *x = part(z, n, name, &xn); if (!x) break;
        ds4_buf_printf(b, "%s[sheet %d]\n", s > 1 ? "\n" : "", s);
        const char *row = NULL;
        for (const char *c = x; (c = strstr(c, "<c ")) && b->len < max; ) {
            const char *ce = strchr(c, '>'); if (!ce) break;
            char ref[16] = ""; const char *r = strstr(c, " r=\""); if (r && r < ce) snprintf(ref, sizeof ref, "%.*s", (int)strcspn(r + 4, "\""), r + 4);
            const char *rowdig = ref; while (*rowdig && !(*rowdig >= '0' && *rowdig <= '9')) rowdig++;
            if (!row || strcmp(row, rowdig)) { if (row) ds4_buf_putc(b, '\n'); row = rowdig; }
            else ds4_buf_puts(b, "  ");
            const bool shared = strstr(c, " t=\"s\"") && strstr(c, " t=\"s\"") < ce;
            const char *cend = ce[-1] == '/' ? ce + 1 : strstr(ce, "</c>");
            const char *v = strstr(ce, "<v>"); const char *f = strstr(ce, "<f>"), *is = strstr(ce, "<is>");
            ds4_buf_printf(b, "%s=", ref);
            if (is && cend && is < cend) { const char *ie = strstr(is, "</is>"); char tmp[4096]; snprintf(tmp, sizeof tmp, "%.*s", (int)(ie && ie < cend ? ie - is : cend - is), is); strip(tmp, NULL, NULL, NULL, b, max); }   /* inlineStr(我们的写出器用它) */
            else if (v && cend && v < cend) {
                const char *ve = strstr(v, "</v>"); char val[512]; snprintf(val, sizeof val, "%.*s", (int)(ve ? ve - v - 3 : 0), v + 3);
                if (shared) { const int k = atoi(val); ds4_buf_puts(b, k >= 0 && k < ns ? str[k] : "?"); } else ds4_buf_puts(b, val);
            }
            if (f && cend && f < cend) { const char *fe = strstr(f, "</f>"); ds4_buf_printf(b, " {=%.*s}", (int)(fe ? fe - f - 3 : 0), f + 3); }
            c = cend ? cend : ce;
        }
        ds4_buf_putc(b, '\n'); free(x);
    }
    for (int i = 0; i < ns; i++) free(str[i]); free(str);
}

char *asst_view_office(const char *abs, size_t max) {
    size_t n = 0; uint8_t *z = (uint8_t *)asst_slurp(abs, &n);
    if (!z) return NULL;
    ds4_buf b = {0};
    const char *e = strrchr(abs, '.'); e = e ? e + 1 : "";
    if (!strcasecmp(e, "docx")) {
        size_t xn = 0; char *x = part(z, n, "word/document.xml", &xn);
        if (!x) { free(z); return NULL; }
        strip(x, "</w:p>", "</w:tc>", "</w:tr>", &b, max); free(x);
    } else if (!strcasecmp(e, "pptx")) {
        int k = 0;
        for (int s = 1; s <= 200 && b.len < max; s++) {
            char name[48]; snprintf(name, sizeof name, "ppt/slides/slide%d.xml", s);
            size_t xn = 0; char *x = part(z, n, name, &xn); if (!x) break;
            ds4_buf_printf(&b, "%s[slide %d]\n", s > 1 ? "\n" : "", s); strip(x, "</a:p>", "</a:tc>", "</a:tr>", &b, max); free(x); k++;
        }
        if (!k) { free(z); ds4_buf_free(&b); return NULL; }
    } else if (!strcasecmp(e, "xlsx")) {
        xlsx(z, n, &b, max);
        if (!b.len) { free(z); return NULL; }
    } else { free(z); return NULL; }
    free(z);
    if (b.len >= max) ds4_buf_puts(&b, "\n[… truncated]");
    return b.ptr ? ds4_buf_take(&b) : strdup("");
}
