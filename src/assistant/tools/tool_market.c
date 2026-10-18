/* tool_market.c — market_data: 股票 / 指数 / 基金的基础行情与财务查询(10-18 用户: "参考我的交易智能体, 实现一个金融 skill, 先把股票财经这些基础查询的能力封装进去")。
 * 数据源与交易智能体(quant_trading_flow 的 datahub / crews 工具)同一家: 东方财富公开接口, 不要 key ——
 *   代码搜索 searchapi suggest(名字/拼音/代码 → secid "市场号.代码")、批量行情 push2 ulist、K 线 push2his kline、
 *   财务主要指标 datacenter RPT_PCF10_FINANCEMAINFINADATA、公告 np-anotice、资讯搜索 search-api-web(JSONP)。
 * 为什么做成工具而不是在技能里写网址让模型 visit_page: 技能不许带命令(docs/assistant_skills.md §1.1); 而且接口原文几十 KB、字段是 f2/f3 这种编号,
 *   让模型自己拼 URL、认字段、算涨跌, 错一处就是一个编出来的价格。这里把字段翻成中文、金额换成亿/万、区间涨跌算好, 模型只管读。
 * 请求经火狐走 https(与 tool_geo.c 同: 本项目不引 TLS 库; 火狐关了 JSON 查看器, 正文就是原始 JSON)。
 * 行情可能延时: push2 对未登录访问 302 到 push2delay(10-18 实测 A 股周五收盘价与 K 线一致), 每只结果都带更新时间(北京时间), 模型据此说"截至几点"。
 * 出错会怎样: 接口改版 / 被风控 → 报 "unexpected response" 加原文开头, 模型照实说拿不到, 不会去编; 没火狐 → 工具不出现(need_browser)。 */
#include "../asst_internal.h"
#include <limits.h>   /* web_internal.h 用 PATH_MAX 却不自带 */
#include "../../web/web_internal.h"
#include <ctype.h>
#include <time.h>

#define MKT_QUOTE_MAX 20      /* 一次最多几只: 名字要逐个走代码搜索, 每个都是一次火狐导航 */
#define MKT_BARS_DEF 30       /* K 线默认根数: 日 K 一个半月, 够看"最近走势" */
#define MKT_BARS_MAX 250      /* 一年交易日; 再多正文就上万 token */
#define MKT_LIST_DEF 10       /* 资讯/公告条数 */
#define MKT_LIST_MAX 30
#define MKT_FIN_DEF 4         /* 财报期数: 最近四期 = 一年的季报/半年报/年报 */
#define MKT_FIN_MAX 12
#define MKT_SUMMARY_MAX 160   /* 资讯摘要截到这么多字节(UTF-8 边界), 十条一起不超过 3 KB */
#define BJ_OFFSET (8 * 3600)  /* 接口给的是 unix 秒; 一律按北京时间报, 港美股也是, 用户在国内 */
#define EM_SUGGEST_TOKEN "D43BF722C8E33BDC906FB84D85E326E8"   /* 东财网页自带的公开 token, 不是账号凭据 */

static char *(*mkt_get)(const char *url, char *err, size_t errn) = asst_web_text;
void asst_mkt_set_fetch(char *(*fn)(const char *, char *, size_t)) { mkt_get = fn ? fn : asst_web_text; }

static char *fail(const char *fmt, const char *a) { ds4_buf b = {0}; ds4_buf_printf(&b, fmt, a ? a : ""); return ds4_buf_take(&b); }
static char *unexpected(const char *what, const char *raw) {   /* 原文开头给模型看: 是风控页还是改版, 它能照实说 */
    ds4_buf b = {0}; ds4_buf_printf(&b, "error: unexpected response from Eastmoney %s: %.200s\n", what, raw ? raw : "(empty)"); return ds4_buf_take(&b);
}

/* 原文 → 数; 字符串("-" 停牌/无数据)与缺失回 false */
static bool num(const char *obj, const char *key, double *v) {
    char *r = NULL; bool ok = false;
    if (asst_jraw(obj, key, &r) && r[0] != '"' && strcmp(r, "null")) { char *e; *v = strtod(r, &e); ok = e != r; }
    free(r); return ok;
}
static void unq(char *s) { size_t n = strlen(s); if (n >= 2 && s[0] == '"' && s[n - 1] == '"') { memmove(s, s + 1, n - 2); s[n - 2] = 0; } }

/* 价格: 两位, 基金/港股仙股有第三位就留着 */
static void px(ds4_buf *b, double v) {
    char t[64]; snprintf(t, sizeof t, "%.3f", v);
    size_t n = strlen(t); if (n && t[n - 1] == '0') t[n - 1] = 0;
    ds4_buf_puts(b, t);
}
static void amt(ds4_buf *b, double v) {   /* 金额 / 股数: 亿、万 */
    const double a = v < 0 ? -v : v;
    if (a >= 1e8) ds4_buf_printf(b, "%.2f亿", v / 1e8); else if (a >= 1e4) ds4_buf_printf(b, "%.2f万", v / 1e4); else ds4_buf_printf(b, "%.0f", v);
}
/* 东财市场号 → 价格/成交额/市值的币种。不标的话港美股的"市值 50125 亿"会被当成人民币讲(10-18 苹果实测) */
static const char *currency(double mk) {
    const int m = (int)mk;
    if (m == 0 || m == 1) return "人民币";
    if (m == 116 || m == 128) return "港元";
    if (m == 105 || m == 106 || m == 107 || m == 153) return "美元";
    return NULL;   /* 100 = 全球指数(点数, 没有币种) */
}
static void bjtime(ds4_buf *b, double unix_s) {
    time_t t = (time_t)unix_s + BJ_OFFSET; struct tm tm; gmtime_r(&t, &tm);
    char s[32]; strftime(s, sizeof s, "%Y-%m-%d %H:%M", &tm); ds4_buf_puts(b, s);
}

/* 用户给的一个标的 → secid。认: "1.600519"(原样) / sh600519 sz000001 bj830799 / 6 位数字(6 9 5 开头沪市, 其余深市; 北交所在东财也是 0.)。
 * 6 位数字一律当股票/基金: 000001 是平安银行不是上证指数(要指数就给名字或 sh000001)。名字、拼音、港股 00700、美股 AAPL → false, 交给代码搜索 */
bool asst_mkt_secid(const char *s, char *out, size_t n) {
    while (*s == ' ') s++;
    const char *dot = strchr(s, '.');
    if (dot && dot > s && dot - s <= 3 && dot[1]) {
        bool d = true; for (const char *p = s; p < dot; p++) if (!isdigit((unsigned char)*p)) d = false;
        if (d) { snprintf(out, n, "%s", s); return true; }
    }
    int mk = -1; const char *code = s;
    if ((s[0] == 's' || s[0] == 'S' || s[0] == 'b' || s[0] == 'B') && strlen(s) == 8) {
        const char c1 = (char)tolower((unsigned char)s[1]), c0 = (char)tolower((unsigned char)s[0]);
        if (c0 == 's' && c1 == 'h') mk = 1; else if ((c0 == 's' && c1 == 'z') || (c0 == 'b' && c1 == 'j')) mk = 0; else return false;
        code = s + 2;
    }
    if (strlen(code) != 6) return false;
    for (int i = 0; i < 6; i++) if (!isdigit((unsigned char)code[i])) return false;
    if (mk < 0) mk = (code[0] == '6' || code[0] == '9' || code[0] == '5') ? 1 : 0;
    snprintf(out, n, "%d.%s", mk, code);
    return true;
}

/* 代码搜索的 JSON → 列表文本(search) 或第一条的 secid(解析标的用) */
char *asst_mkt_parse_suggest(const char *json, char *first, size_t fn, char *first_name, size_t nn) {
    char *t = NULL, *data = NULL, *e = NULL;
    if (first && fn) first[0] = 0;
    if (first_name && nn) first_name[0] = 0;
    if (!asst_jraw(json, "QuotationCodeTable", &t) || !asst_jraw(t, "Data", &data)) { free(t); return NULL; }
    ds4_buf b = {0}; int k = 0;
    for (int i = 0; i < 10 && asst_jarr_at(data, i, &e); i++, free(e), e = NULL) {
        char *name = NULL, *code = NULL, *ty = NULL, *qid = NULL;
        asst_jstr(e, "Name", &name); asst_jstr(e, "Code", &code); asst_jstr(e, "SecurityTypeName", &ty); asst_jstr(e, "QuoteID", &qid);
        if (name && qid && qid[0]) {
            ds4_buf_printf(&b, "%d. %s %s · %s · secid %s\n", ++k, name, code ? code : "", ty && ty[0] ? ty : "?", qid);
            if (k == 1 && first) snprintf(first, fn, "%s", qid);
            if (k == 1 && first_name) snprintf(first_name, nn, "%s", name);
        }
        free(name); free(code); free(ty); free(qid);
    }
    free(t); free(data);
    if (!k) ds4_buf_puts(&b, "(no match)\n");
    return ds4_buf_take(&b);
}

static char *suggest(const char *q, char *err, size_t errn) {
    char *enc = web_url_encode(q); ds4_buf u = {0};
    ds4_buf_printf(&u, "https://searchapi.eastmoney.com/api/suggest/get?input=%s&type=14&token=" EM_SUGGEST_TOKEN "&count=10", enc); free(enc);
    char *j = mkt_get(u.ptr, err, errn); ds4_buf_free(&u);
    return j;
}

/* 一个标的 → secid; 失败把原因写进 log 返回 false */
static bool resolve(const char *sym, char *secid, size_t n, ds4_buf *log) {
    if (asst_mkt_secid(sym, secid, n)) return true;
    char err[512] = ""; char *j = suggest(sym, err, sizeof err);
    if (!j) { ds4_buf_printf(log, "%s: code search failed: %s\n", sym, err); return false; }
    char *list = asst_mkt_parse_suggest(j, secid, n, NULL, 0);
    if (!list) ds4_buf_printf(log, "%s: unexpected code search response: %.120s\n", sym, j);
    else if (!secid[0]) ds4_buf_printf(log, "%s: no stock, index or fund matches this name or code\n", sym);
    free(list); free(j);
    return secid[0] != 0;
}

/* 批量行情 JSON → 文本。full = 每只一段四行(quote); 否则一只一行(market) */
char *asst_mkt_parse_quote(const char *json, bool full) {
    char *d = NULL, *diff = NULL, *e = NULL;
    if (!asst_jraw(json, "data", &d) || !asst_jraw(d, "diff", &diff)) { free(d); return NULL; }
    ds4_buf b = {0}; int k = 0;
    for (int i = 0; i < MKT_QUOTE_MAX * 2 && asst_jarr_at(diff, i, &e); i++, free(e), e = NULL) {
        char *name = NULL, *code = NULL; double v, mk = -1, ts = 0;
        asst_jstr(e, "f14", &name); asst_jstr(e, "f12", &code); num(e, "f13", &mk); num(e, "f124", &ts);
        k++;
        const char *cy = full ? currency(mk) : NULL;
        ds4_buf_printf(&b, "%s%s(%s, secid %.0f.%s%s%s)", full && k > 1 ? "\n" : "", name ? name : "?", code ? code : "?", mk, code ? code : "?", cy ? ", 币种 " : "", cy ? cy : "");
        free(name); free(code);
        if (!num(e, "f2", &v)) { ds4_buf_puts(&b, " 无行情(停牌、未上市或已退市)\n"); continue; }
        ds4_buf_puts(&b, full ? "\n  最新 " : " "); px(&b, v);
        double ch = 0, pct = 0;
        if (num(e, "f4", &ch) && num(e, "f3", &pct)) { ds4_buf_puts(&b, full ? "  涨跌 " : " "); ds4_buf_puts(&b, ch >= 0 ? "+" : ""); px(&b, ch); ds4_buf_printf(&b, " (%+.2f%%)", pct); }
        if (full) {
            if (num(e, "f17", &v)) { ds4_buf_puts(&b, "  今开 "); px(&b, v); }
            if (num(e, "f15", &v)) { ds4_buf_puts(&b, "  最高 "); px(&b, v); }
            if (num(e, "f16", &v)) { ds4_buf_puts(&b, "  最低 "); px(&b, v); }
            if (num(e, "f18", &v)) { ds4_buf_puts(&b, "  昨收 "); px(&b, v); }
            ds4_buf_puts(&b, "\n ");
            if (num(e, "f5", &v)) { ds4_buf_puts(&b, " 成交量 "); amt(&b, v); ds4_buf_puts(&b, mk == 0 || mk == 1 ? "手" : "股"); }   /* 沪深一手 100 股; 港美股接口给的就是股 */
            if (num(e, "f6", &v) && v > 0) { ds4_buf_puts(&b, "  成交额 "); amt(&b, v); }
            if (num(e, "f7", &v)) ds4_buf_printf(&b, "  振幅 %.2f%%", v);
            if (num(e, "f8", &v) && v > 0) ds4_buf_printf(&b, "  换手率 %.2f%%", v);
            if (num(e, "f10", &v) && v > 0) ds4_buf_printf(&b, "  量比 %.2f", v);
            ds4_buf_puts(&b, "\n ");
            const size_t mark = b.len;   /* 指数没有估值与市值(接口给 0), 整行不出 */
            if (num(e, "f9", &v) && v != 0) ds4_buf_printf(&b, " 市盈率(动) %.2f", v);
            if (num(e, "f23", &v) && v != 0) ds4_buf_printf(&b, "  市净率 %.2f", v);
            if (num(e, "f20", &v) && v > 0) { ds4_buf_puts(&b, "  总市值 "); amt(&b, v); }
            if (num(e, "f21", &v) && v > 0) { ds4_buf_puts(&b, "  流通市值 "); amt(&b, v); }
            if (b.len == mark) b.len -= 2;
            ds4_buf_puts(&b, full ? "\n" : "");
        }
        double up = 0, dn = 0, fl = 0;   /* 涨跌家数只有沪深指数有; 个股和外盘给 0 */
        const bool breadth = num(e, "f104", &up) && num(e, "f105", &dn) && up + dn > 0;
        if (breadth) {
            ds4_buf_printf(&b, "%s上涨 %.0f 家 · 下跌 %.0f 家", full ? "  " : " · ", up, dn);
            if (num(e, "f106", &fl)) ds4_buf_printf(&b, " · 平盘 %.0f 家", fl);
            if (full) ds4_buf_putc(&b, '\n');
        }
        if (ts > 0) { ds4_buf_puts(&b, full ? "  更新于 " : " · "); bjtime(&b, ts); ds4_buf_puts(&b, " 北京时间"); }
        ds4_buf_putc(&b, '\n');
    }
    free(d); free(diff);
    if (!k) { ds4_buf_free(&b); return NULL; }
    return ds4_buf_take(&b);
}

#define QUOTE_FIELDS "f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f104,f105,f106,f124"
static char *quote(const char *secids, bool full, char *err, size_t errn) {
    ds4_buf u = {0}; ds4_buf_printf(&u, "https://push2.eastmoney.com/api/qt/ulist.np/get?fltt=2&invt=2&secids=%s&fields=" QUOTE_FIELDS, secids);
    char *j = mkt_get(u.ptr, err, errn); ds4_buf_free(&u);
    if (!j) return NULL;
    char *r = asst_mkt_parse_quote(j, full);
    if (!r) { char *x = unexpected("quotes", j); free(j); snprintf(err, errn, "%s", x + 7); free(x); return NULL; }
    free(j); return r;
}

/* K 线 JSON → 区间摘要 + 表。每根 "日期,开,收,高,低,量,额,振幅,涨跌幅,涨跌额,换手率"(fields2 f51~f61, 与交易智能体同) */
char *asst_mkt_parse_kline(const char *json, const char *period_cn) {
    char *d = NULL, *kl = NULL, *e = NULL, *name = NULL, *code = NULL;
    if (!asst_jraw(json, "data", &d) || d[0] != '{' || !asst_jraw(d, "klines", &kl)) { free(d); return NULL; }
    asst_jstr(d, "name", &name); asst_jstr(d, "code", &code);
    double mk = -1; num(d, "market", &mk); const char *cy = currency(mk);
    ds4_buf rows = {0}; int n = 0; double first_prev = 0, last_close = 0, hi = -1, lo = -1; char first_day[16] = "", last_day[16] = "", hi_day[16] = "", lo_day[16] = "";
    for (int i = 0; i < MKT_BARS_MAX + 10 && asst_jarr_at(kl, i, &e); i++, free(e), e = NULL) {
        unq(e);
        char day[16]; double o, c, h, l, vol, am, amp, pct, ch, tr;
        if (sscanf(e, "%15[^,],%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", day, &o, &c, &h, &l, &vol, &am, &amp, &pct, &ch, &tr) != 11) continue;
        if (!n) { first_prev = c - ch; snprintf(first_day, sizeof first_day, "%s", day); }
        snprintf(last_day, sizeof last_day, "%s", day); last_close = c; n++;
        if (hi < 0 || h > hi) { hi = h; snprintf(hi_day, sizeof hi_day, "%s", day); }
        if (lo < 0 || l < lo) { lo = l; snprintf(lo_day, sizeof lo_day, "%s", day); }
        ds4_buf_printf(&rows, "%s | ", day); px(&rows, o); ds4_buf_puts(&rows, " | "); px(&rows, c); ds4_buf_puts(&rows, " | "); px(&rows, h); ds4_buf_puts(&rows, " | "); px(&rows, l);
        ds4_buf_printf(&rows, " | %+.2f%% | ", pct); amt(&rows, vol); ds4_buf_puts(&rows, " | "); amt(&rows, am); ds4_buf_printf(&rows, " | %.2f%%\n", tr);
    }
    free(kl); free(d);
    ds4_buf b = {0};
    ds4_buf_printf(&b, "%s(%s) %s, 前复权%s%s, %d 根", name ? name : "?", code ? code : "?", period_cn, cy ? ", 币种 " : "", cy ? cy : "", n);
    free(name); free(code);
    if (!n) { ds4_buf_puts(&b, ": 没有数据(代码不对, 或还没上市)\n"); ds4_buf_free(&rows); return ds4_buf_take(&b); }
    ds4_buf_printf(&b, " (%s ~ %s)\n区间: 收盘 ", first_day, last_day); px(&b, first_prev); ds4_buf_puts(&b, "(首根前一收) → "); px(&b, last_close);
    if (first_prev > 0) ds4_buf_printf(&b, " (%+.2f%%)", (last_close / first_prev - 1) * 100);
    ds4_buf_puts(&b, "; 最高 "); px(&b, hi); ds4_buf_printf(&b, " (%s); 最低 ", hi_day); px(&b, lo); ds4_buf_printf(&b, " (%s)\n", lo_day);
    ds4_buf_printf(&b, "日期 | 开 | 收 | 高 | 低 | 涨跌幅 | 成交量(%s) | 成交额 | 换手率\n", mk == 0 || mk == 1 ? "手" : "股");
    ds4_buf_append(&b, rows.ptr, rows.len); ds4_buf_free(&rows);
    return ds4_buf_take(&b);
}

/* 主要财务指标(F10 页面同源的 RPT_F10_FINANCE_MAINFINADATA, 一期一行) → 指标 × 报告期 的表(一列一期, 最新在左)。
 * 不用交易智能体那张 RPT_PCF10_FINANCEMAINFINADATA: 它不管 pageSize 都只回最新一期(10-18 实测), 看不了趋势 */
static const struct { const char *key, *label; char kind; } FIN[] = {   /* kind: a = 金额(亿/万), p = 百分数, y = 元 */
    { "TOTALOPERATEREVE", "营业总收入", 'a' }, { "TOTALOPERATEREVETZ", "营收同比", 'p' }, { "DJD_TOI_YOY", "单季营收同比", 'p' },
    { "PARENTNETPROFIT", "归母净利润", 'a' }, { "PARENTNETPROFITTZ", "归母净利同比", 'p' }, { "DJD_DPNP_YOY", "单季归母净利同比", 'p' },
    { "KCFJCXSYJLR", "扣非净利润", 'a' }, { "KCFJCXSYJLRTZ", "扣非净利同比", 'p' },
    { "EPSJB", "基本每股收益(元)", 'y' }, { "BPS", "每股净资产(元)", 'y' }, { "MGJYXJJE", "每股经营现金流(元)", 'y' },
    { "ROEJQ", "净资产收益率(加权)", 'p' }, { "XSMLL", "销售毛利率", 'p' }, { "XSJLL", "销售净利率", 'p' }, { "ZCFZL", "资产负债率", 'p' },
};
#define NFIN (sizeof FIN / sizeof FIN[0])
char *asst_mkt_parse_fin(const char *json) {
    char *res = NULL, *data = NULL;
    if (!asst_jraw(json, "result", &res)) return NULL;
    if (!strcmp(res, "null") || !asst_jraw(res, "data", &data)) { free(res); return strdup("没有财务数据: 只有沪深京 A 股上市公司有(指数、基金、港美股没有)\n"); }
    char *rep[MKT_FIN_MAX] = {0}; int n = 0; char *name = NULL, *cur = NULL;
    while (n < MKT_FIN_MAX && asst_jarr_at(data, n, &rep[n])) n++;
    free(data); free(res);
    if (!n) return strdup("没有财务数据\n");
    asst_jstr(rep[0], "SECURITY_NAME_ABBR", &name); asst_jstr(rep[0], "CURRENCY", &cur);
    ds4_buf b = {0};
    ds4_buf_printf(&b, "%s 主要财务指标(最近 %d 期, 币种 %s; 金额与累计同比都是年初到报告期末的累计数: 一季报 = 1~3 月, 中报 = 1~6 月, 三季报 = 1~9 月; 单季同比只算那一个季度)\n指标",
                   name ? name : "?", n, cur && cur[0] ? cur : "CNY");
    free(name); free(cur);
    for (int i = 0; i < n; i++) { char *t = NULL, *dt = NULL; asst_jstr(rep[i], "REPORT_DATE_NAME", &t); asst_jstr(rep[i], "REPORT_DATE", &dt); ds4_buf_printf(&b, " | %s(%.10s)", t ? t : "?", dt ? dt : "?"); free(t); free(dt); }
    ds4_buf_putc(&b, '\n');
    for (size_t f = 0; f < NFIN; f++) {
        ds4_buf_puts(&b, FIN[f].label);
        for (int i = 0; i < n; i++) {
            double v; ds4_buf_puts(&b, " | ");
            if (!num(rep[i], FIN[f].key, &v)) { ds4_buf_puts(&b, "-"); continue; }
            if (FIN[f].kind == 'p') ds4_buf_printf(&b, "%.2f%%", v); else if (FIN[f].kind == 'y') ds4_buf_printf(&b, "%.3f", v); else amt(&b, v);
        }
        ds4_buf_putc(&b, '\n');
    }
    for (int i = 0; i < n; i++) free(rep[i]);
    return ds4_buf_take(&b);
}

/* 截到 n 字节以内(退到 UTF-8 字符边界) */
static void clip(ds4_buf *b, const char *s, size_t n) {
    size_t k = strlen(s); if (k <= n) { ds4_buf_puts(b, s); return; }
    while (n > 0 && ((unsigned char)s[n] & 0xC0) == 0x80) n--;
    ds4_buf_append(b, s, n); ds4_buf_puts(b, "…");
}

/* 资讯搜索(JSONP: cb(…)) → 一条三行: 时间 | 标题 | 媒体 / 摘要 / 链接 */
char *asst_mkt_parse_news(const char *text) {
    const char *l = strchr(text, '('), *r = strrchr(text, ')');
    if (!l || !r || r < l) return NULL;
    char *j = strndup(l + 1, (size_t)(r - l - 1)), *res = NULL, *arr = NULL, *e = NULL;
    if (!asst_jraw(j, "result", &res) || !asst_jraw(res, "cmsArticleWebOld", &arr)) { free(j); free(res); return NULL; }
    ds4_buf b = {0}; int k = 0;
    for (int i = 0; i < MKT_LIST_MAX && asst_jarr_at(arr, i, &e); i++, free(e), e = NULL) {
        char *dt = NULL, *ti = NULL, *co = NULL, *me = NULL, *u = NULL;
        asst_jstr(e, "date", &dt); asst_jstr(e, "title", &ti); asst_jstr(e, "content", &co); asst_jstr(e, "mediaName", &me); asst_jstr(e, "url", &u);
        if (ti && ti[0]) {
            ds4_buf_printf(&b, "%d. %.16s | %s | %s\n", ++k, dt ? dt : "?", ti, me && me[0] ? me : "?");
            if (co && co[0]) { ds4_buf_puts(&b, "   "); clip(&b, co, MKT_SUMMARY_MAX); ds4_buf_putc(&b, '\n'); }
            if (u && u[0]) ds4_buf_printf(&b, "   %s\n", u);
        }
        free(dt); free(ti); free(co); free(me); free(u);
    }
    free(j); free(res); free(arr);
    if (!k) ds4_buf_puts(&b, "(no articles)\n");
    return ds4_buf_take(&b);
}

/* 公告 JSON → 一条一行: 日期 | 标题 | 栏目 + 原文链接 */
char *asst_mkt_parse_notices(const char *json, const char *code) {
    char *d = NULL, *list = NULL, *e = NULL;
    if (!asst_jraw(json, "data", &d) || !asst_jraw(d, "list", &list)) { free(d); return NULL; }
    ds4_buf b = {0}; int k = 0;
    for (int i = 0; i < MKT_LIST_MAX && asst_jarr_at(list, i, &e); i++, free(e), e = NULL) {
        char *ti = NULL, *dt = NULL, *ac = NULL, *cols = NULL, *c0 = NULL, *cn = NULL;
        asst_jstr(e, "title", &ti); asst_jstr(e, "notice_date", &dt); asst_jstr(e, "art_code", &ac);
        if (asst_jraw(e, "columns", &cols) && asst_jarr_at(cols, 0, &c0)) asst_jstr(c0, "column_name", &cn);
        if (ti && ti[0]) {
            ds4_buf_printf(&b, "%d. %.10s | %s%s%s\n", ++k, dt ? dt : "?", ti, cn && cn[0] ? " | " : "", cn && cn[0] ? cn : "");
            if (ac && ac[0]) ds4_buf_printf(&b, "   https://data.eastmoney.com/notices/detail/%s/%s.html\n", code, ac);
        }
        free(ti); free(dt); free(ac); free(cols); free(c0); free(cn);
    }
    free(d); free(list);
    if (!k) ds4_buf_puts(&b, "(no announcements)\n");
    return ds4_buf_take(&b);
}

/* 主要指数: A 股五个(带涨跌家数) + 恒生 + 美股三大(名字照交易智能体 market crew 的口径) */
static const char MARKET_SECIDS[] = "1.000001,0.399001,0.399006,1.000300,1.000688,100.HSI,100.DJIA,100.NDX,100.SPX";

static bool is_ashare(const char *secid, char *code, size_t n) {   /* 沪深京 6 位代码(财务/公告接口只有它们) */
    if ((secid[0] != '0' && secid[0] != '1') || secid[1] != '.' || strlen(secid + 2) != 6) return false;
    for (const char *p = secid + 2; *p; p++) if (!isdigit((unsigned char)*p)) return false;
    snprintf(code, n, "%s", secid + 2); return true;
}

static long clampl(long v, long def, long max) { return v <= 0 ? def : v > max ? max : v; }

/* 中文逗号/分号/顿号换成 ',' 再切: strtok 的分隔表按字节算, 直接放 "，" 进去会把汉字的 UTF-8 续字节当分隔符, 名字被切碎 */
static char *split_ready(const char *s) {
    static const char *const seps[] = { "，", "；", "、", NULL };
    ds4_buf b = {0};
    for (const char *p = s; *p; ) {
        int hit = 0;
        for (int i = 0; seps[i]; i++) if (!strncmp(p, seps[i], strlen(seps[i]))) { ds4_buf_putc(&b, ','); p += strlen(seps[i]); hit = 1; break; }
        if (!hit) ds4_buf_putc(&b, *p++);
    }
    return ds4_buf_take(&b);
}

char *asst_market_exec(const char *args, int *ok) {
    *ok = 0;
    char *what = NULL, *syms = NULL, *period = NULL; long count = 0;
    asst_jstr(args, "what", &what); asst_jstr(args, "symbols", &syms); asst_jstr(args, "period", &period); asst_jint(args, "count", &count);
    const char *w = what ? what : "";
    char err[512] = ""; ds4_buf log = {0}; char *out = NULL, *j = NULL;
    const bool need_sym = strcmp(w, "market") != 0;
    if (strcmp(w, "search") && strcmp(w, "quote") && strcmp(w, "history") && strcmp(w, "financials") && strcmp(w, "news") && strcmp(w, "notices") && strcmp(w, "market"))
        out = fail("error: what must be search, quote, history, financials, news, notices or market, got %s\n", w);
    else if (need_sym && (!syms || !syms[0])) out = strdup("error: symbols is required: a name (贵州茅台), a code (600519, 00700, AAPL) or a keyword for news\n");
    if (out) goto done;
    if (!strcmp(w, "market")) { out = quote(MARKET_SECIDS, false, err, sizeof err); goto done; }
    if (!strcmp(w, "search")) {
        if (!(j = suggest(syms, err, sizeof err))) goto done;
        char *list = asst_mkt_parse_suggest(j, NULL, 0, NULL, 0);
        if (!list) { out = unexpected("code search", j); goto done; }
        ds4_buf b = {0}; ds4_buf_printf(&b, "# matches for \"%s\" (pass the secid as symbols to be exact)\n%s", syms, list); free(list);
        out = ds4_buf_take(&b); goto done;
    }
    if (!strcmp(w, "news")) {   /* 关键词原样搜: 公司名、代码、"降息"这类话题都行 */
        long c = clampl(count, MKT_LIST_DEF, MKT_LIST_MAX);
        ds4_buf p = {0}; ds4_buf_puts(&p, "{\"uid\":\"\",\"keyword\":"); ds4_json_escape(&p, syms);
        ds4_buf_printf(&p, ",\"type\":[\"cmsArticleWebOld\"],\"client\":\"web\",\"clientType\":\"web\",\"clientVersion\":\"curr\",\"param\":{\"cmsArticleWebOld\":{\"searchScope\":\"default\",\"sort\":\"default\",\"pageIndex\":1,\"pageSize\":%ld,\"preTag\":\"\",\"postTag\":\"\"}}}", c);
        char *enc = web_url_encode(p.ptr); ds4_buf_free(&p);
        ds4_buf u = {0}; ds4_buf_printf(&u, "https://search-api-web.eastmoney.com/search/jsonp?cb=cb&param=%s", enc); free(enc);
        j = mkt_get(u.ptr, err, sizeof err); ds4_buf_free(&u);
        if (!j) goto done;
        char *r = asst_mkt_parse_news(j);
        if (!r) { out = unexpected("news search", j); goto done; }
        ds4_buf b = {0}; ds4_buf_printf(&b, "# finance news for \"%s\" (Eastmoney search, most relevant first; check each date: older items can rank high. Summaries are excerpts, visit_page a link for the full article)\n%s", syms, r); free(r);
        out = ds4_buf_take(&b); goto done;
    }
    /* 其余都要先把标的解析成 secid */
    char secids[MKT_QUOTE_MAX * 24] = ""; int ns = 0;
    char *cp = split_ready(syms);
    char *save = NULL;
    for (char *s = strtok_r(cp, ",;\n", &save); s && ns < MKT_QUOTE_MAX; s = strtok_r(NULL, ",;\n", &save)) {
        while (*s == ' ') s++;
        size_t k = strlen(s); while (k && s[k - 1] == ' ') s[--k] = 0;
        if (!k) continue;
        char id[64] = "";
        if (!resolve(s, id, sizeof id, &log)) continue;
        snprintf(secids + strlen(secids), sizeof secids - strlen(secids), "%s%s", ns ? "," : "", id); ns++;
        if (strcmp(w, "quote")) break;   /* 只有 quote 收多只; 别的取第一只 */
    }
    free(cp);
    if (!ns) { ds4_buf b = {0}; ds4_buf_printf(&b, "error: no symbol could be resolved.\n%s", log.ptr ? log.ptr : ""); out = ds4_buf_take(&b); goto done; }
    if (!strcmp(w, "quote")) { out = quote(secids, true, err, sizeof err); goto done; }
    char code[16];
    if (!strcmp(w, "history")) {
        const char *pe = period && period[0] ? period : "day";
        const int klt = !strcmp(pe, "week") ? 102 : !strcmp(pe, "month") ? 103 : !strcmp(pe, "day") ? 101 : 0;
        if (!klt) { out = fail("error: period must be day, week or month, got %s\n", pe); goto done; }
        ds4_buf u = {0}; ds4_buf_printf(&u, "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=%s&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=%d&fqt=1&end=20500101&lmt=%ld",
                                         secids, klt, clampl(count, MKT_BARS_DEF, MKT_BARS_MAX));
        j = mkt_get(u.ptr, err, sizeof err); ds4_buf_free(&u);
        if (!j) goto done;
        if (!(out = asst_mkt_parse_kline(j, klt == 101 ? "日K" : klt == 102 ? "周K" : "月K"))) out = unexpected("candles", j);
        goto done;
    }
    if (!is_ashare(secids, code, sizeof code)) { out = fail("error: %s is not a Shanghai/Shenzhen/Beijing listed stock: financials and notices cover A-shares only (for Hong Kong / US companies use news or web_search)\n", secids); goto done; }
    if (!strcmp(w, "financials")) {
        const char *ex = secids[0] == '1' ? "SH" : (code[0] == '8' || code[0] == '4' || !strncmp(code, "92", 2)) ? "BJ" : "SZ";
        ds4_buf u = {0}; ds4_buf_printf(&u, "https://datacenter.eastmoney.com/securities/api/data/get?type=RPT_F10_FINANCE_MAINFINADATA&sty=APP_F10_MAINFINADATA&quoteColumns=&filter=(SECUCODE%%3D%%22%s.%s%%22)"
                                         "&p=1&ps=%ld&sr=-1&st=REPORT_DATE&source=HSF10&client=PC", code, ex, clampl(count, MKT_FIN_DEF, MKT_FIN_MAX));
        j = mkt_get(u.ptr, err, sizeof err); ds4_buf_free(&u);
        if (!j) goto done;
        if (!(out = asst_mkt_parse_fin(j))) out = unexpected("financials", j);
        goto done;
    }
    ds4_buf u = {0}; ds4_buf_printf(&u, "https://np-anotice-stock.eastmoney.com/api/security/ann?sr=-1&page_size=%ld&page_index=1&ann_type=A&client_source=web&stock_list=%s&f_node=0&s_node=0", clampl(count, MKT_LIST_DEF, MKT_LIST_MAX), code);
    j = mkt_get(u.ptr, err, sizeof err); ds4_buf_free(&u);
    if (!j) goto done;
    char *r = asst_mkt_parse_notices(j, code);
    if (!r) { out = unexpected("announcements", j); goto done; }
    ds4_buf b = {0}; ds4_buf_printf(&b, "# latest announcements of %s (newest first; open a link with visit_page for the full text)\n%s", code, r); free(r);
    out = ds4_buf_take(&b);
done:
    if (!out) { ds4_buf b = {0}; ds4_buf_printf(&b, "error: %s\n", err[0] ? err : "request failed"); out = ds4_buf_take(&b); }
    else if (log.len && strncmp(out, "error", 5)) { ds4_buf b = {0}; ds4_buf_printf(&b, "%s\nnot resolved (left out above):\n%s", out, log.ptr); free(out); out = ds4_buf_take(&b); }
    *ok = strncmp(out, "error", 5) != 0;
    free(what); free(syms); free(period); free(j); ds4_buf_free(&log);
    return out;
}
