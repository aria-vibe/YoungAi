/* t_asst_market.c — 行情工具 market_data(tool_market.c)与自带技能默认装上(skill_seed.c)的离线单测(10-18)。
 * 夹具是 10-18 当天东财接口的真返回(截短): 解析器、代码规则、整条执行路(取数换成读夹具, 不上网)、守门、技能账本的五种情形。
 * 真接口验证不在这里(要火狐和外网): ./ds4_asst_test --tool market_data '{"what":"quote","symbols":"贵州茅台,00700,AAPL"}'。
 * 由 t_asst_sbx.c 的 main 在 logic 之后调(助理根目录已建好)。 */
#include "../../src/assistant/asst_internal.h"
#include <sys/stat.h>

static int pass, fail;
#define CHECK(c, ...) do { if (c) pass++; else { fail++; printf("  FAIL market: "); printf(__VA_ARGS__); printf(" (%s:%d)\n", __FILE__, __LINE__); } } while (0)

static const char SUGGEST_MT[] = "{\"QuotationCodeTable\":{\"Data\":[{\"Code\":\"600519\",\"Name\":\"贵州茅台\",\"PinYin\":\"GZMT\",\"Classify\":\"AStock\",\"SecurityTypeName\":\"沪A\",\"MktNum\":\"1\",\"QuoteID\":\"1.600519\"}],\"Status\":0}}";
static const char SUGGEST_AAPL[] = "{\"QuotationCodeTable\":{\"Data\":[{\"Code\":\"AAPL\",\"Name\":\"苹果\",\"SecurityTypeName\":\"美股\",\"QuoteID\":\"105.AAPL\"},{\"Code\":\"AAPL22\",\"Name\":\"Apple Inc Notes 2022\",\"SecurityTypeName\":\"美股\",\"QuoteID\":\"105.AAPL22\"}],\"Status\":0}}";
static const char SUGGEST_NONE[] = "{\"QuotationCodeTable\":{\"Data\":[],\"Status\":0,\"TotalCount\":0}}";
static const char QUOTE[] = "{\"rc\":0,\"data\":{\"total\":3,\"diff\":["
    "{\"f2\":1329.0,\"f3\":1.22,\"f4\":16.03,\"f5\":38502,\"f6\":5075528306.0,\"f7\":2.15,\"f8\":0.31,\"f9\":18.66,\"f10\":1.1,\"f12\":\"600519\",\"f13\":1,\"f14\":\"贵州茅台\",\"f15\":1329.88,\"f16\":1301.68,\"f17\":1311.57,\"f18\":1312.97,\"f20\":1661358447729,\"f21\":1661358447729,\"f23\":6.61,\"f104\":0,\"f105\":0,\"f106\":0,\"f124\":1792138307},"
    "{\"f2\":343.46,\"f3\":0.43,\"f4\":1.46,\"f5\":59946368,\"f6\":20629202944.0,\"f7\":2.15,\"f8\":0.41,\"f9\":38.88,\"f10\":1.56,\"f12\":\"AAPL\",\"f13\":105,\"f14\":\"苹果\",\"f15\":347.82,\"f16\":340.46,\"f17\":346.085,\"f18\":342.0,\"f20\":5012517062800,\"f21\":5012517062800,\"f23\":46.62,\"f124\":1792180800},"
    "{\"f2\":\"-\",\"f3\":\"-\",\"f12\":\"600001\",\"f13\":1,\"f14\":\"邯郸钢铁\",\"f124\":0}]}}";
static const char INDEX[] = "{\"rc\":0,\"data\":{\"total\":2,\"diff\":[{\"f2\":3716.28,\"f3\":0.24,\"f4\":8.8,\"f9\":0.0,\"f20\":0,\"f12\":\"000001\",\"f13\":1,\"f14\":\"上证指数\",\"f104\":1398,\"f105\":902,\"f106\":57,\"f124\":1792138320},"
    "{\"f2\":52002.41,\"f3\":-0.34,\"f4\":-175.28,\"f12\":\"DJIA\",\"f13\":100,\"f14\":\"道琼斯\",\"f104\":0,\"f105\":0,\"f106\":0,\"f124\":1792180784}]}}";
static const char KLINE[] = "{\"rc\":0,\"data\":{\"code\":\"600519\",\"market\":1,\"name\":\"贵州茅台\",\"klines\":["
    "\"2026-10-14,1270.10,1286.00,1286.75,1266.00,23773,3045608010.00,1.63,0.87,11.11,0.19\","
    "\"2026-10-15,1286.00,1312.97,1317.76,1282.00,57677,7537706224.00,2.78,2.10,26.97,0.46\","
    "\"2026-10-16,1311.57,1329.00,1329.88,1301.68,38502,5075528306.00,2.15,1.22,16.03,0.31\"]}}";
static const char FIN[] = "{\"result\":{\"pages\":35,\"data\":["
    "{\"SECUCODE\":\"300750.SZ\",\"SECURITY_NAME_ABBR\":\"宁德时代\",\"REPORT_DATE\":\"2026-06-30 00:00:00\",\"REPORT_DATE_NAME\":\"2026中报\",\"CURRENCY\":\"CNY\",\"EPSJB\":9.51,\"BPS\":81.993,\"MGJYXJJE\":13.015,\"TOTALOPERATEREVE\":276917000000,\"TOTALOPERATEREVETZ\":54.8,\"DJD_TOI_YOY\":56.92,\"PARENTNETPROFIT\":43284000000,\"PARENTNETPROFITTZ\":41.98,\"DJD_DPNP_YOY\":36.46,\"KCFJCXSYJLR\":39013000000,\"KCFJCXSYJLRTZ\":43.44,\"ROEJQ\":12.08,\"XSMLL\":23.93,\"XSJLL\":16.98,\"ZCFZL\":63.65},"
    "{\"SECUCODE\":\"300750.SZ\",\"SECURITY_NAME_ABBR\":\"宁德时代\",\"REPORT_DATE\":\"2026-03-31 00:00:00\",\"REPORT_DATE_NAME\":\"2026一季报\",\"CURRENCY\":\"CNY\",\"EPSJB\":4.58,\"TOTALOPERATEREVE\":129131000000,\"TOTALOPERATEREVETZ\":52.45,\"PARENTNETPROFIT\":20738000000,\"ROEJQ\":5.98,\"XSJLL\":null}],\"count\":2},\"success\":true}";
static const char NEWS[] = "cb({\"code\":0,\"result\":{\"cmsArticleWebOld\":[{\"date\":\"2026-10-15 17:42:00\",\"title\":\"白酒归来！贵州茅台重回1300元\",\"content\":\"10月15日，白酒板块走强。\",\"mediaName\":\"证券时报\",\"url\":\"http://finance.eastmoney.com/a/1.html\"},"
    "{\"date\":\"2026-10-15 22:08:00\",\"title\":\"贵州茅台10月15日大宗交易成交407.02万元\",\"content\":\"\",\"mediaName\":\"\",\"url\":\"\"}]}})";
static const char NOTICES[] = "{\"data\":{\"list\":[{\"art_code\":\"AN202608141827994403\",\"columns\":[{\"column_code\":\"001002002\",\"column_name\":\"半年度报告摘要\"}],\"notice_date\":\"2026-08-15 00:00:00\",\"title\":\"贵州茅台:贵州茅台2026年半年度报告摘要\"}],\"total_hits\":1},\"success\":1}";

static char last_url[2048];
static char *fake_get(const char *url, char *err, size_t errn) {   /* 按地址认接口, 回夹具 */
    snprintf(last_url, sizeof last_url, "%s", url);
    if (strstr(url, "suggest/get")) return strdup(strstr(url, "AAPL") ? SUGGEST_AAPL : strstr(url, "xyz") ? SUGGEST_NONE : SUGGEST_MT);
    if (strstr(url, "ulist.np")) return strdup(strstr(url, "100.DJIA") ? INDEX : QUOTE);
    if (strstr(url, "kline/get")) return strdup(KLINE);
    if (strstr(url, "RPT_F10_FINANCE_MAINFINADATA")) return strdup(FIN);
    if (strstr(url, "search-api-web")) return strdup(NEWS);
    if (strstr(url, "np-anotice")) return strdup(NOTICES);
    snprintf(err, errn, "no fixture for %s", url); return NULL;
}

static void test_secid(void) {
    static const char *const cases[][2] = { { "600519", "1.600519" }, { "000001", "0.000001" }, { "300750", "0.300750" }, { "510300", "1.510300" }, { "830799", "0.830799" },
                                            { "sh000001", "1.000001" }, { "SZ399006", "0.399006" }, { "bj830799", "0.830799" }, { "105.AAPL", "105.AAPL" }, { " 116.00700", "116.00700" } };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) { char o[64] = ""; CHECK(asst_mkt_secid(cases[i][0], o, sizeof o) && !strcmp(o, cases[i][1]), "secid %s → %s, got %s", cases[i][0], cases[i][1], o); }
    static const char *const names[] = { "贵州茅台", "AAPL", "00700", "60051", "sh60051x", "a.600519", "" };
    for (size_t i = 0; i < sizeof names / sizeof names[0]; i++) { char o[64] = ""; CHECK(!asst_mkt_secid(names[i], o, sizeof o), "'%s' goes to code search", names[i]); }
}

static void test_parsers(void) {
    char first[64], nm[64];
    char *r = asst_mkt_parse_suggest(SUGGEST_AAPL, first, sizeof first, nm, sizeof nm);
    CHECK(r && strstr(r, "1. 苹果 AAPL · 美股 · secid 105.AAPL\n2. Apple Inc Notes 2022") && !strcmp(first, "105.AAPL") && !strcmp(nm, "苹果"), "suggest list + first: %s", r ? r : "(null)"); free(r);
    r = asst_mkt_parse_suggest(SUGGEST_NONE, first, sizeof first, NULL, 0); CHECK(r && strstr(r, "(no match)") && !first[0], "suggest empty"); free(r);
    CHECK(!asst_mkt_parse_suggest("<html>风控</html>", NULL, 0, NULL, 0), "suggest garbage → NULL");

    r = asst_mkt_parse_quote(QUOTE, true);
    CHECK(r && strstr(r, "贵州茅台(600519, secid 1.600519, 币种 人民币)\n  最新 1329.00  涨跌 +16.03 (+1.22%)  今开 1311.57  最高 1329.88  最低 1301.68  昨收 1312.97"), "quote A-share line: %s", r ? r : "(null)");
    CHECK(r && strstr(r, "成交量 3.85万手  成交额 50.76亿") && strstr(r, "市盈率(动) 18.66  市净率 6.61  总市值 16613.58亿"), "quote volume in 手, money in 亿");
    CHECK(r && strstr(r, "更新于 2026-10-16 16:11 北京时间"), "quote time is Beijing time: %s", r ? r : "");
    CHECK(r && strstr(r, "苹果(AAPL, secid 105.AAPL, 币种 美元)") && strstr(r, "今开 346.085") && strstr(r, "万股"), "US quote: dollars, shares, 3rd decimal kept");
    CHECK(r && strstr(r, "邯郸钢铁(600001, secid 1.600001, 币种 人民币) 无行情"), "suspended / delisted says so instead of 0");
    CHECK(r && !strstr(r, "上涨"), "no breadth for single stocks"); free(r);
    r = asst_mkt_parse_quote(INDEX, false);
    CHECK(r && strstr(r, "上证指数(000001, secid 1.000001) 3716.28 +8.80 (+0.24%) · 上涨 1398 家 · 下跌 902 家 · 平盘 57 家 · 2026-10-16 16:12 北京时间\n"), "index one-liner with breadth: %s", r ? r : "(null)");
    CHECK(r && strstr(r, "道琼斯(DJIA, secid 100.DJIA) 52002.41 -175.28 (-0.34%) · 2026-10-17 03:59") && !strstr(r, "币种"), "foreign index: no breadth, no currency"); free(r);
    CHECK(!asst_mkt_parse_quote("{\"rc\":102,\"data\":null}", true), "quote without data → NULL");

    r = asst_mkt_parse_kline(KLINE, "日K");
    CHECK(r && strstr(r, "贵州茅台(600519) 日K, 前复权, 币种 人民币, 3 根 (2026-10-14 ~ 2026-10-16)\n"), "kline header: %s", r ? r : "(null)");
    CHECK(r && strstr(r, "区间: 收盘 1274.89(首根前一收) → 1329.00 (+4.24%); 最高 1329.88 (2026-10-16); 最低 1266.00 (2026-10-14)"), "kline period summary computed: %s", r ? r : "");
    CHECK(r && strstr(r, "成交量(手)") && strstr(r, "2026-10-15 | 1286.00 | 1312.97 | 1317.76 | 1282.00 | +2.10% | 5.77万 | 75.38亿 | 0.46%"), "kline row"); free(r);
    r = asst_mkt_parse_kline("{\"data\":{\"code\":\"1\",\"name\":\"x\",\"klines\":[]}}", "周K"); CHECK(r && strstr(r, "没有数据"), "kline empty says so"); free(r);
    CHECK(!asst_mkt_parse_kline("{\"rc\":0,\"data\":null}", "日K"), "kline null data → NULL");

    r = asst_mkt_parse_fin(FIN);
    CHECK(r && strstr(r, "宁德时代 主要财务指标(最近 2 期, 币种 CNY;") && strstr(r, "指标 | 2026中报(2026-06-30) | 2026一季报(2026-03-31)\n"), "fin header: %s", r ? r : "(null)");
    CHECK(r && strstr(r, "营业总收入 | 2769.17亿 | 1291.31亿\n") && strstr(r, "单季营收同比 | 56.92% | -\n") && strstr(r, "销售净利率 | 16.98% | -\n") && strstr(r, "基本每股收益(元) | 9.510 | 4.580"), "fin rows, missing/null → -"); free(r);
    r = asst_mkt_parse_fin("{\"version\":null,\"result\":null,\"success\":false,\"message\":\"返回数据为空\",\"code\":9201}"); CHECK(r && strstr(r, "只有沪深京 A 股"), "fin empty explains coverage"); free(r);

    r = asst_mkt_parse_news(NEWS);
    CHECK(r && strstr(r, "1. 2026-10-15 17:42 | 白酒归来！贵州茅台重回1300元 | 证券时报\n   10月15日，白酒板块走强。\n   http://finance.eastmoney.com/a/1.html\n2. 2026-10-15 22:08 | 贵州茅台10月15日大宗交易成交407.02万元 | ?\n"), "news lines: %s", r ? r : "(null)"); free(r);
    CHECK(!asst_mkt_parse_news("{\"code\":400}"), "news without jsonp wrapper → NULL");
    r = asst_mkt_parse_notices(NOTICES, "600519");
    CHECK(r && strstr(r, "1. 2026-08-15 | 贵州茅台:贵州茅台2026年半年度报告摘要 | 半年度报告摘要\n   https://data.eastmoney.com/notices/detail/600519/AN202608141827994403.html\n"), "notices: %s", r ? r : "(null)"); free(r);
}

static char *run(const char *args, int *ok) { return asst_market_exec(args, ok); }

static void test_exec(void) {
    asst_mkt_set_fetch(fake_get);
    int ok = 0; char *r;
    r = run("{\"what\":\"quote\",\"symbols\":\"贵州茅台，AAPL、不存在xyz\"}", &ok);   /* 中文逗号/顿号要能切, 汉字不能被切碎 */
    CHECK(ok && strstr(r, "secid 1.600519") && strstr(r, "not resolved (left out above):\n不存在xyz: no stock"), "quote: names resolved, the unknown one listed: %s", r);
    CHECK(strstr(last_url, "secids=1.600519,105.AAPL&") != NULL, "one batch request for all symbols: %s", last_url); free(r);
    r = run("{\"what\":\"history\",\"symbols\":\"600519\",\"period\":\"week\",\"count\":9999}", &ok);
    CHECK(ok && strstr(last_url, "secid=1.600519") && strstr(last_url, "klt=102") && strstr(last_url, "lmt=250"), "history: week, count clamped to 250: %s", last_url); free(r);
    r = run("{\"what\":\"history\",\"symbols\":\"600519\",\"period\":\"hour\"}", &ok); CHECK(!ok && strstr(r, "period must be day, week or month"), "bad period refused"); free(r);
    r = run("{\"what\":\"financials\",\"symbols\":\"300750\"}", &ok);
    CHECK(ok && strstr(last_url, "SECUCODE%3D%22300750.SZ%22") && strstr(last_url, "ps=4"), "financials: SZ secucode, 4 periods: %s", last_url); free(r);
    r = run("{\"what\":\"financials\",\"symbols\":\"bj830799\"}", &ok); CHECK(strstr(last_url, "830799.BJ") != NULL, "financials: BJ secucode: %s", last_url); free(r);
    r = run("{\"what\":\"financials\",\"symbols\":\"AAPL\"}", &ok); CHECK(!ok && strstr(r, "A-shares only"), "financials of a US stock refused with a pointer: %s", r); free(r);
    r = run("{\"what\":\"notices\",\"symbols\":\"贵州茅台\",\"count\":3}", &ok); CHECK(ok && strstr(last_url, "stock_list=600519") && strstr(last_url, "page_size=3") && strstr(r, "半年度报告摘要"), "notices via name: %s", last_url); free(r);
    r = run("{\"what\":\"news\",\"symbols\":\"贵州茅台\"}", &ok);
    CHECK(ok && strstr(r, "most relevant first") && strstr(last_url, "sort%22%3A%22default") && strstr(last_url, "pageSize%22%3A10"), "news: relevance sort, 10 items: %s", last_url); free(r);
    r = run("{\"what\":\"market\"}", &ok); CHECK(ok && strstr(r, "上涨 1398 家") && strstr(last_url, "1.000001,0.399001"), "market: index batch"); free(r);
    r = run("{\"what\":\"search\",\"symbols\":\"AAPL\"}", &ok); CHECK(ok && strstr(r, "# matches for \"AAPL\"") && strstr(r, "105.AAPL22"), "search lists all matches"); free(r);
    r = run("{\"what\":\"price\",\"symbols\":\"x\"}", &ok); CHECK(!ok && strstr(r, "what must be search, quote"), "unknown what refused"); free(r);
    r = run("{\"what\":\"quote\"}", &ok); CHECK(!ok && strstr(r, "symbols is required"), "quote without symbols refused"); free(r);
    r = run("{\"what\":\"quote\",\"symbols\":\"xyz\"}", &ok); CHECK(!ok && strstr(r, "no symbol could be resolved"), "nothing resolved → error"); free(r);
    asst_mkt_set_fetch(NULL);
    /* 守门: 只读查询放行并置网页污点; 像卡号的拒 */
    asst_gres g; asst_guard("abcdef123456", "market_data", "{\"what\":\"quote\",\"symbols\":\"600519\"}", &g);
    CHECK(g.dec == ASST_ALLOW && !strcmp(g.cls, "web") && !strcmp(g.target, "market data"), "guard allows a quote as web: %d %s", (int)g.dec, g.cls);
    asst_guard("abcdef123456", "market_data", "{\"what\":\"news\",\"symbols\":\"4111111111111111\"}", &g); CHECK(g.dec == ASST_DENY, "guard refuses a card-number-like keyword");
    char *tj = asst_tools_json(NULL);   /* 这里的助理根没装 finance: 专属工具不该出现(装了之后的有无在 test_seed 里验) */
    CHECK(!strstr(tj, "\"name\":\"market_data\""), "no finance skill → no market_data");
    free(tj);
}

/* 自带技能默认装上: 装 / 补记账 / 没改过跟着升级 / 改过不动 / 删了不装回来; 仓库里真的 skills/ 两个技能都过脚本检查 */
static void put(const char *p, const char *s) { FILE *f = fopen(p, "w"); if (f) { fputs(s, f); fclose(f); } }
static char *rd(const char *p) { return asst_slurp(p, NULL); }
static void test_seed(const char *base) {
    char saved_root[ASST_PATH], saved_sk[ASST_PATH]; snprintf(saved_root, sizeof saved_root, "%s", asst_root); snprintf(saved_sk, sizeof saved_sk, "%s", asst_skills);
    char src[ASST_PATH], p[ASST_PATH + 64], msg[1024];
    snprintf(asst_root, sizeof asst_root, "%s/seedroot", base); mkdir(asst_root, 0700);
    snprintf(asst_skills, sizeof asst_skills, "%s/skills", asst_root); mkdir(asst_skills, 0700);
    snprintf(src, sizeof src, "%s/seedsrc", base); mkdir(src, 0700);
    snprintf(p, sizeof p, "%s/demo", src); mkdir(p, 0700);
    snprintf(p, sizeof p, "%s/demo/SKILL.md", src); put(p, "---\nname: demo\ndescription: v1\n---\n# Demo\n");
    snprintf(p, sizeof p, "%s/demo/ref", src); mkdir(p, 0700);
    snprintf(p, sizeof p, "%s/demo/ref/a.md", src); put(p, "ref v1\n");
    snprintf(p, sizeof p, "%s/bad", src); mkdir(p, 0700);
    snprintf(p, sizeof p, "%s/bad/SKILL.md", src); put(p, "# Bad\n\n1. bash x.sh\n");
    CHECK(asst_skills_seed(src, msg, sizeof msg) == 1 && strstr(msg, "installed demo") && strstr(msg, "bad/SKILL.md skipped"), "first start installs demo, refuses the scripted one: %s", msg);
    snprintf(p, sizeof p, "%s/demo/ref/a.md", asst_skills); char *t = rd(p); CHECK(t && !strcmp(t, "ref v1\n"), "reference file copied"); free(t);
    snprintf(p, sizeof p, "%s/bad", asst_skills); struct stat st; CHECK(stat(p, &st) != 0, "refused skill leaves no folder");
    CHECK(asst_skills_seed(src, msg, sizeof msg) == 0, "second start does nothing: %s", msg);
    /* 开关: 关掉 → 工具表没有 use_skill(只有这一个技能)、点名拒、页面 JSON enabled:false; 关掉不算"用户改过", 仓库新版照样升级且升级后仍关着 */
    CHECK(asst_skill_op("{\"op\":\"enable\",\"name\":\"demo\",\"on\":false}", msg, sizeof msg) && !asst_skill_enabled("demo"), "switch demo off: %s", msg);
    ds4_buf sb = {0}; CHECK(!asst_skill_build_tool(NULL, &sb), "off skill not offered to the model: %s", sb.ptr ? sb.ptr : ""); ds4_buf_free(&sb);
    char *off = asst_skill_load("demo"); CHECK(off && strstr(off, "turned off"), "use_skill refuses an off skill: %s", off ? off : "(null)"); free(off);
    asst_skills_json(&sb); CHECK(sb.ptr && strstr(sb.ptr, "\"name\":\"demo\"") && strstr(sb.ptr, "\"enabled\":false"), "page sees enabled:false"); ds4_buf_free(&sb);
    CHECK(!asst_skill_op("{\"op\":\"enable\",\"name\":\"nope\",\"on\":true}", msg, sizeof msg) && strstr(msg, "no such skill"), "switch on unknown skill rejected: %s", msg);
    snprintf(p, sizeof p, "%s/demo/SKILL.md", src); put(p, "---\nname: demo\ndescription: v2\n---\n# Demo\n");
    CHECK(asst_skills_seed(src, msg, sizeof msg) == 1 && strstr(msg, "updated demo"), "untouched copy follows the repo: %s", msg);
    CHECK(!asst_skill_enabled("demo"), "upgrade keeps the skill switched off");
    CHECK(asst_skill_op("{\"op\":\"enable\",\"name\":\"demo\",\"on\":true}", msg, sizeof msg) && asst_skill_build_tool(NULL, &sb) && strstr(sb.ptr, "- demo: v2"), "switch back on: listed again"); ds4_buf_free(&sb);
    snprintf(p, sizeof p, "%s/demo/SKILL.md", asst_skills); t = rd(p); CHECK(t && strstr(t, "v2"), "v2 on disk"); free(t);
    put(p, "---\nname: demo\ndescription: mine\n---\n# Demo\n");   /* 用户在技能页改过 */
    snprintf(p, sizeof p, "%s/demo/SKILL.md", src); put(p, "---\nname: demo\ndescription: v3\n---\n# Demo\n");
    CHECK(asst_skills_seed(src, msg, sizeof msg) == 0, "user-edited copy is left alone");
    snprintf(p, sizeof p, "%s/demo/SKILL.md", asst_skills); t = rd(p); CHECK(t && strstr(t, "mine"), "user's edit kept"); free(t);
    CHECK(asst_skill_op("{\"op\":\"delete\",\"name\":\"demo\"}", msg, sizeof msg), "user deletes demo");
    CHECK(asst_skills_seed(src, msg, sizeof msg) == 0, "deleted skill is not brought back");
    snprintf(p, sizeof p, "%s/demo", asst_skills); CHECK(stat(p, &st) != 0, "still deleted");
    /* 仓库里真的 skills/(测试从仓库根跑): pptx 与 finance 都装得进来, use_skill 列出 finance */
    CHECK(asst_skills_seed("skills", msg, sizeof msg) >= 2 && strstr(msg, "installed finance") && strstr(msg, "installed pptx") && !strstr(msg, "skipped"), "repo skills pass the script check: %s", msg);
    ds4_buf b = {0}; CHECK(asst_skill_build_tool(NULL, &b) && strstr(b.ptr, "- finance: 股票、指数、基金"), "use_skill lists finance"); ds4_buf_free(&b);
    char *body = asst_skill_load("finance"); CHECK(body && strstr(body, "market_data") && strstr(body, "## 按问题选查法"), "finance body loads"); free(body);
    /* 专属工具跟着技能开关走: 装着且开着 → 有火狐就列; 关掉 → 工具表没有、硬调拒(不出网); 再开 → 回来 */
    const bool ff = asst_browser_kind()[0] != 0;
    char *tj = asst_tools_json(NULL); CHECK((strstr(tj, "\"name\":\"market_data\"") != NULL) == ff, "finance on → market_data offered iff Firefox"); free(tj);
    CHECK(asst_skill_set_enabled("finance", false), "switch finance off");
    tj = asst_tools_json(NULL); CHECK(!strstr(tj, "\"name\":\"market_data\""), "finance off → market_data gone from the tool list"); free(tj);
    asst_gres g = {0}; int ok = 1; body = asst_tool_exec("abcdef123456", "market_data", "{\"what\":\"market\"}", &g, &ok);
    CHECK(!ok && body && strstr(body, "turned off"), "finance off → calling market_data anyway is refused: %s", body ? body : "(null)"); free(body);
    CHECK(asst_skill_set_enabled("finance", true), "switch finance back on");
    tj = asst_tools_json(NULL); CHECK((strstr(tj, "\"name\":\"market_data\"") != NULL) == ff, "finance on again → market_data back"); free(tj);
    snprintf(asst_root, sizeof asst_root, "%s", saved_root); snprintf(asst_skills, sizeof asst_skills, "%s", saved_sk);
}

int asst_test_market(const char *base) {
    pass = fail = 0;
    test_secid(); test_parsers(); test_exec(); test_seed(base);
    printf("asst market: %d passed, %d failed\n", pass, fail);
    return fail;
}
