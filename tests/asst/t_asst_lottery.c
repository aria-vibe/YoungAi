/* t_asst_lottery.c — 彩票工具 lottery_data(tool_lottery.c / tool_lottery_match.c)与自带技能 lottery 的离线单测(10-18)。
 * 夹具 tests/asst/fixtures/lottery/ 下的 json 是 10-18 晚 spark 上从体彩中心、中彩网官方接口取的真返回(用 jq 删了用不到的字段和多余场次)。
 * 真接口验证不在这里(要火狐和中国大陆出口): ./ds4_asst_test --tool lottery_data '{"what":"draws","game":"ssq"}'。
 * 由 t_asst_sbx.c 的 main 调, 从仓库根跑(夹具按相对路径读)。 */
#include "../../src/assistant/asst_internal.h"
#include <sys/stat.h>

static int pass, fail;
#define CHECK(c, ...) do { if (c) pass++; else { fail++; printf("  FAIL lottery: "); printf(__VA_ARGS__); printf(" (%s:%d)\n", __FILE__, __LINE__); } } while (0)

static char *fx(const char *name) { char p[256]; snprintf(p, sizeof p, "tests/asst/fixtures/lottery/%s", name); return asst_slurp(p, NULL); }

static void test_draws(void) {
    char *j = fx("spt_dlt.json"), *r = j ? asst_lot_parse_spt(j, "大乐透") : NULL;
    CHECK(r && strstr(r, "第 26112 期 2026-09-30: 前区 01 03 05 10 22 | 后区 03 09 · 销售额 3.06亿元 · 奖池 7.99亿元\n第 26111 期"), "dlt line split 5+2: %s", r ? r : "(null)");
    CHECK(r && strstr(r, "\n第 26112 期奖级(每注奖金为税前):\n  一等奖: 6 注, 每注 8047843 元\n  一等奖(追加): 0 注\n"), "dlt prize table, exact yuan, no amount for 0 winners: %s", r ? r : "");
    CHECK(r && strstr(r, "开奖公告原件: https://pdf.sporttery.cn/"), "dlt pdf link"); free(r); free(j);

    j = fx("spt_sfc.json"); r = j ? asst_lot_parse_spt(j, "任九") : NULL;
    CHECK(r && strstr(r, "第 26144 期 2026-10-18: 赛果 0 3 3 0 3 0 1 3 3 0 3 1 0 1 · 销售额 2719.45万元 · 任选9场销售额 2199.27万元\n"), "sfc line with 任九 sales: %s", r ? r : "(null)");
    CHECK(r && strstr(r, "  任选9场: 11592 注, 每注 1214 元\n") && strstr(r, "对阵(赛果: 3 = 主队胜, 1 = 平, 0 = 主队负):\n  1. 2026-10-17 英超 布伦特福德 1:2 利物浦 (半场 1:0) → 0\n"), "sfc 任九 prize + matches with full team names: %s", r ? r : ""); free(r); free(j);

    j = fx("spt_jqc.json"); r = j ? asst_lot_parse_spt(j, "jqc") : NULL;
    CHECK(r && strstr(r, "赛果两位 = 主队进球, 客队进球") && strstr(r, "利物浦 (半场 1:0) → 1,2"), "jqc legend + per-match goals: %s", r ? r : "(null)"); free(r); free(j);
    j = fx("spt_qxc.json"); r = j ? asst_lot_parse_spt(j, "七星彩") : NULL;
    CHECK(r && strstr(r, "前六位 5 2 7 0 1 0 | 第七位 11 · 销售额 1594.52万元 · 奖池 5853.85万元") && strstr(r, "  二等奖: 2 注, 每注 238596 元"), "qxc 6+1: %s", r ? r : "(null)"); free(r); free(j);

    j = fx("cwl_ssq.json"); r = j ? asst_lot_parse_cwl(j, "ssq") : NULL;
    CHECK(r && strstr(r, "第 2026119 期 2026-10-18(日): 红球 12 22 23 25 27 30 | 蓝球 04 · 销售额 3.64亿元 · 奖池 10.25亿元\n第 2026118 期 2026-10-15(四): 红球 05 21 22 23 28 31 | 蓝球 14"), "ssq lines: %s", r ? r : "(null)");
    CHECK(r && strstr(r, "  一等奖: 2 注, 每注 10000000 元\n  二等奖: 61 注, 每注 472442 元\n") && !strstr(r, "七等奖") && strstr(r, "一等奖中奖地: 内蒙古1注，湖南1注，共2注。\n开奖公告: https://www.cwl.gov.cn/c/2026/10/18/670173.shtml"), "ssq prizes (empty level 7 skipped), winners' provinces, link: %s", r ? r : ""); free(r); free(j);
    j = fx("cwl_3d.json"); r = j ? asst_lot_parse_cwl(j, "福彩3D") : NULL;
    CHECK(r && strstr(r, "第 2026277 期 2026-10-18(日): 开奖号 5 6 8 · 销售额 1.03亿元\n") && strstr(r, "固定奖金玩法"), "3d: digits, no pool, fixed-prize note: %s", r ? r : "(null)"); free(r); free(j);
    j = fx("cwl_kl8.json"); r = j ? asst_lot_parse_cwl(j, "kl8") : NULL;
    CHECK(r && strstr(r, "(20 个)") && strstr(r, "  选十: 中十 0 注; 中九 15 注 每注 8000 元;") && strstr(r, "  选一: 中一 51927 注 每注 4.5 元;\n"), "kl8 prizes grouped by pick count: %s", r ? r : "(null)"); free(r); free(j);

    CHECK(!asst_lot_parse_spt("<html>请求已被拦截</html>", "dlt") && !asst_lot_parse_cwl("{\"state\":1}", "ssq"), "garbage → NULL");
    r = asst_lot_unexpected("sporttery.cn", "<title>请求已被拦截</title>"); CHECK(strstr(r, "refused the request") && strstr(r, "mainland-China"), "firewall page recognised: %s", r); free(r);
    double v; CHECK(asst_lot_num("1,234,567.89", &v) && v > 1234567.8 && v < 1234567.9 && !asst_lot_num("---", &v) && !asst_lot_num("", &v), "number parse with commas");
}

static void test_jc(void) {
    char *j = fx("jc_fb_matches.json"), *r = j ? asst_lot_parse_jc_matches(j, false, NULL) : NULL;
    CHECK(r && strstr(r, "周日018 英超 诺丁汉森林[英超17](主) vs 阿森纳[英超2](客) · 2026-10-18 23:30 · 胜平负 7.20/4.20/1.33 · 让球(+1) 2.70/3.45/2.14\n"), "football match line: %s", r ? r : "(null)");
    CHECK(r && strstr(r, "周日024 西甲 皇家马德里") && strstr(r, "胜平负 未开售 · 让球(-2) 2.31/3.90/2.27"), "单关 only handicap: 胜平负 shows 未开售: %s", r ? r : ""); free(r);
    r = j ? asst_lot_parse_jc_matches(j, false, "英超，意甲") : NULL;
    CHECK(r && strstr(r, "诺丁汉森林") && strstr(r, "AC米兰") && strstr(r, "热刺") && !strstr(r, "皇家马德里"), "league filter with Chinese comma: %s", r ? r : "(null)"); free(r);
    r = j ? asst_lot_parse_jc_matches(j, false, "法甲") : NULL; CHECK(r && strstr(r, "(no match of these leagues on sale now"), "league filter empty says so"); free(r); free(j);

    j = fx("jc_bb_matches.json"); r = j ? asst_lot_parse_jc_matches(j, true, NULL) : NULL;
    CHECK(r && strstr(r, "周日307 西篮联 特内里费[西篮联9](主) vs 巴斯克尼亚[西篮联5](客) · 2026-10-19 00:00 · 胜负 1.36/2.25 · 让分(主 -4.5) 1.65/1.75 · 大小分 174.5 1.74/1.66\n"), "basketball match line: %s", r ? r : "(null)"); free(r); free(j);

    j = fx("jc_fb_results.json"); r = j ? asst_lot_parse_jc_results(j, false, NULL) : NULL;
    CHECK(r && strstr(r, "周日016 2026-10-18 荷甲 格罗宁根 vs 阿贾克斯: 还没有赛果\n"), "unfinished: %s", r ? r : "(null)");
    CHECK(r && strstr(r, "周日007 2026-10-18 瑞超 天狼星 2:2 埃尔夫斯堡 (半场 0:1) · 平 · 让球(主 -1) 让球客胜\n"), "draw + handicap computed: %s", r ? r : "");
    CHECK(r && strstr(r, "拜仁慕尼黑 6:1 莱比锡红牛 (半场 3:0) · 主胜 · 让球(主 -2) 让球主胜") && strstr(r, "(only the newest 5 of 71 matches"), "big win -2, total note: %s", r ? r : ""); free(r); free(j);

    j = fx("jc_bb_results.json"); r = j ? asst_lot_parse_jc_results(j, true, NULL) : NULL;
    CHECK(r && strstr(r, "周日305 2026-10-18 西篮联 皇家马德里 119:84 毕尔巴鄂 (各节 36:13 26:22 27:22 30:27) · 主胜 · 让分(主 -11.50) 让分主胜\n"), "basketball score flipped to home:away: %s", r ? r : "(null)");
    CHECK(r && strstr(r, "墨尔本联 89:91 伊拉瓦拉老鹰") && strstr(r, "· 客胜 · 让分(主 -10.50) 让分客胜"), "away win stays away: %s", r ? r : ""); free(r); free(j);
}

static const char *fx_name;   /* 下一次取数回哪份夹具 */
static char last_page[256], last_api[1024];
static char *fake_get(const char *page, const char *api, char *err, size_t errn) {
    snprintf(last_page, sizeof last_page, "%s", page); snprintf(last_api, sizeof last_api, "%s", api);
    if (!fx_name) { snprintf(err, errn, "no fixture"); return NULL; }
    if (!strcmp(fx_name, "blocked")) return strdup("<!doctype html><title>请求已被拦截</title>");
    return fx(fx_name);
}
static char *run(const char *fixture, const char *args, int *ok) { fx_name = fixture; return asst_lottery_exec(args, ok); }

static void test_exec(void) {
    asst_lot_set_fetch(fake_get);
    int ok = 0; char *r;
    r = run("cwl_ssq.json", "{\"what\":\"draws\",\"game\":\"双色球\"}", &ok);
    CHECK(ok && strstr(r, "# 双色球 开奖(中国福利彩票 cwl.gov.cn 官方数据; 最近 5 期, 新的在前)\n") && strstr(last_page, "www.cwl.gov.cn") && strstr(last_api, "name=ssq&issueCount=5"), "ssq default 5 via cwl page: %s | %s", r, last_api); free(r);
    r = run("spt_dlt.json", "{\"what\":\"draws\",\"game\":\"dlt\",\"issue\":\"2026112\"}", &ok);
    CHECK(ok && strstr(last_api, "gameNo=85") && strstr(last_api, "&startTerm=26112&endTerm=26112") && strstr(last_page, "sporttery.cn") && strstr(r, "第 26112 期)"), "dlt issue 7→5 digits: %s", last_api); free(r);
    r = run("cwl_ssq.json", "{\"what\":\"draws\",\"game\":\"ssq\",\"issue\":\"26119\",\"count\":9}", &ok);
    CHECK(ok && strstr(last_api, "issueStart=2026119&issueEnd=2026119") && !strstr(last_api, "issueCount"), "ssq issue 5→7 digits, count ignored: %s", last_api); free(r);
    r = run("spt_dlt.json", "{\"what\":\"draws\",\"game\":\"pl5\",\"count\":99}", &ok); CHECK(strstr(last_api, "gameNo=350133") && strstr(last_api, "pageSize=30"), "count clamped to 30: %s", last_api); free(r);
    r = run(NULL, "{\"what\":\"draws\",\"game\":\"ssq\",\"issue\":\"2026&x=1\"}", &ok); CHECK(!ok && strstr(r, "7 digits"), "non-digit issue refused before any request: %s", r); free(r);
    r = run(NULL, "{\"what\":\"draws\",\"game\":\"六合彩\"}", &ok); CHECK(!ok && strstr(r, "unknown game") && strstr(r, "ssq 双色球") && strstr(r, "jqc 4场进球"), "unknown game lists games: %s", r); free(r);
    r = run(NULL, "{\"what\":\"draws\"}", &ok); CHECK(!ok && strstr(r, "game is required"), "draws needs game"); free(r);
    r = run("blocked", "{\"what\":\"draws\",\"game\":\"kl8\"}", &ok); CHECK(!ok && strstr(r, "cwl.gov.cn refused"), "blocked page → error naming the site: %s", r); free(r);
    r = run("jc_fb_matches.json", "{\"what\":\"matches\",\"league\":\"英超\"}", &ok);
    CHECK(ok && strstr(last_api, "uniform/football/getMatchCalculatorV1.qry?channel=c&poolCode=had,hhad") && strstr(r, "热刺") && !strstr(r, "AC米兰"), "matches default football: %s", last_api); free(r);
    r = run("jc_bb_matches.json", "{\"what\":\"matches\",\"sport\":\"篮球\"}", &ok); CHECK(ok && strstr(last_api, "basketball/getMatchCalculatorV1.qry?channel=c&poolCode=mnl,hdc,hilo"), "basketball matches"); free(r);
    r = run("jc_fb_results.json", "{\"what\":\"results\",\"date\":\"2026-10-01\",\"days\":3}", &ok);
    CHECK(ok && strstr(last_api, "matchBeginDate=2026-09-29&matchEndDate=2026-10-01&") && strstr(last_api, "pageSize=100") && strstr(r, "# 竞彩足球 赛果 2026-09-29 ~ 2026-10-01"), "results range crosses month: %s", last_api); free(r);
    r = run("jc_bb_results.json", "{\"what\":\"results\",\"sport\":\"basketball\",\"days\":30}", &ok); CHECK(ok && strstr(last_api, "uniform/basketball/getUniformMatchResultV1"), "basketball results"); free(r);
    r = run(NULL, "{\"what\":\"results\",\"date\":\"10/18\"}", &ok); CHECK(!ok && strstr(r, "date must look like"), "bad date refused"); free(r);
    r = run(NULL, "{\"what\":\"matches\",\"sport\":\"tennis\"}", &ok); CHECK(!ok && strstr(r, "football or basketball"), "unknown sport refused"); free(r);
    r = run(NULL, "{\"what\":\"odds\"}", &ok); CHECK(!ok && strstr(r, "draws, matches or results"), "unknown what refused"); free(r);
    asst_lot_set_fetch(NULL);
    asst_gres g; asst_guard("abcdef123456", "lottery_data", "{\"what\":\"draws\",\"game\":\"ssq\"}", &g);
    CHECK(g.dec == ASST_ALLOW && !strcmp(g.cls, "web") && !strcmp(g.target, "lottery data"), "guard allows as web: %d %s", (int)g.dec, g.cls);
    char *tj = asst_tools_json(NULL);
    const bool listed = strstr(tj, "\"name\":\"lottery_data\"") && strstr(tj, "\"enum\":[\"draws\",\"matches\",\"results\"]");
    CHECK(listed == (asst_browser_kind()[0] != 0), "lottery_data offered iff Firefox is installed");
    free(tj);
}

/* 仓库 skills/lottery 过脚本检查、装得进、use_skill 列得出。科研技能 skills/research(10-19, 没有专属工具所以也住这里) 同口:
 * 正文里有数据源网址、pdftotext、统计检验表, 随附 reference/templates.md, 都不许被脚本检查当成命令拒掉 */
static void test_skill(const char *base) {
    char saved_root[ASST_PATH], saved_sk[ASST_PATH], msg[1024]; snprintf(saved_root, sizeof saved_root, "%s", asst_root); snprintf(saved_sk, sizeof saved_sk, "%s", asst_skills);
    snprintf(asst_root, sizeof asst_root, "%s/lotroot", base); mkdir(asst_root, 0700);
    snprintf(asst_skills, sizeof asst_skills, "%s/skills", asst_root); mkdir(asst_skills, 0700);
    CHECK(asst_skills_seed("skills", msg, sizeof msg) >= 4 && strstr(msg, "installed lottery") && strstr(msg, "installed research") && !strstr(msg, "skipped"), "repo skills incl. lottery and research pass the script check: %s", msg);
    ds4_buf b = {0}; CHECK(asst_skill_build_tool(NULL, &b) && strstr(b.ptr, "- lottery: ") && strstr(b.ptr, "- research: 科研、学术、论文相关的事用这个"), "use_skill lists lottery and research"); ds4_buf_free(&b);
    char *body = asst_skill_load("lottery"); CHECK(body && strstr(body, "lottery_data"), "lottery body loads"); free(body);
    body = asst_skill_load("research");
    CHECK(body && strstr(body, "  reference/templates.md (") && strstr(body, "---\n# 科研助理\n") && strstr(body, "api.openalex.org") && strstr(body, "pdftotext") && strstr(body, "Mann-Whitney"), "research body loads with its template listed: %.200s", body ? body : "(null)"); free(body);
    snprintf(asst_root, sizeof asst_root, "%s", saved_root); snprintf(asst_skills, sizeof asst_skills, "%s", saved_sk);
}

int asst_test_lottery(const char *base) {
    pass = fail = 0;
    test_draws(); test_jc(); test_exec(); test_skill(base);
    printf("asst lottery: %d passed, %d failed\n", pass, fail);
    return fail;
}
