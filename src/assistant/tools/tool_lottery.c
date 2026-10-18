/* tool_lottery.c — lottery_data: 体彩 / 福彩开奖查询 + 竞彩赛事的分发(10-18 用户: "实现一个彩票 skill 默认加入当前项目的 skill, 先把近期查询封装了, 包含体彩, 福彩, 到各大赛事")。
 * 数据只从两家官方来, 不走百度/500 这类转载站(转载站是 HTML 卡片, 改版就挂, 而且是二手数):
 *   体彩 = 国家体育总局体彩中心 webapi.sporttery.cn: 开奖历史 getHistoryPageListV1(大乐透/排列/七星彩/传统足彩), 竞彩 uniform/{football,basketball}(tool_lottery_match.c)
 *   福彩 = 中彩网 www.cwl.gov.cn: findDrawNotice(双色球/3D/七乐彩/快乐8)
 * 为什么在官网页面里 fetch 而不是直接导航到接口: 两家都有防火墙, 直接打接口(curl 或火狐地址栏)回 "Access Restricted"/空页,
 *   从 www.sporttery.cn / www.cwl.gov.cn 页面里发 fetch 才放行(10-18 spark 实测)。
 * 出错会怎样: 两家都只对中国大陆 IP 放行 —— 机器走海外代理时(Mac 10-18 出口在乌克兰)主页都回 567 "请求已被拦截",
 *   这里报 "refused" 并点明是出口 IP, 模型照实说拿不到, 不去网页里找二手数顶上。 */
#include "../asst_internal.h"
#include <ctype.h>

#define LOT_DRAWS_DEF 5     /* "最近几期"默认: 双色球一周三期, 五期 ≈ 半个月 */
#define LOT_DRAWS_MAX 30    /* 两家接口一页都给得了 30; 看走势够了, 再多正文就几千 token */
#define SPT_PAGE "https://www.sporttery.cn/"
#define CWL_PAGE "https://www.cwl.gov.cn/ygkj/wqkjgg/"
#define SPT_API "https://webapi.sporttery.cn/gateway/"
#define CWL_API "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice?"   /* 别加 systemType=PC(网上抄来的写法): 带上它防火墙直接断连接, fetch 报 NetworkError(10-18 实测) */

/* 在 page 页面里 fetch(api) 拿原文。api 只由本文件拼(常量 + 校验过的数字/日期), 不含用户自由文本 */
static char *lot_fetch_ff(const char *page, const char *api, char *err, size_t errn) {
    ds4_buf js = {0}; ds4_buf_puts(&js, "fetch("); ds4_json_escape(&js, api); ds4_buf_puts(&js, ").then(r=>r.text())");   /* 10-18 探针就是这个写法跑通的; 别加 credentials: 跨域带凭据要服务端回 Allow-Credentials, 没回浏览器直接拒 */
    char *r = asst_web_js(page, js.ptr, true, err, errn);
    ds4_buf_free(&js); return r;
}
static char *(*lot_get)(const char *, const char *, char *, size_t) = lot_fetch_ff;
void asst_lot_set_fetch(char *(*fn)(const char *, const char *, char *, size_t)) { lot_get = fn ? fn : lot_fetch_ff; }
char *asst_lot_fetch(const char *page, const char *api, char *err, size_t errn) { return lot_get(page, api, err, errn); }

/* 解析不了的原文 → 给模型的错误。拦截页单独认出来: 不认的话模型只看到一串 HTML, 会以为接口改版 */
char *asst_lot_unexpected(const char *site, const char *raw) {
    ds4_buf b = {0};
    if (raw && (strstr(raw, "拦截") || strstr(raw, "Access Restricted") || strstr(raw, "禁止访问") || strstr(raw, "访问行为存在异常")))
        ds4_buf_printf(&b, "error: %s refused the request (its firewall answers only mainland-China addresses; this machine may be going out through an overseas proxy or VPN). Say the official data could not be fetched; do not take numbers from other sites instead.\n", site);
    else ds4_buf_printf(&b, "error: unexpected response from %s: %.200s\n", site, raw ? raw : "(empty)");
    return ds4_buf_take(&b);
}

/* "1,234,567.89" → 1234567.89; "---"、空、非数 → false */
bool asst_lot_num(const char *s, double *v) {
    if (!s) return false;
    char t[64]; size_t k = 0;
    for (; *s && k + 1 < sizeof t; s++) if (*s != ',' && *s != ' ') t[k++] = *s;
    t[k] = 0; char *e; *v = strtod(t, &e);
    return k && e != t && *e == 0;
}
static void yuan_big(ds4_buf *b, const char *s) {   /* 销售额 / 奖池: 亿、万 */
    double v; if (!asst_lot_num(s, &v)) { ds4_buf_puts(b, "-"); return; }
    if (v >= 1e8) ds4_buf_printf(b, "%.2f亿元", v / 1e8); else if (v >= 1e4) ds4_buf_printf(b, "%.2f万元", v / 1e4); else ds4_buf_printf(b, "%.0f元", v);
}
static void yuan_exact(ds4_buf *b, const char *s) {   /* 每注奖金照原数写到元(4.5 元这种保留), 47.24万这种四舍五入会让人对不上奖 */
    for (; s && *s; s++) if (*s != ',') ds4_buf_putc(b, *s);
    ds4_buf_puts(b, " 元");
}
static char *jget(const char *obj, const char *key) { char *s = NULL; return asst_jstr(obj, key, &s) ? s : (free(s), (char *)NULL); }

typedef struct { const char *id, *name, *aliases; char src, kind; const char *code; } lot_game;
/* src: T = 体彩, F = 福彩。kind: 开奖号怎么分段 —— D 大乐透 5+2, Q 七星彩 6+1, P 排列(纯数字), S 胜负彩, B 半全场, J 进球彩, s 双色球 6+1, q 七乐彩 7+1, 3 福彩3D, k 快乐8 */
static const lot_game GAMES[] = {
    { "ssq", "双色球", "|ssq|双色球|福彩双色球|", 'F', 's', "ssq" },
    { "fc3d", "福彩3D", "|fc3d|3d|福彩3d|", 'F', '3', "3d" },
    { "qlc", "七乐彩", "|qlc|七乐彩|7乐彩|", 'F', 'q', "qlc" },
    { "kl8", "快乐8", "|kl8|快乐8|快乐八|福彩快乐8|", 'F', 'k', "kl8" },
    { "dlt", "超级大乐透", "|dlt|大乐透|超级大乐透|体彩大乐透|", 'T', 'D', "85" },
    { "pl3", "排列3", "|pl3|排列3|排列三|排三|", 'T', 'P', "35" },
    { "pl5", "排列5", "|pl5|排列5|排列五|排五|", 'T', 'P', "350133" },
    { "qxc", "7星彩", "|qxc|七星彩|7星彩|", 'T', 'Q', "04" },
    { "sfc", "胜负彩(14场 / 任选9场)", "|sfc|胜负彩|足彩|14场|胜负游戏|任九|任选9场|任选九场|", 'T', 'S', "90" },
    { "bqc", "6场半全场", "|bqc|半全场|6场半全场|六场半全场|", 'T', 'B', "98" },
    { "jqc", "4场进球", "|jqc|进球彩|4场进球|四场进球|", 'T', 'J', "94" },
};
#define NGAMES (sizeof GAMES / sizeof GAMES[0])

static const lot_game *find_game(const char *s) {
    if (!s) return NULL;
    char t[96] = "|"; size_t k = 1;
    for (; *s && k + 2 < sizeof t; s++) if (*s != ' ') t[k++] = (char)tolower((unsigned char)*s);
    t[k++] = '|'; t[k] = 0;
    for (size_t i = 0; i < NGAMES; i++) if (strstr(GAMES[i].aliases, t)) return &GAMES[i];
    return NULL;
}

/* 开奖号按玩法分段。体彩号码空格分隔, 福彩逗号分隔 —— 都换成空格 */
static void put_nums(ds4_buf *b, const char *s, int from, int to) {   /* 第 from..to-1 个号(to < 0 = 到底) */
    int i = 0; const char *p = s;
    bool first = true;
    while (*p) {
        while (*p == ' ' || *p == ',') p++;
        if (!*p) break;
        const char *e = p; while (*e && *e != ' ' && *e != ',') e++;
        if (i >= from && (to < 0 || i < to)) { if (!first) ds4_buf_putc(b, ' '); ds4_buf_append(b, p, (size_t)(e - p)); first = false; }
        i++; p = e;
    }
}
static int count_nums(const char *s) { int n = 0; for (const char *p = s; *p; ) { while (*p == ' ' || *p == ',') p++; if (!*p) break; n++; while (*p && *p != ' ' && *p != ',') p++; } return n; }

static void put_result(ds4_buf *b, const lot_game *g, const char *red, const char *blue) {
    switch (g->kind) {
    case 'D': ds4_buf_puts(b, "前区 "); put_nums(b, red, 0, 5); ds4_buf_puts(b, " | 后区 "); put_nums(b, red, 5, -1); break;
    case 'Q': ds4_buf_puts(b, "前六位 "); put_nums(b, red, 0, 6); ds4_buf_puts(b, " | 第七位 "); put_nums(b, red, 6, -1); break;
    case 's': ds4_buf_puts(b, "红球 "); put_nums(b, red, 0, -1); ds4_buf_puts(b, " | 蓝球 "); put_nums(b, blue ? blue : "", 0, -1); break;
    case 'q': ds4_buf_puts(b, "基本号 "); put_nums(b, red, 0, -1); ds4_buf_puts(b, " | 特别号 "); put_nums(b, blue ? blue : "", 0, -1); break;
    case 'k': ds4_buf_puts(b, "开奖号 "); put_nums(b, red, 0, -1); ds4_buf_printf(b, " (%d 个)", count_nums(red)); break;
    case 'S': case 'B': case 'J': ds4_buf_puts(b, "赛果 "); put_nums(b, red, 0, -1); break;
    default: ds4_buf_puts(b, "开奖号 "); put_nums(b, red, 0, -1);
    }
}

static const char *const CN_NUM[] = { "零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十" };

/* 体彩奖级表: prizeLevelList / prizeLevelListRj 一项一行 */
static void spt_prizes(ds4_buf *b, const char *list) {
    char *e = NULL;
    for (int i = 0; i < 40 && asst_jarr_at(list, i, &e); i++, free(e), e = NULL) {
        char *lv = jget(e, "prizeLevel"), *cnt = jget(e, "stakeCount"), *am = jget(e, "stakeAmount");
        double c = 0, a;
        if (lv) {
            ds4_buf_printf(b, "  %s: ", lv);
            if (asst_lot_num(cnt, &c)) ds4_buf_printf(b, "%.0f 注", c); else ds4_buf_puts(b, "-");
            if (c > 0 && asst_lot_num(am, &a)) { ds4_buf_puts(b, ", 每注 "); yuan_exact(b, am); }
            ds4_buf_putc(b, '\n');
        }
        free(lv); free(cnt); free(am);
    }
}

/* 传统足彩的对阵与赛果。result 每场: 胜负彩一位(3 主胜 1 平 0 主负), 半全场两位(半场,全场), 进球彩两位(主队进球,客队进球; 3+ = 三球及以上) */
static void spt_matches(ds4_buf *b, const lot_game *g, const char *list) {
    char *e = NULL;
    ds4_buf_puts(b, g->kind == 'S' ? "对阵(赛果: 3 = 主队胜, 1 = 平, 0 = 主队负):\n" : g->kind == 'B' ? "对阵(赛果两位 = 半场, 全场; 3 = 主队胜, 1 = 平, 0 = 主队负):\n" : "对阵(赛果两位 = 主队进球, 客队进球; 3+ = 三球及以上):\n");
    for (int i = 0; i < 20 && asst_jarr_at(list, i, &e); i++, free(e), e = NULL) {
        char *lg = jget(e, "matchName"), *h = jget(e, "masterTeamAllName"), *a = jget(e, "guestTeamAllName"), *sc = jget(e, "czScore"), *hs = jget(e, "czHalfScore"), *r = jget(e, "result"), *d = jget(e, "startTime");
        ds4_buf_printf(b, "  %d. %s %s %s %s %s", i + 1, d ? d : "", lg ? lg : "", h ? h : "?", sc && sc[0] ? sc : "vs", a ? a : "?");
        if (hs && hs[0]) ds4_buf_printf(b, " (半场 %s)", hs);
        ds4_buf_printf(b, " → %s\n", r && r[0] ? r : "未出");
        free(lg); free(h); free(a); free(sc); free(hs); free(r); free(d);
    }
}

/* 体彩开奖历史 JSON → 文本: 一期一行, 第一期(最新)再带奖级表与足彩对阵 */
char *asst_lot_parse_spt(const char *json, const char *game_id) {
    const lot_game *g = find_game(game_id);
    char *v = NULL, *list = NULL, *e = NULL;
    if (!g || !asst_jraw(json, "value", &v) || !asst_jraw(v, "list", &list)) { free(v); return NULL; }
    ds4_buf b = {0}, top = {0}; int n = 0;
    for (int i = 0; i < LOT_DRAWS_MAX && asst_jarr_at(list, i, &e); i++, free(e), e = NULL) {
        char *no = jget(e, "lotteryDrawNum"), *dt = jget(e, "lotteryDrawTime"), *res = jget(e, "lotteryDrawResult"), *sale = jget(e, "totalSaleAmount"), *pool = jget(e, "poolBalanceAfterdraw"), *saleRj = jget(e, "totalSaleAmountRj");
        double pv = 0;
        ds4_buf_printf(&b, "第 %s 期 %s: ", no ? no : "?", dt ? dt : "?");
        put_result(&b, g, res ? res : "", NULL);
        ds4_buf_puts(&b, " · 销售额 "); yuan_big(&b, sale);
        if (g->kind == 'S' && saleRj && saleRj[0]) { ds4_buf_puts(&b, " · 任选9场销售额 "); yuan_big(&b, saleRj); }
        if (asst_lot_num(pool, &pv) && pv > 0) { ds4_buf_puts(&b, " · 奖池 "); yuan_big(&b, pool); }
        ds4_buf_putc(&b, '\n');
        if (!n) {   /* 最新一期的详情 */
            char *pl = NULL, *rj = NULL, *ml = NULL, *pdf = jget(e, "drawPdfUrl");
            ds4_buf_printf(&top, "\n第 %s 期奖级(每注奖金为税前):\n", no ? no : "?");
            if (asst_jraw(e, "prizeLevelList", &pl)) spt_prizes(&top, pl);
            if (asst_jraw(e, "prizeLevelListRj", &rj) && strcmp(rj, "[]")) spt_prizes(&top, rj);
            if (asst_jraw(e, "matchList", &ml) && strcmp(ml, "[]") && strcmp(ml, "null")) spt_matches(&top, g, ml);
            if (pdf && pdf[0]) ds4_buf_printf(&top, "开奖公告原件: %s\n", pdf);
            free(pl); free(rj); free(ml); free(pdf);
        }
        n++;
        free(no); free(dt); free(res); free(sale); free(pool); free(saleRj);
    }
    free(v); free(list);
    if (!n) { ds4_buf_free(&top); ds4_buf_puts(&b, "(no draws: the issue number may not exist yet)\n"); return ds4_buf_take(&b); }
    ds4_buf_append(&b, top.ptr, top.len); ds4_buf_free(&top);
    return ds4_buf_take(&b);
}

/* 福彩奖级: 双色球/七乐彩 type 是 1..7 等奖; 快乐8 是 "x10z9"(选十中九), 一个"选几"一行 */
static void cwl_prizes(ds4_buf *b, const lot_game *g, const char *list) {
    char *e = NULL; int cur_x = -1; bool any = false;
    for (int i = 0; i < 60 && asst_jarr_at(list, i, &e); i++, free(e), e = NULL) {
        char *ty = jget(e, "type"), *cnt = jget(e, "typenum"), *am = jget(e, "typemoney"), *raw = NULL;
        if (!ty && asst_jraw(e, "type", &raw)) ty = strdup(raw);
        free(raw);
        double c;
        if (ty && asst_lot_num(cnt, &c)) {
            int x = -1, z = -1;
            if (g->kind == 'k' && sscanf(ty, "x%dz%d", &x, &z) == 2 && x >= 1 && x <= 10 && z >= 0 && z <= 10) {
                if (x != cur_x) { ds4_buf_printf(b, "%s  选%s:", any ? "\n" : "", CN_NUM[x]); cur_x = x; }
                ds4_buf_printf(b, " 中%s %.0f 注", CN_NUM[z], c);
                if (c > 0) { ds4_buf_puts(b, " 每注 "); yuan_exact(b, am); }
                ds4_buf_putc(b, ';');
            } else {
                const int lv = atoi(ty);
                if (lv >= 1 && lv <= 10) ds4_buf_printf(b, "  %s等奖: %.0f 注", CN_NUM[lv], c); else ds4_buf_printf(b, "  %s: %.0f 注", ty, c);
                if (c > 0 && am && am[0]) { ds4_buf_puts(b, ", 每注 "); yuan_exact(b, am); }
                ds4_buf_putc(b, '\n');
            }
            any = true;
        }
        free(ty); free(cnt); free(am);
    }
    if (cur_x >= 0) ds4_buf_putc(b, '\n');
    if (!any) ds4_buf_puts(b, "  (官方这一期没给奖级明细: 3D 是固定奖金玩法)\n");
}

/* 中彩网开奖 JSON → 文本 */
char *asst_lot_parse_cwl(const char *json, const char *game_id) {
    const lot_game *g = find_game(game_id);
    char *res = NULL, *e = NULL;
    if (!g || !asst_jraw(json, "result", &res) || res[0] != '[') { free(res); return NULL; }
    ds4_buf b = {0}, top = {0}; int n = 0;
    for (int i = 0; i < LOT_DRAWS_MAX && asst_jarr_at(res, i, &e); i++, free(e), e = NULL) {
        char *no = jget(e, "code"), *dt = jget(e, "date"), *red = jget(e, "red"), *blue = jget(e, "blue"), *sale = jget(e, "sales"), *pool = jget(e, "poolmoney");
        double pv = 0;
        ds4_buf_printf(&b, "第 %s 期 %s: ", no ? no : "?", dt ? dt : "?");
        put_result(&b, g, red ? red : "", blue);
        ds4_buf_puts(&b, " · 销售额 "); yuan_big(&b, sale);
        if (asst_lot_num(pool, &pv) && pv > 0) { ds4_buf_puts(&b, " · 奖池 "); yuan_big(&b, pool); }
        ds4_buf_putc(&b, '\n');
        if (!n) {
            char *pg = NULL, *ct = jget(e, "content"), *dl = jget(e, "detailsLink");
            ds4_buf_printf(&top, "\n第 %s 期奖级(每注奖金为税前):\n", no ? no : "?");
            if (asst_jraw(e, "prizegrades", &pg)) cwl_prizes(&top, g, pg);
            if (ct && ct[0] && strncmp(ct, "共0注", strlen("共0注"))) ds4_buf_printf(&top, "一等奖中奖地: %s\n", ct);   /* 一等奖没人中时官方写"共0注。", 照抄就成了"中奖地: 共0注" */
            if (dl && dl[0]) ds4_buf_printf(&top, "开奖公告: https://www.cwl.gov.cn%s\n", dl);
            free(pg); free(ct); free(dl);
        }
        n++;
        free(no); free(dt); free(red); free(blue); free(sale); free(pool);
    }
    free(res);
    if (!n) { ds4_buf_free(&top); ds4_buf_puts(&b, "(no draws: the issue number may not exist yet)\n"); return ds4_buf_take(&b); }
    ds4_buf_append(&b, top.ptr, top.len); ds4_buf_free(&top);
    return ds4_buf_take(&b);
}

/* 期号: 体彩 5 位(26118), 福彩 7 位(2026119)。用户常把两种混着说, 能换就换; 纯数字以外一律拒(它要拼进网址) */
static bool norm_issue(const char *s, char src, char *out, size_t n) {
    size_t k = strlen(s);
    for (size_t i = 0; i < k; i++) if (!isdigit((unsigned char)s[i])) return false;
    if (src == 'T' && k == 7 && !strncmp(s, "20", 2)) { s += 2; k = 5; }
    if (src == 'F' && k == 5) { snprintf(out, n, "20%s", s); return true; }
    if ((src == 'T' && k != 5) || (src == 'F' && k != 7)) return false;
    snprintf(out, n, "%s", s); return true;
}

static char *draws(const char *game, const char *issue, long count, char *err, size_t errn) {
    const lot_game *g = find_game(game);
    if (!g) {
        ds4_buf b = {0}; ds4_buf_printf(&b, "error: unknown game '%s'. Games: ", game ? game : "");
        for (size_t i = 0; i < NGAMES; i++) ds4_buf_printf(&b, "%s%s %s", i ? ", " : "", GAMES[i].id, GAMES[i].name);
        ds4_buf_puts(&b, "\n"); return ds4_buf_take(&b);
    }
    char iss[16] = "";
    if (issue && issue[0] && !norm_issue(issue, g->src, iss, sizeof iss))
        return strdup(g->src == 'T' ? "error: issue must be the draw number, 5 digits for Sports Lottery games (e.g. 26118)\n" : "error: issue must be the draw number, 7 digits for Welfare Lottery games (e.g. 2026119)\n");
    const long c = iss[0] ? 1 : count <= 0 ? LOT_DRAWS_DEF : count > LOT_DRAWS_MAX ? LOT_DRAWS_MAX : count;
    ds4_buf u = {0};
    if (g->src == 'T') {
        ds4_buf_printf(&u, SPT_API "lottery/getHistoryPageListV1.qry?gameNo=%s&provinceId=0&isVerify=1&pageNo=1&pageSize=%ld", g->code, c);
        if (iss[0]) ds4_buf_printf(&u, "&startTerm=%s&endTerm=%s", iss, iss);
    } else {
        ds4_buf_printf(&u, CWL_API "name=%s", g->code);
        if (iss[0]) ds4_buf_printf(&u, "&issueStart=%s&issueEnd=%s", iss, iss); else ds4_buf_printf(&u, "&issueCount=%ld", c);
    }
    char *j = lot_get(g->src == 'T' ? SPT_PAGE : CWL_PAGE, u.ptr, err, errn); ds4_buf_free(&u);
    if (!j) return NULL;
    char *r = g->src == 'T' ? asst_lot_parse_spt(j, g->id) : asst_lot_parse_cwl(j, g->id);
    if (!r) { r = asst_lot_unexpected(g->src == 'T' ? "sporttery.cn" : "cwl.gov.cn", j); free(j); return r; }
    free(j);
    ds4_buf b = {0};
    ds4_buf_printf(&b, "# %s 开奖(%s官方数据; ", g->name, g->src == 'T' ? "中国体育彩票 sporttery.cn " : "中国福利彩票 cwl.gov.cn ");
    if (iss[0]) ds4_buf_printf(&b, "第 %s 期)\n", iss); else ds4_buf_printf(&b, "最近 %ld 期, 新的在前)\n", c);
    ds4_buf_puts(&b, r); free(r);
    return ds4_buf_take(&b);
}

char *asst_lottery_exec(const char *args, int *ok) {
    *ok = 0;
    char *what = NULL, *game = NULL, *issue = NULL, *sport = NULL, *league = NULL, *date = NULL; long count = 0, days = 0;
    asst_jstr(args, "what", &what); asst_jstr(args, "game", &game); asst_jstr(args, "issue", &issue); asst_jint(args, "count", &count);
    asst_jstr(args, "sport", &sport); asst_jstr(args, "league", &league); asst_jstr(args, "date", &date); asst_jint(args, "days", &days);
    const char *w = what ? what : "";
    char err[512] = ""; char *out = NULL;
    if (!strcmp(w, "draws")) out = (game && game[0]) ? draws(game, issue, count, err, sizeof err) : strdup("error: game is required for draws (ssq, fc3d, qlc, kl8, dlt, pl3, pl5, qxc, sfc, bqc, jqc)\n");
    else if (!strcmp(w, "matches") || !strcmp(w, "results")) out = asst_lot_jc(!strcmp(w, "results"), sport, league, date, days, err, sizeof err);
    else { ds4_buf b = {0}; ds4_buf_printf(&b, "error: what must be draws, matches or results, got %s\n", w); out = ds4_buf_take(&b); }
    if (!out) { ds4_buf b = {0}; ds4_buf_printf(&b, "error: %s\n", err[0] ? err : "request failed"); out = ds4_buf_take(&b); }
    *ok = strncmp(out, "error", 5) != 0;
    free(what); free(game); free(issue); free(sport); free(league); free(date);
    return out;
}
