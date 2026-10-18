/* tool_lottery_match.c — lottery_data 的竞彩半边: 在售比赛 + 官方固定奖金(matches)、赛果(results), 足球与篮球(10-18)。
 * 接口是体彩中心 webapi.sporttery.cn/gateway/uniform/: getMatchCalculatorV1(在售, 一次全量)、getUniformMatchResultV1(赛果, 按日期段, 一页 100 场)。
 *   旧路径 jc/football/getMatchResultV1 在页面里 fetch 也回 "禁止访问"(10-18 实测), 只有 uniform 的赛果接口放行。
 * 两个坑(都是 10-18 真返回撞出来的), 改了会怎样写在旁边:
 *   1. 篮球比分接口给的是"客:主"(体彩篮球客队在前: 皇马主场 119:84 胜, finalScore 写 "84:119"), 这里一律翻成"主:客"再给模型;
 *      不翻, 模型会把主队输赢说反。
 *   2. 足球赛果行里的 h/d/a 赔率没注明是哪个玩法的终盘, 不输出; 让球结果用比分和让球数自己算(主队进球 + 让球数 对 客队进球)。
 * 时间都是北京时间(体彩官方就按北京时间排期)。 */
#include "../asst_internal.h"
#include <ctype.h>
#include <time.h>

#define JC_PAGE "https://www.sporttery.cn/"
#define JC_API "https://webapi.sporttery.cn/gateway/uniform/"
#define JC_RESULT_MAX 100    /* 赛果一页最多 100 场(接口认 pageSize=100); 三天的足球赛果 10-18 实测 71 场 */
#define JC_LIST_MAX 120      /* 在售比赛最多列这么多行: 平时 15~40 场, 世界杯/欧洲杯期间多一些 */
#define JC_DAYS_DEF 3        /* 赛果默认看最近三天(含今天): 周末联赛 + 跨夜比赛都在里面 */
#define JC_DAYS_MAX 7
#define BJ_OFFSET (8 * 3600)

static char *jget(const char *obj, const char *key) { char *s = NULL; return asst_jstr(obj, key, &s) ? s : (free(s), (char *)NULL); }

/* 联赛筛选: 逗号分隔(中文逗号/顿号也认), 简称或全称包含任一词就留下。空 = 全要 */
static bool league_ok(const char *filter, const char *abbr, const char *all) {
    if (!filter || !filter[0]) return true;
    const char *p = filter;
    while (*p) {
        const char *e = p; size_t sep = 0;
        while (*e) {
            if (*e == ',' || *e == ' ') { sep = 1; break; }
            if (!strncmp(e, "，", 3) || !strncmp(e, "、", 3)) { sep = 3; break; }
            e++;
        }
        if (e > p) {
            char t[64]; size_t n = (size_t)(e - p) < sizeof t - 1 ? (size_t)(e - p) : sizeof t - 1; memcpy(t, p, n); t[n] = 0;
            if ((abbr && strstr(abbr, t)) || (all && strstr(all, t))) return true;
        }
        if (!*e) break;
        p = e + sep;
    }
    return false;
}

/* 一个玩法对象 {h,d,a,...} → "2.70/3.45/2.14"; 未开售(null/空)回 false */
static bool odds3(ds4_buf *b, const char *pool, const char *k1, const char *k2, const char *k3) {
    char *x = jget(pool, k1), *y = k2 ? jget(pool, k2) : NULL, *z = jget(pool, k3);
    const bool ok = x && x[0] && z && z[0] && (!k2 || (y && y[0]));
    if (ok) { if (k2) ds4_buf_printf(b, "%s/%s/%s", x, y, z); else ds4_buf_printf(b, "%s/%s", x, z); }
    free(x); free(y); free(z); return ok;
}
static char *pool_of(const char *m, const char *key) { char *p = NULL; if (!asst_jraw(m, key, &p) || p[0] != '{' || !strcmp(p, "{}")) { free(p); return NULL; } return p; }

/* 在售比赛 JSON → 一场一行 */
char *asst_lot_parse_jc_matches(const char *json, bool bb, const char *league) {
    char *v = NULL, *days = NULL, *d = NULL, *subs = NULL, *m = NULL, *upd = NULL;
    if (!asst_jraw(json, "value", &v) || !asst_jraw(v, "matchInfoList", &days)) { free(v); return NULL; }
    upd = jget(v, "lastUpdateTime");
    ds4_buf b = {0}; int n = 0, total = 0;
    ds4_buf_printf(&b, "# 竞彩%s 在售比赛(中国体育彩票 sporttery.cn 官方; 北京时间; 赔率是体彩公布的固定奖金倍数(每 1 元可得), 开赛前会变%s%s)\n",
                   bb ? "篮球" : "足球", upd && upd[0] ? ", 更新于 " : "", upd ? upd : "");
    ds4_buf_puts(&b, bb ? "赔率顺序: 胜负 = 主胜/客胜; 让分(主 -4.5 = 主队让 4.5 分) = 主/客; 大小分(总分线) = 大/小\n"
                        : "赔率顺序: 主胜/平/客胜; 让球(+1 = 主队受让一球, -1 = 主队让一球)\n");
    for (int i = 0; i < 20 && asst_jarr_at(days, i, &d); i++, free(d), d = NULL) {
        if (!asst_jraw(d, "subMatchList", &subs)) continue;
        for (int j = 0; j < 200 && asst_jarr_at(subs, j, &m); j++, free(m), m = NULL) {
            char *no = jget(m, "matchNumStr"), *la = jget(m, "leagueAbbName"), *ll = jget(m, "leagueAllName"), *h = jget(m, "homeTeamAllName"), *a = jget(m, "awayTeamAllName"),
                 *hr = jget(m, "homeRank"), *ar = jget(m, "awayRank"), *dt = jget(m, "matchDate"), *tm = jget(m, "matchTime"), *st = jget(m, "matchStatus");
            if (league_ok(league, la, ll)) {
                total++;
                if (n < JC_LIST_MAX) {
                    n++;
                    ds4_buf_printf(&b, "%s %s %s%s(主) vs %s%s(客) · %s %.5s", no ? no : "?", la ? la : "?", h ? h : "?", hr ? hr : "", a ? a : "?", ar ? ar : "", dt ? dt : "?", tm ? tm : "");
                    char *p;
                    if (!bb) {
                        ds4_buf_puts(&b, " · 胜平负 ");
                        if (!((p = pool_of(m, "had")) && odds3(&b, p, "h", "d", "a"))) ds4_buf_puts(&b, "未开售");
                        free(p);
                        if ((p = pool_of(m, "hhad"))) { char *gl = jget(p, "goalLine"); ds4_buf_printf(&b, " · 让球(%s) ", gl ? gl : "?"); if (!odds3(&b, p, "h", "d", "a")) ds4_buf_puts(&b, "未开售"); free(gl); free(p); }
                    } else {
                        ds4_buf_puts(&b, " · 胜负 ");
                        if (!((p = pool_of(m, "mnl")) && odds3(&b, p, "h", NULL, "a"))) ds4_buf_puts(&b, "未开售");
                        free(p);
                        if ((p = pool_of(m, "hdc"))) { char *gl = jget(p, "goalLine"); ds4_buf_printf(&b, " · 让分(主 %s) ", gl ? gl : "?"); if (!odds3(&b, p, "h", NULL, "a")) ds4_buf_puts(&b, "未开售"); free(gl); free(p); }
                        if ((p = pool_of(m, "hilo"))) { char *gl = jget(p, "goalLine"); ds4_buf_printf(&b, " · 大小分 %s ", gl ? gl + (gl[0] == '+') : "?"); if (!odds3(&b, p, "h", NULL, "l")) ds4_buf_puts(&b, "未开售"); free(gl); free(p); }
                    }
                    if (st && strcmp(st, "Selling")) ds4_buf_printf(&b, " · 状态 %s", st);
                    ds4_buf_putc(&b, '\n');
                }
            }
            free(no); free(la); free(ll); free(h); free(a); free(hr); free(ar); free(dt); free(tm); free(st);
        }
        free(subs); subs = NULL;
    }
    free(v); free(days); free(upd);
    if (!total) ds4_buf_puts(&b, league && league[0] ? "(no match of these leagues on sale now; try without league)\n" : "(no match on sale now)\n");
    else if (total > n) ds4_buf_printf(&b, "(%d more not listed; narrow with league)\n", total - n);
    return ds4_buf_take(&b);
}

static bool score2(const char *s, int *x, int *y) { return s && sscanf(s, "%d:%d", x, y) == 2; }
static void outcome(ds4_buf *b, double diff, bool bb, const char *prefix) {   /* diff = 主 - 客(已加让球) */
    ds4_buf_printf(b, "%s%s", prefix, diff > 0 ? "主胜" : diff < 0 ? "客胜" : (bb ? "走盘" : "平"));
}

/* 赛果 JSON → 一场一行, 新的在前 */
char *asst_lot_parse_jc_results(const char *json, bool bb, const char *league) {
    char *v = NULL, *list = NULL, *m = NULL;
    if (!asst_jraw(json, "value", &v) || !asst_jraw(v, "matchResult", &list)) { free(v); return NULL; }
    double tot = 0; { char *t = NULL; if (asst_jraw(v, "total", &t)) tot = atof(t); free(t); }
    ds4_buf b = {0}; int n = 0, k = 0;
    if (bb) ds4_buf_puts(&b, "比分一律 主:客(接口原文是客队在前, 已翻过来); 各节同样主:客\n");
    for (int i = 0; i < JC_RESULT_MAX && asst_jarr_at(list, i, &m); i++, free(m), m = NULL) {
        n++;
        char *no = jget(m, "matchNumStr"), *dt = jget(m, "matchDate"), *la = jget(m, "leagueNameAbbr"), *ll = jget(m, "leagueName"), *h = jget(m, "allHomeTeam"), *a = jget(m, "allAwayTeam"),
             *rs = jget(m, "matchResultStatus"), *gl = jget(m, bb ? "goalline" : "goalLine"), *full = jget(m, bb ? "finalScore" : "sectionsNo999"), *half = bb ? NULL : jget(m, "sectionsNo1");
        if (league_ok(league, la, ll)) {
            k++;
            int x, y;
            ds4_buf_printf(&b, "%s %s %s ", no ? no : "?", dt ? dt : "?", la ? la : "?");
            const bool done = rs && !strcmp(rs, "2") && score2(full, &x, &y);
            if (!done) ds4_buf_printf(&b, "%s vs %s: 还没有赛果\n", h ? h : "?", a ? a : "?");
            else {
                const int hs = bb ? y : x, as = bb ? x : y;   /* 篮球原文客:主 */
                ds4_buf_printf(&b, "%s %d:%d %s", h ? h : "?", hs, as, a ? a : "?");
                if (half && half[0]) ds4_buf_printf(&b, " (半场 %s)", half);
                if (bb) {
                    static const char *const Q[] = { "oneScore", "twoScore", "threeScore", "fourScore", "overtimeScore" };
                    bool first = true;
                    for (int q = 0; q < 5; q++) {
                        char *s = jget(m, Q[q]); int qa, qh;
                        if (score2(s, &qa, &qh)) { ds4_buf_printf(&b, "%s%s%d:%d", first ? " (各节 " : " ", q == 4 ? "加时 " : "", qh, qa); first = false; }
                        free(s);
                    }
                    if (!first) ds4_buf_putc(&b, ')');
                }
                outcome(&b, hs - as, bb, " · ");
                double g;
                if (gl && gl[0] && (g = atof(gl), strchr(gl, '.') || g != 0)) { ds4_buf_printf(&b, " · %s(主 %s) ", bb ? "让分" : "让球", gl); outcome(&b, hs - as + g, bb, bb ? "让分" : "让球"); }
                ds4_buf_putc(&b, '\n');
            }
        }
        free(no); free(dt); free(la); free(ll); free(h); free(a); free(rs); free(gl); free(full); free(half);
    }
    free(v); free(list);
    if (!k) ds4_buf_puts(&b, league && league[0] ? "(no result of these leagues in this date range)\n" : "(no match in this date range)\n");
    if (tot > n) ds4_buf_printf(&b, "(only the newest %d of %.0f matches are listed; ask a shorter date range for the rest)\n", n, tot);
    return ds4_buf_take(&b);
}

/* "YYYY-MM-DD" 校验(要拼进网址) + 往前推 back 天 */
static bool parse_day(const char *s, struct tm *tm) {
    int y, mo, d; char tail;
    if (!s || strlen(s) != 10 || sscanf(s, "%4d-%2d-%2d%c", &y, &mo, &d, &tail) != 3 || mo < 1 || mo > 12 || d < 1 || d > 31) return false;
    memset(tm, 0, sizeof *tm); tm->tm_year = y - 1900; tm->tm_mon = mo - 1; tm->tm_mday = d; tm->tm_hour = 12;   /* 正午: mktime 规整时躲开夏令时跳变 */
    return true;
}

static bool is_basketball(const char *s) { return s && (strstr(s, "basket") || strstr(s, "篮")); }

char *asst_lot_jc(bool results, const char *sport, const char *league, const char *date, long days, char *err, size_t errn) {
    if (sport && sport[0] && !is_basketball(sport) && !strstr(sport, "foot") && !strstr(sport, "soccer") && !strstr(sport, "足"))
        return strdup("error: sport must be football or basketball (竞彩足球 / 竞彩篮球)\n");
    const bool bb = is_basketball(sport);
    ds4_buf u = {0}, b = {0};
    char from[16] = "", to[16] = "";
    if (!results) ds4_buf_printf(&u, JC_API "%s/getMatchCalculatorV1.qry?channel=c&poolCode=%s", bb ? "basketball" : "football", bb ? "mnl,hdc,hilo" : "had,hhad");
    else {
        struct tm tm;
        if (date && date[0]) { if (!parse_day(date, &tm)) return strdup("error: date must look like 2026-10-18\n"); }
        else { time_t t = time(NULL) + BJ_OFFSET; gmtime_r(&t, &tm); tm.tm_hour = 12; }
        const long nd = days <= 0 ? JC_DAYS_DEF : days > JC_DAYS_MAX ? JC_DAYS_MAX : days;
        struct tm e = tm; mktime(&e); strftime(to, sizeof to, "%Y-%m-%d", &e);
        struct tm s = tm; s.tm_mday -= (int)(nd - 1); mktime(&s); strftime(from, sizeof from, "%Y-%m-%d", &s);
        ds4_buf_printf(&u, JC_API "%s/getUniformMatchResultV1.qry?matchBeginDate=%s&matchEndDate=%s&leagueId=&pageSize=%d&pageNo=1&isFix=0&matchPage=1&pcOrWap=1",
                       bb ? "basketball" : "football", from, to, JC_RESULT_MAX);
    }
    char *j = asst_lot_fetch(JC_PAGE, u.ptr, err, errn); ds4_buf_free(&u);
    if (!j) return NULL;
    char *r = results ? asst_lot_parse_jc_results(j, bb, league) : asst_lot_parse_jc_matches(j, bb, league);
    if (!r) { r = asst_lot_unexpected("sporttery.cn", j); free(j); return r; }
    free(j);
    if (!results) return r;
    ds4_buf_printf(&b, "# 竞彩%s 赛果 %s ~ %s(中国体育彩票 sporttery.cn 官方; 北京时间; %s)\n", bb ? "篮球" : "足球", from, to,
                   bb ? "胜负含加时; 让分按主队得分加让分数比" : "比分是 90 分钟(含补时), 不含加时和点球; 让球按主队进球加让球数比");
    ds4_buf_puts(&b, r); free(r);
    return ds4_buf_take(&b);
}
