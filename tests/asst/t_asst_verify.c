/* t_asst_verify.c — 校验相(asst_verify.c)+ view_file(tool_view.c)单测: 账本记账 / run 前后扫描 / 收口事实消息的触发与上限 / 点名缺失 / 五类文件的"看"。
 * 由 t_asst_sbx.c 的 main 调用(助理根目录与工作区已建好); 网页那段要火狐, 没有就 SKIP 不 FAIL。 */
#include "../../src/assistant/asst_verify.h"
#include "../../src/image/ds4_image.h"
#include <sys/stat.h>
#include <unistd.h>

static int fails, passes, skips;
#define CHECK(cond, ...) do { if (cond) passes++; else { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void put(const char *rel, const char *s) { char p[ASST_PATH * 2]; snprintf(p, sizeof p, "%s/%s", asst_ws, rel); char *sl = strrchr(p, '/'); if (sl) { *sl = 0; asst_mkdir_p(p); *sl = '/'; } FILE *f = fopen(p, "w"); if (f) { fputs(s, f); fclose(f); } }
static asst_gres gres(const char *rel, bool rw) { asst_gres g; memset(&g, 0, sizeof g); snprintf(g.path, sizeof g.path, "%s/%s", asst_ws, rel); g.rw = rw; snprintf(g.cls, sizeof g.cls, "%s", rw ? "write" : "read"); return g; }
static const asst_ledger_item *item(const asst_ledger *l, const char *rel) { for (int i = 0; i < l->n; i++) if (!strcmp(l->it[i].rel, rel)) return &l->it[i]; return NULL; }

static void test_ledger(void) {
    asst_ledger l; asst_ledger_init(&l);
    put("a.html", "<p>x</p>");
    asst_gres g = gres("a.html", true);
    asst_ledger_note(&l, "write_file", &g, "created a.html", true);
    CHECK(l.n == 1 && l.step == 1 && item(&l, "a.html") && item(&l, "a.html")->wrote_at == 1 && !strcmp(item(&l, "a.html")->by, "write_file") && item(&l, "a.html")->size == 8, "write_file enters the ledger: n=%d", l.n);
    asst_gres r = gres("a.html", false);
    asst_ledger_note(&l, "read_file", &r, "1: <p>x</p>", true);
    CHECK(item(&l, "a.html")->seen_at == 2, "read_file marks it looked at: seen_at=%d", item(&l, "a.html")->seen_at);
    asst_ledger_note(&l, "write_file", &g, "replaced a.html", true);
    CHECK(item(&l, "a.html")->wrote_at == 3 && item(&l, "a.html")->seen_at == 2 && l.n == 1, "rewrite updates the same entry: wrote_at=%d", item(&l, "a.html")->wrote_at);
    asst_ledger_note(&l, "write_file", &g, "error: nope", false);
    CHECK(l.step == 4 && item(&l, "a.html")->wrote_at == 3, "a failed call counts but changes nothing");
    asst_gres img = gres("inbox/src.jpg", false);
    asst_ledger_note(&l, "edit_image", &img, "image: out/src-edited.png\nimage: .img/src.mask.png\nedited\n", true);
    CHECK(l.n == 2 && item(&l, "out/src-edited.png") && !item(&l, ".img/src.mask.png") && !item(&l, "inbox/src.jpg"), "edit_image: product from the image: lines, mask in .img/ and the source are not products: n=%d", l.n);
    asst_gres outside = gres("../outside.txt", true);
    asst_ledger_note(&l, "write_file", &outside, "created", true);
    CHECK(l.n == 2, "a product outside the workspace is not booked");
    asst_gres dot = gres(".web/page.md", true);
    asst_ledger_note(&l, "write_file", &dot, "created", true);
    CHECK(l.n == 2 && l.ntool == 3 && l.tool[0].count == 5 && !strcmp(l.tool[0].name, "write_file"), "dot folders are not booked; tally write_file=%d ntool=%d", l.tool[0].count, l.ntool);
}

static void test_scan(void) {
    put("keep.txt", "same"); put("old.txt", "v1"); put(".trash/junk.txt", "x");
    void *before = asst_ledger_snap();
    usleep(20000);
    put("old.txt", "v2-longer"); put("d/new.json", "{}"); put(".view/shot.png", "p");
    asst_ledger l; asst_ledger_init(&l); l.step = 5;
    asst_ledger_diff(&l, before, "run");
    CHECK(l.n == 2 && item(&l, "old.txt") && item(&l, "d/new.json") && !item(&l, "keep.txt") && !item(&l, ".view/shot.png"), "scan diff: changed + new, not unchanged, not dot dirs: n=%d", l.n);
    CHECK(item(&l, "d/new.json")->wrote_at == 5 && !strcmp(item(&l, "d/new.json")->by, "run") && item(&l, "old.txt")->size == 9, "diff entries carry step, tool and size");
}

static void test_note(void) {
    asst_ledger l; asst_ledger_init(&l);
    char *n = asst_verify_note(&l, "Redis 流是一种日志结构。");
    CHECK(n == NULL, "no tool call → no check (plain answers are not interrupted)"); free(n);
    put("p.html", "<p>hi</p>");
    asst_gres g = gres("p.html", true);
    asst_ledger_note(&l, "write_file", &g, "created", true);
    asst_ledger_note(&l, "web_search", &g, "results", true);
    n = asst_verify_note(&l, "写好了, 见 p.html");
    CHECK(n && strstr(n, "check 1 of 2") && strstr(n, "2 tool calls (write_file 1, web_search 1)") && strstr(n, "p.html (9 B, write_file, not looked at since)") && strstr(n, "view_file") && !strstr(n, "last check"), "first check lists tally and products: %s", n ? n : "(null)");
    CHECK(l.rounds == 1 && l.noted_step == 2, "round and step recorded");
    free(n);
    n = asst_verify_note(&l, "写好了, 见 p.html");
    CHECK(n == NULL, "nothing changed since the check → the run may end"); free(n);
    asst_gres r = gres("p.html", false);
    asst_ledger_note(&l, "view_file", &r, "image: .view/p.html.0s.png\n", true);
    asst_gres e = gres("p.html", true);
    asst_ledger_note(&l, "edit_file", &e, "edited", true);
    n = asst_verify_note(&l, "改好了");
    CHECK(n && strstr(n, "check 2 of 2") && strstr(n, "changed since the last check") && strstr(n, "not looked at since") && strstr(n, "last check in this run"), "second check after a change, marked last: %s", n ? n : "(null)");
    free(n);
    asst_ledger_note(&l, "edit_file", &e, "edited", true);
    n = asst_verify_note(&l, "再改了一次");
    CHECK(n == NULL, "after %d checks the run ends no matter what", ASST_VERIFY_ROUNDS); free(n);
    /* 看过之后的措辞 */
    asst_ledger l2; asst_ledger_init(&l2);
    asst_ledger_note(&l2, "write_file", &g, "created", true);
    asst_ledger_note(&l2, "view_file", &r, "image: x\n", true);
    n = asst_verify_note(&l2, "好了");
    CHECK(n && strstr(n, "looked at after that"), "looked-at wording: %s", n ? n : "(null)"); free(n);
    /* 点名却不存在: 没动手也要核 */
    asst_ledger l3; asst_ledger_init(&l3);
    n = asst_verify_note(&l3, "报告在 report.pdf, 数据见 data/out.csv; 参考 bing.com 与 v1.0 和 p.html。");
    CHECK(n && strstr(n, "Named in your answer but not in the workspace: report.pdf, data/out.csv.") && !strstr(n, "bing.com") && !strstr(n, "p.html"), "missing mentions, existing and non-file tokens left out: %s", n ? n : "(null)"); free(n);
    char *m = asst_file_mentioned_missing("见 `out/鹈鹕.html` 和 sandbox:/mnt/data/x.png");
    CHECK(m && strstr(m, "out/") && strstr(m, "/mnt/data/x.png"), "sandbox: prefix stripped, missing paths listed: %s", m ? m : "(null)"); free(m);
}

static char *view(const char *sid, const char *args, int *ok) {
    asst_gres g; asst_guard(sid, "view_file", args, &g);
    *ok = 0;
    if (g.dec != ASST_ALLOW) return strdup("guard refused");
    char trash[ASST_PATH + 16]; snprintf(trash, sizeof trash, "%s/.trash", asst_ws);
    return asst_tool_exec(sid, "view_file", args, &g, ok);
}

static void test_view(void) {
    char sid[ASST_ID_LEN + 1]; asst_session_new(sid, "v", "校验", 0);
    asst_gres g; asst_guard(sid, "view_file", "{\"path\":\"notes.txt\"}", &g);
    CHECK(g.dec == ASST_ALLOW && !strcmp(g.cls, "read"), "view_file is a read-class tool: %d %s", g.dec, g.cls);
    asst_tctx tc = { "", "", NULL, false }; char *tools = asst_tools_json(&tc);
    CHECK(strstr(tools, "\"name\":\"view_file\"") && strstr(tools, "the way the user will see it"), "view_file is in the tool list"); free(tools);
    int ok = 0; char *r;
    put("notes.txt", "line one\nline two\n第三行\n");
    r = view(sid, "{\"path\":\"notes.txt\"}", &ok);
    CHECK(ok && strstr(r, "notes.txt: 3 lines") && strstr(r, "第三行"), "text file: head with line count: %.120s", r); free(r);
    r = view(sid, "{\"path\":\"nope.txt\"}", &ok);
    CHECK(!ok && strstr(r, "no such file"), "missing file: %s", r); free(r);
    const uint8_t px[12] = { 255, 0, 0, 0, 255, 0, 0, 0, 255, 9, 9, 9 };
    char pp[ASST_PATH + 16]; snprintf(pp, sizeof pp, "%s/pic.png", asst_ws); img_png_write(pp, 2, 2, 3, px);
    r = view(sid, "{\"path\":\"pic.png\"}", &ok);
    CHECK(ok && !strncmp(r, "image: pic.png\n", 15) && strstr(r, "png 2x2"), "image: picture back + dimensions: %.80s", r); free(r);
    /* 真写一份 xlsx 与 docx(沙盒里走官方产出工具), 再看回正文 */
    char trash[ASST_PATH + 16]; snprintf(trash, sizeof trash, "%s/.trash", asst_ws);
    const char *xa = "{\"path\":\"sales.xlsx\",\"sheets\":[{\"name\":\"S\",\"columns\":[\"Month\",\"Revenue\"],\"rows\":[[\"Jan\",120],[\"Feb\",80],[\"Total\",\"=SUM(B2:B3)\"]]}]}";
    asst_gres gx; asst_guard(sid, "write_xlsx", xa, &gx); int wok = 0; char *w = asst_office_exec("write_xlsx", xa, &gx, trash, &wok);
    CHECK(wok, "write_xlsx fixture: %.100s", w); free(w);
    r = view(sid, "{\"path\":\"sales.xlsx\"}", &ok);
    CHECK(ok && strstr(r, "[sheet 1]") && strstr(r, "A1=Month") && strstr(r, "B2=120") && strstr(r, "B4=200 {=SUM(B2:B3)}"), "xlsx: cells with shared strings and formula cache: %.300s", r); free(r);
    const char *da = "{\"path\":\"r.docx\",\"markdown\":\"# 三季度经营分析\\n\\n毛利 **160** 万, 同比增长。\\n\\n| 区域 | 收入 |\\n|---|---|\\n| 华东 | 510 |\\n\"}";
    asst_gres gd; asst_guard(sid, "write_document", da, &gd); w = asst_office_exec("write_document", da, &gd, trash, &wok);
    CHECK(wok, "write_document fixture: %.100s", w); free(w);
    r = view(sid, "{\"path\":\"r.docx\"}", &ok);
    CHECK(ok && strstr(r, "三季度经营分析") && strstr(r, "毛利 160 万") && strstr(r, "华东\t510"), "docx: paragraphs and table cells as text: %.300s", r); free(r);
    const char *pa = "{\"path\":\"deck.pptx\",\"markdown\":\"# 开场\\n\\n一句话\\n\\n---\\n\\n# 第二页\\n\\n- 要点甲\\n- 要点乙\\n\"}";
    asst_gres gp; asst_guard(sid, "write_document", pa, &gp); w = asst_office_exec("write_document", pa, &gp, trash, &wok);
    CHECK(wok, "write_document pptx fixture: %.100s", w); free(w);
    r = view(sid, "{\"path\":\"deck.pptx\"}", &ok);
    CHECK(ok && strstr(r, "[slide 1]") && strstr(r, "开场") && strstr(r, "[slide 2]") && strstr(r, "要点乙"), "pptx: per slide text: %.300s", r); free(r);
    put("bad.docx", "not a zip");
    r = view(sid, "{\"path\":\"bad.docx\"}", &ok);
    CHECK(!ok && strstr(r, "not a readable Office package"), "corrupt docx: %s", r); free(r);
    r = view(sid, "{\"path\":\"pic.png\",\"width\":10}", &ok);
    CHECK(!ok && strstr(r, "between 64 and 4096"), "bad viewport refused: %s", r); free(r);
    /* 网页: 要火狐 */
    if (!asst_browser_kind()[0]) { skips++; printf("SKIP view_file web (no browser)\n"); return; }
    put("site/style.css", "body { background: #123456; } .box { width: 100px; height: 100px; background: red; animation: move 2s linear infinite; } @keyframes move { from { transform: translateX(0); } to { transform: translateX(400px); } }");
    put("site/anim.html", "<!doctype html><link rel=stylesheet href=style.css><link rel=stylesheet href=\"https://fonts.googleapis.com/css2?family=Inter\"><img src=missing.png><div class=box></div><script>wheel.spin();</script>");
    r = view(sid, "{\"path\":\"site/anim.html\"}", &ok);
    CHECK(ok && !strncmp(r, "image: .view/site_anim.html.0s.png\nimage: .view/site_anim.html.1s.png\n", 70) && strstr(r, "picture at 0 s, picture at 1 s."), "animated page: two pictures 1 s apart, named by relative path: %.220s", r);
    CHECK(r && strstr(r, "Page errors reported by the browser") && strstr(r, "wheel"), "script error reported as a fact");
    CHECK(r && strstr(r, "failed to load") && strstr(r, "missing.png") && strstr(r, "fonts.googleapis.com"), "missing same-dir resource and blocked external URL listed: %s", r ? strstr(r, "failed") : "(null)");
    free(r);
    char shot[ASST_PATH + 32]; snprintf(shot, sizeof shot, "%s/.view/site_anim.html.1s.png", asst_ws); struct stat st;
    CHECK(!stat(shot, &st) && st.st_size > 100, "screenshots land in .view/");
    put("site/still.html", "<!doctype html><body style='background:#fff'><h1>静止</h1></body>");
    r = view(sid, "{\"path\":\"site/still.html\"}", &ok);
    CHECK(ok && !strncmp(r, "image: .view/site_still.html.0s.png\n", 36) && !strstr(r, ".1s.png") && strstr(r, "1 s identical to 0 s") && strstr(r, "Nothing moves") && strstr(r, "No page errors"), "static page: one picture, no errors: %.220s", r); free(r);
    r = view(sid, "{\"path\":\"site/anim.html\",\"times\":[2,0.5,0,0.5]}", &ok);   /* 乱序带重复 → 升序去重 0 / 0.5 / 2 */
    CHECK(ok && strstr(r, "image: .view/site_anim.html.0s.png\nimage: .view/site_anim.html.0.5s.png\nimage: .view/site_anim.html.2s.png\n") && strstr(r, "picture at 0 s, picture at 0.5 s, picture at 2 s."), "times: sorted, deduplicated, one picture each: %.300s", r); free(r);
    r = view(sid, "{\"path\":\"site/anim.html\",\"times\":[0,1,2,3,4]}", &ok);
    CHECK(!ok && strstr(r, "1 to 4 moments"), "more than 4 moments refused: %s", r); free(r);
    r = view(sid, "{\"path\":\"site/anim.html\",\"times\":[-1]}", &ok);
    CHECK(!ok && strstr(r, "between 0 and 300"), "negative moment refused: %s", r); free(r);
    put("other/anim.html", "<!doctype html><body style='background:#0f0'></body>");
    r = view(sid, "{\"path\":\"other/anim.html\"}", &ok); free(r);
    char s2[ASST_PATH + 32]; snprintf(s2, sizeof s2, "%s/.view/other_anim.html.0s.png", asst_ws);
    CHECK(!stat(s2, &st) && !stat(shot, &st), "same file name in two folders: both screenshots kept");
    asst_ledger l; asst_ledger_init(&l);
    asst_gres gs = gres("site/still.html", false);
    asst_ledger_note(&l, "view_file", &gs, "image: .view/still.html.0s.png\n", true);
    CHECK(l.n == 0, "view_file books nothing as a product (screenshots are in a dot folder)");
}

int asst_test_verify(void) {
    test_ledger(); test_scan(); test_note(); test_view();
    printf("asst verify: %d passed, %d failed%s\n", passes, fails, skips ? " (web part skipped: no browser)" : "");
    return fails;
}
