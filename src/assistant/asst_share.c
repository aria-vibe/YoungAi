/* asst_share.c — 把模型写的网页分享成一个别人点开就能看的链接(方案稿 docs/assistant_page.md §7; 10-18 用户: "分享的应该是URL不是文件")。
 *
 * 和预览(/p/, asst_preview.c)的区别: 预览口令 60 分钟过期、放行网页所在的整个目录, 是"登录的延伸", 只给自己看;
 *   分享链接要发进微信/钉钉, 收到的人没有登录、也不会在一小时内点开, 所以 /s/<口令>/ 不查登录、不过期。
 * 为什么拍快照, 不直接指向工作区: 链接一旦发出去就收不回来, 拿到它的人能按文件名去试同目录的别的文件;
 *   网页在工作区根上时, 同目录就是你的 pptx、md 和 inbox/。所以分享那一刻把网页拷进 asst_root/shares/<口令>/, 链接只读这份拷贝:
 *   - 网页在子目录里(site/index.html): 整个子目录拷过去(多文件网页照样能跑), 跳过 . 开头的和软链;
 *   - 网页在工作区根上: 只拷它自己, 加上它用 src= / href= / poster= / url( 按文件名引用到的、根上真有的文件(模型画的图常放根上)。
 *   之后模型再改这个网页, 已经发出去的链接内容不变; 同一版本(文件列表 + 内容的哈希一样)再点分享, 复用同一个口令, 不重复拷。
 * 安全头和预览一样(asst_web_serve): CSP sandbox 不透明源 + 断网。收到链接的人是在他自己的浏览器里开, 你自己点开时网页脚本也碰不到工作台的登录票。
 * 出错会怎样: 网页所在目录超过 64 MiB 或 2000 个文件 → 拒绝分享并说明原因(多半是网页放进了 inbox/ 这类大目录, 让模型把网页单独放一个文件夹);
 *   口令不对 → 404; 快照拷到一半失败 → 留下 shares/.tmp-<口令>/(. 开头, 永远不会被当成分享), 不影响别的。
 * 撤销: 删掉 shares/<口令>/ 目录, 链接立刻 404。目前页面上没有撤销按钮。 */
#include "asst_internal.h"
#include <dirent.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

static pthread_mutex_t sh_mu = PTHREAD_MUTEX_INITIALIZER;   /* 连点两下分享: 第二下要看到第一下的快照, 不能两份一起拷 */

typedef struct { char **rel; int n, cap; size_t bytes; } flist;

static bool fl_has(const flist *L, const char *rel) { for (int i = 0; i < L->n; i++) if (!strcmp(L->rel[i], rel)) return true; return false; }

static bool fl_add(flist *L, const char *base, const char *rel, char *err, size_t errn) {
    char p[ASST_PATH * 2]; struct stat st;
    snprintf(p, sizeof p, "%s/%s", base, rel);
    if (lstat(p, &st) || !S_ISREG(st.st_mode) || fl_has(L, rel)) return true;   /* 软链、目录、不存在的引用: 不拷, 也不算错 */
    if (L->n >= ASST_SHARE_MAX_FILES || L->bytes + (size_t)st.st_size > ASST_SHARE_MAX_BYTES) {
        snprintf(err, errn, "the page's folder is too large to share (over %d files or %u MiB); put the page and its files in a folder of their own", ASST_SHARE_MAX_FILES, ASST_SHARE_MAX_BYTES >> 20);
        return false;
    }
    if (L->n == L->cap) { L->cap = L->cap ? L->cap * 2 : 16; L->rel = realloc(L->rel, (size_t)L->cap * sizeof *L->rel); }
    L->rel[L->n++] = strdup(rel); L->bytes += (size_t)st.st_size;
    return true;
}

static void fl_free(flist *L) { for (int i = 0; i < L->n; i++) free(L->rel[i]); free(L->rel); memset(L, 0, sizeof *L); }

static bool walk(const char *base, const char *sub, flist *L, char *err, size_t errn) {   /* sub = 相对 base 的子目录, "" = base 本身 */
    char d[ASST_PATH * 2]; snprintf(d, sizeof d, "%s%s%s", base, sub[0] ? "/" : "", sub);
    DIR *dp = opendir(d); if (!dp) return true;
    bool ok = true; struct dirent *de;
    while (ok && (de = readdir(dp))) {
        if (de->d_name[0] == '.') continue;   /* 含 . 和 ..; .trash/.web 这类内部目录也不发出去 */
        char rel[ASST_PATH], p[ASST_PATH * 3]; struct stat st;
        snprintf(rel, sizeof rel, "%s%s%s", sub, sub[0] ? "/" : "", de->d_name);
        snprintf(p, sizeof p, "%s/%s", base, rel);
        if (lstat(p, &st)) continue;
        if (S_ISDIR(st.st_mode)) ok = walk(base, rel, L, err, errn);
        else ok = fl_add(L, base, rel, err, errn);
    }
    closedir(dp);
    return ok;
}

/* 根上的网页: 按文件名引用到的根上文件。只认不带目录、不带协议/查询的名字, 别的引用(外链、子目录)一概不管 */
static bool root_refs(const char *html, flist *L, char *err, size_t errn) {
    static const char *const KEYS[] = { "src=", "href=", "poster=", "url(" };
    for (size_t k = 0; k < sizeof KEYS / sizeof KEYS[0]; k++) {
        const size_t kl = strlen(KEYS[k]);
        for (const char *p = html; (p = strcasestr(p, KEYS[k])); ) {
            p += kl; while (*p == ' ') p++;
            const char q = (*p == '"' || *p == '\'') ? *p++ : 0;
            char v[256]; size_t n = 0;
            while (*p && n + 1 < sizeof v && (q ? *p != q : !strchr(" \t\r\n>)", *p))) v[n++] = *p++;
            v[n] = 0;
            const char *name = !strncmp(v, "./", 2) ? v + 2 : v;
            if (!name[0] || name[0] == '.' || strpbrk(name, "/\\:?#")) continue;
            if (!fl_add(L, asst_ws, name, err, errn)) return false;
        }
    }
    return true;
}

static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static bool is_tok(const char *s) { for (int i = 0; i < 32; i++) if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f'))) return false; return true; }

static void shares_dir(char *out, size_t n) { snprintf(out, n, "%s/shares", asst_root); }

/* 同一版本已经分享过 → 口令; 没有 → false */
static bool find_same(const char *key, char tok[33]) {
    char sd[ASST_PATH + 16]; shares_dir(sd, sizeof sd);
    DIR *dp = opendir(sd); if (!dp) return false;
    bool hit = false; struct dirent *de;
    while (!hit && (de = readdir(dp))) {
        if (strlen(de->d_name) != 32 || !is_tok(de->d_name)) continue;
        char p[ASST_PATH * 2]; snprintf(p, sizeof p, "%s/%s/.src", sd, de->d_name);
        char *s = asst_slurp(p, NULL);
        if (s && !strcmp(s, key)) { snprintf(tok, 33, "%s", de->d_name); hit = true; }
        free(s);
    }
    closedir(dp);
    return hit;
}

bool asst_share_path(const char *path) { return !strncmp(path, "/s/", 3); }

bool asst_share_issue(const char *in, char *url, size_t un, char *err, size_t errn) {
    char abs[ASST_PATH]; err[0] = 0;
    if (!asst_page_resolve(in, abs, sizeof abs, err, errn)) return false;
    char base[ASST_PATH]; snprintf(base, sizeof base, "%s", abs); *strrchr(base, '/') = 0;
    const char *name = strrchr(abs, '/') + 1;
    flist L = {0}; bool ok;
    if (!strcmp(base, asst_ws)) {
        char *html = asst_slurp(abs, NULL);
        ok = html && fl_add(&L, base, name, err, errn) && root_refs(html, &L, err, errn);
        free(html);
    } else ok = walk(base, "", &L, err, errn);
    if (!ok) { fl_free(&L); if (!err[0]) snprintf(err, errn, "could not read the page"); return false; }
    qsort(L.rel, (size_t)L.n, sizeof *L.rel, cmp_str);
    uint64_t h = 1469598103934665603ULL;   /* FNV-1a 64 over 每个文件的 相对路径 \0 内容: 内容或文件清单变了就是新版本 */
    for (int i = 0; i < L.n; i++) {
        char p[ASST_PATH * 2]; snprintf(p, sizeof p, "%s/%s", base, L.rel[i]);
        size_t n = 0; char *d = asst_slurp(p, &n);
        for (const char *c = L.rel[i]; ; c++) { h = (h ^ (unsigned char)*c) * 1099511628211ULL; if (!*c) break; }
        for (size_t j = 0; d && j < n; j++) h = (h ^ (unsigned char)d[j]) * 1099511628211ULL;
        free(d);
    }
    const char *wrel = abs + strlen(asst_ws) + 1;
    char key[ASST_PATH + 96]; snprintf(key, sizeof key, "share 1\n%s\n%016llx %d %zu\n", wrel, (unsigned long long)h, L.n, L.bytes);
    char tok[33], sd[ASST_PATH + 16]; shares_dir(sd, sizeof sd);
    pthread_mutex_lock(&sh_mu);
    if (!find_same(key, tok)) {
        ok = asst_rand_hex32(tok) && asst_mkdir_p(sd);
        char tmp[ASST_PATH * 2], fin[ASST_PATH * 2];
        snprintf(tmp, sizeof tmp, "%s/.tmp-%s", sd, tok); snprintf(fin, sizeof fin, "%s/%s", sd, tok);
        for (int i = 0; ok && i < L.n; i++) {
            char src[ASST_PATH * 2], dst[ASST_PATH * 3], dd[ASST_PATH * 3];
            snprintf(src, sizeof src, "%s/%s", base, L.rel[i]); snprintf(dst, sizeof dst, "%s/%s", tmp, L.rel[i]);
            snprintf(dd, sizeof dd, "%s", dst); *strrchr(dd, '/') = 0;
            size_t n = 0; char *d = asst_slurp(src, &n);
            ok = d && asst_mkdir_p(dd) && asst_write_file(dst, d, n, 0644);
            free(d);
        }
        char sp[ASST_PATH * 3]; snprintf(sp, sizeof sp, "%s/.src", tmp);
        ok = ok && asst_write_file(sp, key, strlen(key), 0644) && !rename(tmp, fin);   /* 整份拷完才改名: 半份快照永远不会被当成分享 */
        if (!ok) snprintf(err, errn, "could not save the shared copy of the page");
    }
    pthread_mutex_unlock(&sh_mu);
    fl_free(&L);
    if (ok) asst_url_name("/s/", tok, abs, url, un);
    return ok;
}

int asst_share_get(const char *method, const char *path, const char *host, ds4_buf *hdr, ds4_buf *body) {
    if (strcmp(method, "GET")) return 405;
    if (!asst_root[0] || !asst_share_path(path)) return 404;
    const char *p = path + 3, *sl = strchr(p, '/');
    if (!sl || sl - p != 32 || !is_tok(p)) return 404;
    char sd[ASST_PATH + 16], d[ASST_PATH * 2], real[4096]; shares_dir(sd, sizeof sd);
    snprintf(d, sizeof d, "%s/%.32s", sd, p);
    if (!realpath(d, real)) return 404;   /* asst_web_serve 拿 realpath 后的文件比前缀, root 也得是 realpath(家目录带软链时) */
    return asst_web_serve(real, false, sl + 1, host, hdr, body);
}

void asst_share_serve(int fd, const char *method, const char *path, const char *host) {
    ds4_buf hdr = {0}, body = {0};
    asst_web_respond(fd, asst_share_get(method, path, host, &hdr, &body), &hdr, &body);
}
