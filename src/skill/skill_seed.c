/* skill_seed.c — 仓库自带技能默认装上(10-18 用户: "实现一个金融 skill, 默认加入当前项目的 skill")。
 * ds4-train 起来时把 <仓库根或发布包>/skills/<名>/ 装进 ~/.ds4/assistant/skills/<名>/。走和页面导入同一个口 asst_skill_put_file
 * (扩展名白名单 + 脚本语句检查), 自带的技能不享受特权: 写了命令照样装不进去。
 * 账本 <助理根>/skills.seeded, 一行 "名字 指纹", 记"装的时候是哪一版":
 *   没装过                          → 装, 记账;
 *   装过, 盘上没被改过(指纹 == 账本), 仓库版变了 → 旧的挪进 trash/skills/ 再装新版(升级跟着发布走);
 *   装过, 用户删了                  → 不再装回来(删是用户的决定);
 *   装过, 用户在技能页改过          → 不动(用户的版本优先, 仓库升级也不覆盖);
 *   账本里没有但盘上已有同名(早先手动导入过) → 内容相同就补记账(以后能跟着升级), 不同就不动也不记。
 * 指纹 = 按相对路径排序的每个文件 "路径\0内容" 的 FNV-1a 64: 防"拿错版本", 不防篡改。
 * 为什么不在 asst_init 里做: 单测拿临时目录当助理根, 自动装两个技能进去, "一个没装 → 没有 use_skill" 那类断言就全变了; 由 ds4-train 显式调。 */
#include "../assistant/asst_internal.h"
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>

#define SEED_FILES 256   /* 一个技能最多收这么多个文件(与 asst_skill_load 的文件表上限同量级) */
#define SEED_DEPTH 4

typedef struct { char rel[SEED_FILES][256]; int n; } flist;

static void collect(const char *root, const char *rel, flist *f, int depth) {
    char d[ASST_PATH * 2]; snprintf(d, sizeof d, "%s%s%s", root, rel[0] ? "/" : "", rel);
    DIR *dir = opendir(d); if (!dir) return;
    struct dirent *e;
    while ((e = readdir(dir)) && f->n < SEED_FILES) {
        if (e->d_name[0] == '.') continue;   /* 隐藏文件不收(导入也不收): .DS_Store 之类 */
        char r[256]; if (snprintf(r, sizeof r, "%s%s%s", rel, rel[0] ? "/" : "", e->d_name) >= (int)sizeof r) continue;
        char p[ASST_PATH * 2]; snprintf(p, sizeof p, "%s/%s", root, r);
        struct stat st; if (stat(p, &st)) continue;
        if (S_ISDIR(st.st_mode)) { if (depth < SEED_DEPTH) collect(root, r, f, depth + 1); }
        else if (S_ISREG(st.st_mode)) snprintf(f->rel[f->n++], 256, "%s", r);
    }
    closedir(dir);
}
static int cmp_rel(const void *a, const void *b) { return strcmp((const char *)a, (const char *)b); }

static uint64_t dir_fnv(const char *root, flist *f) {
    f->n = 0; collect(root, "", f, 0);
    qsort(f->rel, (size_t)f->n, sizeof f->rel[0], cmp_rel);
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < f->n; i++) {
        char p[ASST_PATH * 2]; snprintf(p, sizeof p, "%s/%s", root, f->rel[i]);
        size_t len = 0; char *data = asst_slurp(p, &len);
        for (const char *s = f->rel[i]; ; s++) { h ^= (unsigned char)*s; h *= 1099511628211ull; if (!*s) break; }   /* 带上 \0: "ab"+"c" 与 "a"+"bc" 不同 */
        for (size_t k = 0; data && k < len; k++) { h ^= (unsigned char)data[k]; h *= 1099511628211ull; }
        free(data);
    }
    return h;
}

static bool ledger_get(const char *led, const char *name, uint64_t *h) {
    for (const char *p = led; p && *p; ) {
        const char *nl = strchr(p, '\n'); const size_t n = strlen(name);
        unsigned long long v;
        if (!strncmp(p, name, n) && p[n] == ' ' && sscanf(p + n + 1, "%llx", &v) == 1) { *h = v; return true; }
        p = nl ? nl + 1 : NULL;
    }
    return false;
}
static void ledger_set(ds4_buf *out, const char *led, const char *name, uint64_t h) {   /* 重写一份: 去掉这个名字的旧行, 末尾加新行 */
    const size_t n = strlen(name);
    for (const char *p = led; p && *p; ) {
        const char *nl = strchr(p, '\n'); const size_t len = nl ? (size_t)(nl - p) : strlen(p);
        if (len && !(len > n && !strncmp(p, name, n) && p[n] == ' ')) { ds4_buf_append(out, p, len); ds4_buf_putc(out, '\n'); }
        p = nl ? nl + 1 : NULL;
    }
    ds4_buf_printf(out, "%s %016llx\n", name, (unsigned long long)h);
}

/* 一个技能整份装进去; SKILL.md 先装(它被拒就整个不装, 不留半拉子文件夹) */
static bool install(const char *src, const char *name, const flist *f, ds4_buf *log) {
    int order[SEED_FILES], k = 0;
    for (int i = 0; i < f->n; i++) if (!strcmp(f->rel[i], "SKILL.md")) order[k++] = i;
    if (!k) { ds4_buf_printf(log, "%s: no SKILL.md; ", name); return false; }
    for (int i = 0; i < f->n; i++) if (strcmp(f->rel[i], "SKILL.md")) order[k++] = i;
    for (int i = 0; i < k; i++) {
        char p[ASST_PATH * 2]; snprintf(p, sizeof p, "%s/%s", src, f->rel[order[i]]);
        size_t len = 0; char *data = asst_slurp(p, &len);
        char *why = data ? asst_skill_put_file(name, f->rel[order[i]], data, len) : strdup("cannot read");
        free(data);
        if (why) { ds4_buf_printf(log, "%s/%s skipped: %s; ", name, f->rel[order[i]], why); free(why); if (!i) return false; }
    }
    return true;
}

int asst_skills_seed(const char *src_root, char *msg, size_t msgn) {
    if (msg && msgn) msg[0] = 0;
    if (!asst_skills[0] || !src_root) return 0;
    DIR *d = opendir(src_root); if (!d) return 0;
    char lp[ASST_PATH + 32]; snprintf(lp, sizeof lp, "%s/skills.seeded", asst_root);
    char *led = asst_slurp(lp, NULL);
    flist *fs = malloc(sizeof *fs), *fd = malloc(sizeof *fd);   /* 各 64 KB: 工人线程栈小, 放堆上 */
    ds4_buf log = {0}; int done = 0; bool dirty = false;
    struct dirent *e;
    while (fs && fd && (e = readdir(d))) {
        const char *name = e->d_name;
        if (!asst_skill_name_ok(name)) continue;
        char src[ASST_PATH + 96]; snprintf(src, sizeof src, "%s/%s", src_root, name);
        char sm[ASST_PATH + 112]; snprintf(sm, sizeof sm, "%s/SKILL.md", src);
        struct stat st; if (stat(sm, &st) || !S_ISREG(st.st_mode)) continue;
        const uint64_t hs = dir_fnv(src, fs);
        char dst[ASST_PATH + 96]; snprintf(dst, sizeof dst, "%s/%s", asst_skills, name);
        const bool have = !stat(dst, &st) && S_ISDIR(st.st_mode);
        uint64_t hr = 0; const bool known = led && ledger_get(led, name, &hr);
        const uint64_t hd = have ? dir_fnv(dst, fd) : 0;
        bool record = false;
        if (!known && !have) { if (install(src, name, fs, &log)) { ds4_buf_printf(&log, "installed %s; ", name); done++; record = true; } }
        else if (!known) record = hd == hs;                     /* 早先手动导入的同一版: 补记账 */
        else if (have && hd == hr && hs != hr) {                /* 没被改过, 仓库出了新版 */
            char t[ASST_PATH + 32]; snprintf(t, sizeof t, "%s/trash", asst_root); mkdir(t, 0700);
            snprintf(t, sizeof t, "%s/trash/skills", asst_root); mkdir(t, 0700);
            char to[ASST_PATH + 128]; snprintf(to, sizeof to, "%s/trash/skills/%s-%ld-replaced", asst_root, name, (long)time(NULL));   /* 后缀: 与页面删除的 <名>-<秒> 同一秒撞名时 rename 会失败 */
            const bool on = asst_skill_enabled(name);   /* 开关(.disabled)不进指纹, 但旧文件夹整个挪走了: 用户关掉的, 新版装好照样关着 */
            if (!rename(dst, to) && install(src, name, fs, &log)) { if (!on) asst_skill_set_enabled(name, false); ds4_buf_printf(&log, "updated %s; ", name); done++; record = true; }
        }
        if (record) { ds4_buf nl = {0}; ledger_set(&nl, led, name, hs); free(led); led = ds4_buf_take(&nl); dirty = true; }
    }
    closedir(d);
    if (dirty && led) asst_write_file(lp, led, strlen(led), 0600);
    if (msg && msgn) snprintf(msg, msgn, "%s", log.ptr ? log.ptr : "");
    free(led); free(fs); free(fd); ds4_buf_free(&log);
    return done;
}
