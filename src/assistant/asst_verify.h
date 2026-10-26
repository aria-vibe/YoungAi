/* asst_verify.h — 校验相(asst_verify.c, 2026-10-23, 方案稿 docs/assistant_verify.md): 做 → 核 → 答, 一个 agent 范式, 不分任务类型。
 * 动过手(调过工具)的运行, 模型想收口时框架先摆一条事实消息(调了什么、产出/改动了哪些文件、哪些没再看过、回答里点名却不存在的), 模型自己核、修、再答。
 * 框架只记账与摆事实, 不判对错; ASST_VERIFY_ROUNDS 条之后无论如何结束(不是兜底: 该不该再修是模型的事, 上限只是别替它硬扛)。
 * 单独一个头(照 tools/tool_image.h): asst_internal.h 已顶着 500 行。 */
#ifndef DS4_ASST_VERIFY_H
#define DS4_ASST_VERIFY_H
#include "asst_internal.h"

#define ASST_VERIFY_ROUNDS 2
#define ASST_LEDGER_MAX 64
#define ASST_LEDGER_TOOLS 24
#define ASST_LEDGER_SCAN_MAX 20000     /* run 前后扫工作区的条目上限: 超了只记"没扫全" */
typedef struct { char rel[ASST_PATH]; char by[24]; int wrote_at, seen_at; long size; } asst_ledger_item;   /* rel = 工作区相对路径; by = 写它的工具; *_at = 第几次工具调用(从 1 数, 0 = 没有) */
typedef struct {
    asst_ledger_item it[ASST_LEDGER_MAX]; int n; bool more;
    struct { char name[32]; int count; } tool[ASST_LEDGER_TOOLS]; int ntool;
    int step;            /* 这一趟到现在的工具调用次数 */
    int noted_step;      /* 上一条校验事实消息发出时的 step(0 = 还没发) */
    int rounds;          /* 已发几条校验事实消息 */
} asst_ledger;
void asst_ledger_init(asst_ledger *l);
void asst_ledger_note(asst_ledger *l, const char *tool, const asst_gres *g, const char *result, bool ok);   /* 每次工具调用后: 记次数; 产出工具成功 → 记产物; 看的工具 → 记看过 */
void *asst_ledger_snap(void);                                          /* run 之前: 工作区快照(mtime + size), 之后交给 asst_ledger_diff */
void asst_ledger_diff(asst_ledger *l, void *before, const char *by);  /* run 之后: 新增/改动的文件进账(by = 工具名), 释放快照 */
char *asst_verify_note(asst_ledger *l, const char *answer);           /* 模型想收口时: 有要核的事且轮数没到 → 事实消息(malloc), 否则 NULL */
char *asst_file_mentioned_missing(const char *answer);                /* 回答里写得像工作区文件却不存在的 → JSON 数组(asst_file.c); 没有 NULL */
char *asst_view_exec(const char *args, const asst_gres *g, const char *trash, int *ok);   /* view_file(tools/tool_view.c): 像用户那样看一个文件; 已过守门(读类, g->path 绝对路径) */
char *asst_view_office(const char *abs, size_t max);                   /* tools/tool_view_doc.c: docx / pptx / xlsx 的正文(malloc); 不是合法包 NULL */
const char *asst_ff_clock_js(void);                                     /* tool_video.c 的虚拟时钟 + 错误收集预载(view_file 复用, 不抄第二份) */
asst_vsrv *asst_vsrv_start_at(const char *page, size_t n, const char *rel, char *url, size_t urln, char *err, size_t errn);   /* 页面挂在 /<口令>/<rel>: 相对资源按它所在目录解析 */

#endif
