# 助理的校验相(verify)方案稿: 做 → 核 → 答

2026-10-23。读者 = 第一次接触这个项目的人。先说用起来什么样、框架管什么、代码落在哪; 为什么这么设计放最后。
总纲 `docs/assistant_arch.md`(四条铁律, 本稿不违反任何一条), 计划模式 `docs/assistant_plan.md`, 网页预览 `docs/assistant_page.md`。
参照物: DeepSeek Harness(`deepseek-ai/deepseek-harness`, 2026-08 开源, 下文简称 dsh), §7 说清借了什么、没借什么。落地过程与判决进 `fable5.md`。

用户三句话定的边界(10-23): ① "校验是 agent 架构里重要的一环"(10-16 "不要截图"是误会, 不算否决); ② "不要硬编码"; ③ "是一个 agent 范式, 不是单独在生成网页的时候起作用"。
所以本稿**不按文件类型写任何判定规则**, 也不只管网页: 它改的是 agent 调用循环本身。

## 0. 一句话

**以前的循环里, "做完" = 模型不再调工具; 现在 "做完" = 模型核过了自己的活。** 这一趟只要动过手(调过 ≥1 次工具), 模型第一次想收口时框架不结束, 而是以 user 角色摆回一条**事实消息**:
这趟调了哪些工具、产出或改动了哪些文件(每个多大、谁写的、写完之后它看过没有)、回答里点名了却不存在的文件。
核什么、怎么核、要不要修, 全是模型的事 —— 它有 `view_file`(像用户那样看任何文件)、`read_file`、`run`。核完再答一遍, 最多两轮。
纯问答(零工具调用)不进校验相: 没有可以核对的现实, 加一轮只是拖慢每句话。

鹈鹕这一条("创建一个 HTML, SVG 鹈鹕骑自行车的 2D 动画")今天是这样的:

| 现状(10-20 九趟真模型) | 本稿之后 |
|---|---|
| `write_file pelican.html` → 两句话收工。页面给**用户**渲染了, 模型自己从没看过它画的是什么; 九趟里只有 2 趟自己想起来调 `render_image` 看一眼 | 想收工时收到: `[verify before answering — check 1 of 2. This run so far: 1 tool call (write_file 1). Files produced or changed: pelican.html (7.2 KB, write_file, not looked at since). Before the final answer, check the work …]` → 它调 `view_file pelican.html` → 看到 0 s / 1 s 两张截图 + 浏览器报的脚本错误 → 改或不改 → 再答 |
| 脚本有错也照样"做好了" | 模型看见错误再决定; 改了之后第二条事实("changed since the last check", 标 last) → 再答 → 结束 |
| 说"报告在 report.pdf"但盘上没有 | 事实消息里一句 `Named in your answer but not in the workspace: report.pdf` |

## 1. 用起来什么样

| 你做的 | 页面上 | 模型那头 |
|---|---|---|
| "附近有什么好吃的"(查了定位、搜了周边, 没写文件) | 模型答完 → 一条灰色小气泡"校验 1: [verify … 3 tool calls (locate 1, nearby 1, …) …]" → 它核对数字再答一遍(或直接重申) | 收到事实消息; 产物一栏没有, 只有工具清单 |
| "写一页鹈鹕骑车的 SVG 动画" | 工具卡 `write_file` → 灰气泡"校验 1" → 工具卡 `view_file`(结果里两张截图) → 可能 `edit_file` → 灰气泡"校验 2" → 终稿 → 文件卡 | 见 §0 |
| "把这份 csv 按月汇总出一张 xlsx" | `write_xlsx` → 校验 1 → 模型 `view_file out.xlsx` 看回每格的值与公式缓存 → 终稿 | `view_file` 对 xlsx 回 `A1=Month  B1=Revenue / A2=Jan  B2=120 … B4=200 {=SUM(B2:B3)}` |
| 模型用 `run` 跑 python 写出 `out/chart.png` | 校验 1 里 `out/chart.png (12 KB, run, not looked at since)` | run 前后扫工作区比出来的, 不靠模型报 |
| "Redis 流是什么" | 和今天一样, 没有任何变化 | 零工具调用 → 不进校验相 |
| 两轮校验后模型还是没修好 | 终稿照常出, 校验气泡留着, 用户看得见它说了什么 | 第二条事实末尾: "This is the last check in this run … say plainly what could not be fixed" |

出错会怎样:
- 本机没火狐 → `view_file` 对网页回"这台机器渲染不了, 用户在文件卡里看; 用 read_file 查源码"; 别的格式照常。不拦、不当成产物有错。
- 页面脚本死循环 → 火狐 20 s 没回音被杀(与 make_video 同), `view_file` 回"页面没响应"。
- 一次 `run` 改了几十个文件 → 账本最多 64 条, 超了事实里写 "(and more not listed)"; 工作区超 2 万个条目只记"没扫全", 不卡运行。
- 事实消息里"点名却不存在"是启发式(词尾像文件名 + 认识的扩展名): `node.js` 会被误报。只是事实, 模型可以说明那不是文件。

## 2. 架构: 一张账本 + 一条事实消息 + 一只手

```
 用户消息 ──▶ 模型 ──▶ tool_calls ──▶ 守门 ──▶ 执行 ──▶ 回填 ──▶ 模型 …
                 ▲                              │ 账本(asst_verify.c): 每次调用记一笔 —— 工具名计数; 产出工具成功 → 路径/大小/第几步;
                 │                              │   run → 执行前后扫工作区(mtime+size)比出新增/改动; view_file / read_file / view_image → 记"看过"
                 │                              ▼
                 │     模型不再调工具 ──▶ ★校验相: 动过手 或 回答里点名了不存在的文件 → 事实消息(user 角色) + verify_note 事件 + 审计 ──┘
                 │                          │   第 2 次只在"上一条之后又写了东西"或仍有点名缺失时发; 第 ASST_VERIFY_ROUNDS(=2) 条之后无论如何结束
                 └──────────────────────────┘
                        没得核 / 轮数到 ──▶ files 事件 → end(与今天同)
```

### 2.1 账本(`asst_ledger`, 住在运行线程栈上, 一趟一本)

| 记什么 | 来源 | 为什么不是别的 |
|---|---|---|
| 工具名 → 次数 | 每次调用 | 事实消息第一句"这趟调了什么" |
| 产物: 相对路径、大小、写它的工具、第几步写的、第几步看过 | `write_file / edit_file / write_document / write_xlsx / render_image / make_video / fetch_image` 成功 → `g->path`; `edit_image` → 结果开头 `image:` 行(源图不是产物); `run` → 前后扫描 | 只认工具真做了的事, 不猜 |
| "看过" | `view_file / read_file / view_image` 成功且路径在账上 | 事实消息能说"写完之后没再看过" |

点开头目录(`.trash/.view/.img/.web/.video/.phone`)一律不进账: 内部产物。工作区外的产物(审批放行写出去的)不进账: 账本只管工作区。

### 2.2 事实消息(全文, 英文, 与别的 note 同语种; 回答语言由 system 与用户消息定, 10-11 实测工具结果语种不影响)

```
[verify before answering — check 1 of 2. This run so far: 3 tool calls (write_file 1, view_file 1, edit_file 1).
 Files produced or changed: pelican.html (7.2 KB, write_file, changed since the last check, not looked at since).
 Named in your answer but not in the workspace: report.pdf.
 Before the final answer, check the work against what the user asked: look at what you produced the way the user will see it (view_file),
 check the facts you state against the tool results, fix what is wrong, then answer again in full.
 This is the last check in this run: after your next answer the run ends, so say plainly what could not be fixed.]
```

它和现有的 `[note: … identical call …]` / `[note: N tool calls … answer now]` 是同一类东西: 框架的机械护栏, 不是提示词里的场景规则。
以 `{"role":"user","content":…}` 进 `messages.jsonl`(模型可见即已记录, 下一轮历史完整, KV 前缀接得上); 事件流里是 `verify_note`, 页面渲染成**灰色系统气泡**, 不渲染成用户气泡(它不是用户说的); 审计一行 `tool:"verify", decision:"note"`。

### 2.3 `view_file(path, width?, height?)`: 像用户那样看

一个工具、一个概念: "看"。实现里按格式走不同的路是不可避免的, 但**只报事实, 不判好坏**:

| 格式 | 怎么看 | 回什么 |
|---|---|---|
| html / htm / svg | 火狐: `tool_video_srv.c` 的本机只读 HTTP 源把页面挂在它在工作区里的**相对路径**上(同目录 css/js/图按相对路径取得到), 预载 = `tool_video.c` 的虚拟时钟(含脚本错误 / CSP 拦截收集) + 一个捕获阶段的资源加载失败监听; `__ds4.step(0)` 截一张, `__ds4.step(1000)` 再截一张, 两张逐字节同就只给一张 | `image:` 行(1 或 2 张, 落 `.view/<名>.0s.png` / `.1s.png`) + "picture at 0 s and at 1 s are identical (nothing moves)" 或 "they differ" + 浏览器报的页面错误原文 + 没加载到的资源(缺文件 / 被拦的外网 URL) |
| png / jpg / gif / webp / bmp | 读头 | `image:` 行 + 尺寸、大小 |
| mp4 | `make_video` 存在 `.video/<名>/` 的帧 | 首、中、末三帧 `image:` 行 |
| docx / pptx / xlsx | 读 zip 中央目录 → `office_inflate` → 去标签(`tool_view_doc.c`) | 正文: docx 按段落与表格(制表分格); pptx 按页; xlsx 逐格 `A1=…`(inlineStr / sharedStrings 都认, 公式给缓存值与 `{=公式}`) |
| pdf | 工具箱有 `pdftotext` 就在沙盒里抽字 | 文字(保留版面); 没有 pdftotext 直说 |
| 其余 | 开头 16 KB | 行数、大小、正文 |

守门按读类(与 `read_file` 同口径): 工作区内放行, 工作区外审批。网页的截图落盘走 `asst_save_bytes`(沙盒子进程写, 旧版进 `.trash`)。

### 2.4 不改的东西

提示词**一个字不改**。守门决策序、沙盒、`run` 不审批、事实行、重复调用 ×3 提醒、10 次调用收口压力、计划模式、任务、通知、"终稿写路径 → 文件卡"、页面不扫工作区 —— 全不动。
`write_file` 等产出工具的结果也不动(10-23 用户: 不要硬编码 —— "网页的结果自带截图 + 动画有没有在动"那套按类型的复验器表在方案第一版里, 已删)。

## 3. 代码落点(10-23 已落地; 每文件 ≤500 行, `make linecount` 过)

```
src/assistant/asst_verify.h          新: 账本结构 / ASST_VERIFY_ROUNDS 2 / 声明(asst_internal.h 顶着 500 行, 照 tools/tool_image.h 另起一个头)
src/assistant/asst_verify.c          新(129 行): asst_ledger_note(记次数 / 产物 / 看过) · asst_ledger_snap + asst_ledger_diff(run 前后扫描, 归并比对) · asst_verify_note(事实消息: 触发条件 + 文本 + 轮数)
src/assistant/tools/tool_view.c      新(168 行): view_file 五类的"看"; 网页复用 asst_vsrv_start_at + asst_ff_clock_js, 截图经 asst_save_bytes 落 .view/
src/assistant/tools/tool_view_doc.c  新(126 行): zip 中央目录 → 部件 → 去标签正文(docx / pptx / xlsx)
src/assistant/tools/tool_video_srv.c 页面可挂在任意相对路径(asst_vsrv_start_at; 原 asst_vsrv_start 成为它的壳), 路径段百分号编码
src/assistant/tools/tool_video.c     asst_ff_clock_js() 把虚拟时钟预载公开给 view_file(不抄第二份)
src/assistant/asst_file.c            asst_file_mentioned_missing: 与"终稿写路径 → 文件卡"同一套分词, 认识的扩展名 + 不存在 → 列出
src/assistant/asst_tools.c           注册表一行 view_file(读类, 不按火狐有无忽隐忽现); 执行分发
src/sandbox/sbx_guard.c              view_file 走 read 口径
src/assistant/asst_run.c             run_one: 账本随运行; run 前后快照; 每次调用后 asst_ledger_note; 模型无 tool_calls → asst_verify_note 有话就追加 user 消息 + verify_note 事件 + 审计, continue
web/studio.html                      verify_note 事件 → .vnote 灰色系统气泡(标签"校验 N", 正文默认折两行点开全看); 文案 vf_tag 中英; 零旋钮
tests/asst/t_asst_verify.c           新, 37 项(§5)
tests/asst/fake_model.c              剧本 "e2e 校验": 写带脚本错误的网页 → 收口(点名不存在的 pdf) → 收到事实 → view_file → edit_file → 收口 → 第二条事实 → 终稿
tests/asst/e2e_device.mjs            校验段 7 项; 原"第二条消息"断言改为跳过事实消息(它也是 user 角色)
```

## 4. 页面

一个新节点: `.vnote` 灰色系统气泡(左侧小标签"校验 1 / 校验 2", 右侧事实原文, 默认两行, 点开全看)。手机上同样。不加任何开关; 活动页审计多一类 `verify`。

## 5. 门(10-23 Mac, 原始输出进 fable5)

- **离线 `ds4_asst_test`**: 新段 `asst verify` **37/37**(账本: 八类产出工具进账 / 失败不进 / edit_image 从 image: 行取成品而源图与 .img/ 不算 / 工作区外与点开头目录不进 / 工具计数;
  扫描: 改动 + 新增进账、没变的与点开头目录不进、带步号与大小; 事实消息: 零调用不发 / 第 1 条文本(计数、产物、"not looked at since"、view_file 指路) / 没变化不发第 2 条 /
  改过发第 2 条且标 last / 第 3 条不发 / 看过的措辞 / 点名缺失(中文文件名、sandbox: 前缀、bing.com 与 v1.0 不算、存在的不算);
  view_file: 守门读类、工具表有它、文本头、缺文件、png 尺寸、沙盒里真写 xlsx/docx/pptx 再看回正文(inlineStr 格值与公式缓存 / 段落与表格格 / 逐页)、坏包、视口越界;
  网页(要火狐): 带 TypeError + 外网字体 + 缺图 + CSS 动画的页 → 两张图 "they differ" + 错误原文 + 两条没加载到的资源; 静态页 → 一张图 "identical" + 无错; 截图落 .view/; view_file 不进账)。
  全套 `./ds4_asst_test` 其余各段计数与改前相同。
- **`make asst-e2e`** 校验段 7 项: 两条事实(第 1 条列产物 + 点名缺失; 第 2 条 changed + last) / 做 → 核 → 修三个工具卡 / 终稿是校验后的那条 / 页面两条灰气泡且用户气泡只有一条 / messages.jsonl 两条 user 角色事实 / 审计两条。
  附带: 原"附近美食"全流程也进了校验相(9 次调用 → 第 1 条事实 → 假模型重申终稿 → 没变化 → 结束), 原有断言全过。
- **`make linecount`** 过。
- **真模型(spark, 待跑)**: 同一句鹈鹕 ×3(与 10-20 九趟同口径), 判据是审计流水: 每趟有 `verify` note ≥1; ≥2/3 趟事实之后调了 `view_file`; 看过之后有没有 `edit_file`、最终 html 的页面错误数; 事实消息每趟 ≤2 条; "附近有什么好吃的"也会进一轮(它调了工具), 看它是重申还是改数字; 纯问答零事实。
  失败不改提示词: 先看事实消息写清了没、`view_file` 的图进了没(服务端 400 = 没视觉塔)、火狐在 spark 上截图通不通(render_image 已通)。

## 6. 代价

- 动过手的每趟多一轮模型调用(事实消息 → 再答); 模型要修就再多一轮。spark 真模型开思考一轮 20~60 s。10-20 06:51 那趟模型自发走的就是"写 → 看 → 改 → 看 → 收口", 后续轮的思考 158~1386 位, 远短于首轮。
- `view_file` 网页: 火狐加载 + 两张截图 1~3 s; 两张 1280×800 图进上下文几千 token。
- `run` 前后各扫一遍工作区(readdir, 不读内容), 几千个文件毫秒级。
- 代码: C 约 480 行新 + 60 行改, JS 20 行, 测试 160 行 + e2e 25 行。提示词不变, 工具表对一段对话稳定(多一行常量), KV 前缀不变。

## 7. 对照 dsh: 借了什么, 没借什么

| dsh 里的东西 | 本稿 | 为什么 |
|---|---|---|
| `agent/turn-stopping` —— 模型自然停下时的串行检查点, 监听器可以不让轮次结束 | **借**: 校验相就在这个位置 | 没有这个位置, "做完"只能是模型的自我声明 |
| `tools/post-execute` 的 `additionalContexts` —— 工具结果之后注入一条有来源标注的合成 user 消息, 建议性不否决 | **借**: 事实消息(user 角色, `verify_note` 标注来源) | 模型可见即已记录; 下一轮历史完整 |
| `deliverables/workspace-changes` —— 每轮前后 git 快照比出改了哪些文件 | **借思路换实现**: run 前后 mtime+size 扫描 | 工作区不是 git 仓库, 用户的照片和料都在里面 |
| `deliverables/present` —— 模型声明"这些是最终交付" | **不借**: 已有"终稿写路径 → 文件卡"; 只多"点名的存不存在" | 不加第二套交付声明 |
| `guard/repeat-tool-reminder`(3/5/8 提醒) | 已有(×3), 不动 | — |
| `goal` + `goal-round-driver` —— 持久目标 + 自动续跑到上限, 阻塞带稳定码 | **借形不借体**: 校验轮上限 + 审计里 `verify` 类 | 持久目标用现有任务/计划 |
| 独立评审器(evaluator) | dsh **没有**(文档明写 "Completion is worker self-declaration", 评审器 deferred); Claude Code 的 goal 用小模型评审 | 用户 10-23 定不做("没看有哪家这么做的"); 一份模型一个 KV 槽也付不起预填税 |
| Cordis 插件内核、子代理、workflow 脚本、ralph、上下文压缩 | **不借** | C 项目, 注册表一行就是"插件"; 一份模型; 1M 上下文 + 磁盘 KV 暂不需要压缩 |

## 8. 明确不做的

- 按文件类型写判定规则的复验器(网页要有动画 / 表格要有合计 / 图不能是单色): 硬编码, 没完; 判是模型的事, 框架只给"看"的手和事实。
- 产出工具的结果自带截图(第一版的"门 A"): 同上, 也是按类型特判; 现在模型想看就 `view_file`。
- 干净上下文的第二次模型调用当评审(第一版的"门 C"): 用户定不做。
- 校验结果否决收口(拒绝结束、强制重做): 建议性, 两轮后如实呈现。
- 页面或服务端按关键词猜"这是要产物的任务": 触发条件只有"调过工具"这一条机械事实。
- 给校验加开关或轮数旋钮: 页面不放旋钮; `ASST_VERIFY_ROUNDS` 是常量不是配置。

## 9. 为什么这样设计(放最后)

- **范式而不是特例。** 做 → 核 → 答对每一类任务都成立: 查行情核对数字、写网页看渲染、改图看成品、跑脚本看产物。只在网页上起作用的"复验"是又一个场景规则, 下一个任务类型来了又要加。
- **根因不是模型懒, 是它瞎 + 没人叫它回头看。** 10-20 九趟鹈鹕, 模型从没见过自己画的东西; 自发去看的两趟都接着改了。提示词里写"写完要看一眼"是场景规则(铁律 1), 而且 10-13 加的"别在思考里先写一遍"实测它不听。框架机械地在收口前摆事实, 每趟都摆。
- **框架摆事实, 不下判断。** "pelican.html 7.2 KB, 写完没看过"是事实; "请修复"不是。判要不要紧、要不要修是模型的事。这和计划模式"待确认态只拒写, 不替模型判"是同一种分工。
- **为什么是 2 轮不是 1 也不是 5。** 10-20 06:51 那趟模型自己走的就是两轮修正; 再多就是在替模型硬扛, 该让它如实说。
- **为什么不请第二个模型评审。** 用户定不做; dsh 也把评审器推迟了(输入、权限、隔离、裁决权要单独设计); 这台机器一份模型一个 KV 槽, 评审会挤掉作者的检查点。

## 10. DeepSeek Harness 对照实验与三处 harness 修正(10-23 晚)

用户: "校验有用, 但实际效果跟 DeepSeek Harness 里面差一大截" → 装 dsh(npm `@deepseek-ai/dsh@0.2.1-rc.2`, 装在 `gguf/dsh/`, 用户同意)对照。
脚本 `tests/asst/dsh_compare.sh official|local "请求"`: 同一个 dsh、同一个适配器(DeepSeek Messages 协议, ds4-server 本来就讲 `/v1/messages`)、同一份 patch, **只换 baseURL**;
运行目录单独 `git init`(不然 dsh 往上找到本仓库 `CLAUDE.md` 塞进系统提示), 模型工作目录 `ws/` 与日志分开(不然它 `ls` 就看见 patch 里的 baseURL)。

| 同一句鹈鹕 | dsh | 我们的助理 |
|---|---|---|
| 官方原件 deepseek-flash | 两趟 514 / 551 s, 41 / 47 次调用, 40 / 48 KB: 完整插画(头盔、围巾、喉囊、羽毛、海岸公路) | 没测 |
| 我们的量化模型 | 中途停(33 分钟、31 步): 31 KB, 完整海岸场景, 但鹈鹕本体简陋 | 10.7 KB, 车 + 鸟 + 两朵云 |

判读:
- **模型差距**(同 dsh, 换模型): 角色造型差一截, harness 改不了 —— 归量化。
- **harness 差距**(同量化模型, 换 harness): dsh 的系统提示只有工具用法、零设计指导; 差在我们这边的两处压制(下面 ①②)。
- **校验方向不是瓶颈**: dsh 没有校验相; 三趟都是我 patch 漏写 `inputModalities`, `read_image` 报"不支持图片", 模型自己用 bash 起无头 Chrome 截图、写 Python 把 PNG 转字符画、采样像素、用 Swift 调苹果 Vision、走 CDP 查 DOM —— 自发核对的劲头来自模型 + 不受限的 bash。

三处修正(证据都是 10-23 我们那趟鹈鹕会话 0621522fb9bd 的思考原文):
1. **删掉"第 10 次调用后每条结果催收口"**(`asst_run.c` 原 `ASST_CALL_BUDGET`): 原文 "I got a nudge to answer. I should quickly verify the final render once and then answer" / "The note said answer now"。dsh 没有这条。当初为 10-12 "附近有什么吃的" 30 次猜坐标而加, 那个病根后来由 `locate` + `nearby` 坐标参数治了; 打转剩同名同参 ×3 提醒与校验相两道。
2. **身份段把"简洁"限定到说话、把"不扩大范围"限定到不加别的事, 并补"这件事本身做足"**(`asst_memory.c` 中英两份): 第一轮思考原文 "Keep minimal" / "keep it simpler" / "Might be overkill" / "Add palm tree? Not necessary; keep focus"。
3. **`view_file` 截图按相对路径命名**(原来只按文件名, `a/index.html` 与 `b/index.html` 互相覆盖) **+ 可选 `times`**(最多 4 个时刻, 升序去重; 原文 "view_file gives 0s and 1s only")。

门: Mac/spark `ds4_asst_test` 全绿(memory 30 / verify 41, 其余计数不变), `make linecount` ok, `make asst-e2e` 全过; spark 主进程已换(模型子进程没动)。真模型复验见 fable5。
