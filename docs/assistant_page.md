# 助理生成网页 + 页面里直接预览

2026-10-16。读者 = 第一次接触这个项目的人。先说做成什么样、用户看到什么、代码放哪、怎么验证；
为什么这么做放后面。总纲 `docs/assistant_arch.md` 的四条铁律全部适用：提示词一个字不改，
场景知识只进工具描述。落地过程与判决进 `fable5.md`。

## 0. 一句话

模型 `write_file` 写出一个 `.html`，回答下面的文件卡上点"预览"，网页就在卡里真渲染出来（iframe）；点"打开"整页看。
以前点"预览"看到的是一坨源码。

| 件 | 以前 | 现在 |
|---|---|---|
| 生成 | `write_file` 写 .html / .css / .js | **不加工具**；`write_file` 描述多一句"html 会渲染预览、无网络、资源用同目录相对路径" |
| 用户预览 | 源码文本 | 卡内 iframe 真渲染 + "打开"整页；手机经穿透同样能看 |

**不做模型自己截图看页面**（10-16 用户定："直接 HTML 就能渲染，不要截图"）。曾写过一版 `view_page`（火狐截图 + 控制台错误回给模型），已整刀回滚。

## 0.5 落地状态（10-16，Mac）

| 门 | 结果 |
|---|---|
| `ds4_asst_test` 预览段 | 35/35（口令子树、`..` / 编码 `..` / 点开头段 / 软链跳出 / 改一位口令 / 过期 → 404；CSP 头；表满挤最老；`/api/assistant/file` 取 html 仍是纯文本） |
| `make asst-e2e`（系统 Chrome） | 预览段 8 项全过，原文见 fable5 10-16 |
| `make test` | 全绿（含 linecount） |

未提交，未同步 spark。

## 1. 模型看到的东西

**不加新工具。** 网页就是 HTML，模型写 HTML 是训练里见得最多的东西（`make_video` 收 HTML 也是这个理由）。
改的只有 `write_file` 的描述（场景知识唯一合法入口），在原描述后面加：

> An .html file is also shown rendered under your answer, sandboxed with no network: put CSS, JS and images inline
> or as files in the same folder (or its subfolders) and reference them by relative path; external URLs (CDN
> scripts, web fonts, remote images) do not load there.

为什么无网络而不是放行 https：① 预览里放行外网，预览的行为就随网络变（spark 连不上 GitHub、手机经穿透走国内网，
CDN 时通时不通），坏了没法归因；② `make_video` 的页面已经是"no network, external URLs fail"，两个收 HTML 的地方一个口径；
③ 安全边界（§3）靠 `connect-src 'none'` 封死出网，放行 https 就要在"哪些能出"上开口子。要图表就让模型自己写 Canvas/SVG。

## 2. 用户看到的东西（`web/studio.html` 文件卡）

`.html/.htm` 的卡：`文件名 · [下载] · [预览]`，点开预览后多一个 `[打开]`。

- **预览**：先 `POST /api/assistant/preview {path}`（走现有鉴权与 `X-DS4-Assistant` 头）拿一个 `/p/<口令>/<文件名>` 地址，
  卡里展开 `<iframe sandbox="allow-scripts allow-modals">`，高 `min(70vh, 640px)`、白底。**每次展开都重新要口令**（60 分钟过期，不缓存）。
- **打开**：真的 `<a href="/p/…" target="_blank">`（不用 `window.open`：iPhone 装成 App 后开出来是白页，`docs/assistant.md` §13.1 实撞）。
  新标签页里同一个 CSP 响应头照样生效——锁靠的是服务端响应头，不是 iframe 属性。
- 其余文件类型的卡不变；html 不再出"源码文本"预览（要源码就下载）。页面不加任何旋钮。

多文件网页（`site/index.html` + `site/app.js` + `site/img/logo.png`）：同一个口令放行**网页所在目录及其子目录**，整站能跑。

### 2.1 为什么要口令：它就是登录的延伸

只有登录了的工作台页面才拿得到 `/p/` 地址（签口令的接口要登录票）。`/p/` 本身不查登录票，是因为登录票存在
localStorage（10-12 用户定，不用 cookie），浏览器打开一个地址（iframe / 新标签页）时带不上它，服务端看不到你登录了。
所以由已登录的页面先换一个口令拼进地址——效果就是"登录了才能看"。

## 3. 安全：为什么这条路可以给 `text/html`

威胁不是外人，是**网页自己的脚本**：网页是模型写的，模型读过的网页可能带提示注入；网页在你登录的浏览器里打开，
它的脚本就跑在你的登录身份下。若和工作台同源，脚本读得到 localStorage 里的登录票，能以你的身份打 `/api/assistant/*`
（读全部会话、改 policy、调云手机，含付款）。所以 `/api/assistant/file` 永远把 html 按 `text/plain` 给（`asst_file.c` 文件头），
预览另开一条路，四层挡：

| 层 | 怎么做 | 挡什么 |
|---|---|---|
| **不透明源** | 响应头 `Content-Security-Policy: sandbox allow-scripts allow-modals`（iframe 元素再带同样的 `sandbox` 属性，双保险） | 文档的 origin 是 `null`，不是工作台：读 `localStorage` 抛错、`parent.document` 抛 SecurityError |
| **封出网** | 同一个头里 `default-src 'none'; script-src 'self' <Host> 'unsafe-inline' 'unsafe-eval'; style-src …; img-src 'self' <Host> data: blob:; font-src …; media-src …; connect-src 'none'; form-action 'none'; base-uri 'none'; frame-src 'none'; worker-src 'none'; manifest-src 'none'` | fetch / XHR / WebSocket / sendBeacon / 表单 / 嵌框全没有；资源只能来自同一条 `/p/<口令>/` 路。`'self'` 在不透明源里照样认同主机（Chrome 实测：把 Host 改坏、只剩 `'self'` 时同目录 js 和图照样加载）；显式的 `<Host>`（取自请求 Host 头，去掉换行等字符防头注入）是给没测过的浏览器（Safari）的第二道 |
| **口令** | 128 位随机、内存表最多 64 条（满了挤最老的）、60 分钟过期、只对网页所在目录的子树有效（`realpath` 后还得在子树里，软链跳出去不行；相对路径里 `.` 开头的段一律拒——`.trash/.web/.phone` 是内部目录）；**网页直接放在工作区根上时只放行根上的文件，不下钻子目录**（根下面是 `inbox/` 你上传的照片和别的活儿）。口令不对 / 过期 / 越界一律 404 | 没登录的人扫不到工作区文件 |
| **不漏口令** | `Referrer-Policy: no-referrer`、`Cache-Control: no-store`、`X-Content-Type-Options: nosniff` | 页面里点外链跳走时 Referer 不带 `/p/<口令>/…` |

**诚实的边界**：没有哪条 CSP 指令能挡文档**导航自己**到外站（`navigate-to` 已从标准删除），所以页面脚本理论上能把
`location.href` 里的口令拼进外站 URL 带出去。拿到口令的人在过期前能读的就是上面那个范围（网页所在目录子树；在根上时只有根上那一层）；
会话、记忆、policy、云手机一个都碰不到。写在这里，不写成"隔离"。

## 4. 代码落点

```
src/assistant/asst_preview.c   口令表 + asst_preview_issue(签) + asst_preview_get(纯判定 + 取字节 + 安全头, 单测直接调) + asst_preview_serve(写完整响应并关 fd)
                               + asst_web_ctype(按扩展名的网页资源类型; make_video 的渲染源 tool_video_srv.c 也改用它, 一件事一处)
src/assistant/asst_api.c       POST /api/assistant/preview {path} → {url, ttl_s}(过 asst_authorized)
src/studio/studio_main.c       takeover 钩子认 /p/ 前缀交给 asst_preview_serve(它要自己的响应头, handle 的统一响应写不出来; 登录闸只拦 /api 与 /metrics, /p/ 的闸是口令)
src/assistant/asst_tools.c     write_file 描述加 §1 那句
web/studio.html                fileCard: html 卡 [预览]=iframe sandbox + [打开]=<a target=_blank>; 口令每次展开时 POST 取
tests/asst/t_asst_preview.c    离线单测(进 ds4_asst_test)
tests/asst/e2e_device.mjs      网页预览段(§5)
```

不碰：`asst_memory.c` 提示词、`asst_run.c` 循环、`asst_file.c` 的下载路、守门。

## 5. 门

- `ds4_asst_test` 预览段（离线）：口令发出 → 同目录文件 200 且 `text/html`；`../`、`%2e%2e/`、点开头文件、软链跳出、NUL、裸口令 → 404；
  改一位的口令 → 404；过期 → 404；POST → 405；响应头含 `sandbox allow-scripts allow-modals`、`connect-src 'none'`、`no-referrer`、显式 Host 源；
  Host 头里的 CR/LF 进不了 CSP；工作区根上的网页取不到 `inbox/` 与别的子目录；`.trash` 里的 / 非 html / 工作区外 / 不存在的签不出口令；
  第 65 个口令把最老的挤掉；`/api/assistant/file` 取同一个 html 仍是 `text/plain`。
- `make asst-e2e`（系统 Chrome）：工作区放一个会"越狱"的网页（脚本依次试读 localStorage、碰 parent.document、fetch `/api`、加载同目录 js 与子目录图），
  页面里调真的 `fileCard()` 点预览 → iframe 里读到 `ls:blocked parent:blocked origin:null js:loaded fetch:blocked img:loaded`；
  "打开"的新标签页同样 `ls:blocked origin:null fetch:blocked`；经一个把 Host 改成 `rewritten.invalid` 的小代理打开，资源照样加载（验 `'self'`）；
  同一口令越出目录 → 404。
- `make test` 全绿（含 linecount）。
- 待跑（spark 真模型 + 手机 https）：一句"做一个介绍 X 的单页网页" → `write_file` 出 .html，终稿写路径，手机上点"预览"渲染正常、"打开"整页正常；原文贴 fable5。

## 6. 明确不做

- 模型截图自检（`view_page`）：10-16 用户定不要，已回滚。
- 把 `/p/<口令>/` 当分享链接发给别人：口令 60 分钟过期、放行整个目录，它不是发布机制。要发给别人走 §7 的 `/s/`。
- 构建链（npm / 打包 / 框架）：沙盒无网络，模型写的就是浏览器直接跑的 HTML/CSS/JS。
- 预览里放行外网、可视化编辑、服务端渲染、页面旋钮（宽度切换、缩放）、`write_page` 工具。

## 7. 分享链接（10-18 加）

**怎么用**：网页的文件卡上点"分享"，系统分享面板里拿到的是一个链接（`https://<你打开工作台的地址>/s/<口令>/<文件名>`），
发进微信/钉钉，对方点开就是渲染好的网页，不用登录，链接不过期。不再把 .html 文件本身发出去——对方收到文件多半只能看源码。
电脑上没有分享面板的浏览器（Firefox）改成把链接复制到剪贴板。

10-18 用户："分享的应该是URL不是文件"。起因：10-16 起模型写了文件只在回答里写路径、不再每次调 `share`，
自动出的文件卡只有下载/预览，网页就分享不出去了。现在终稿点名出的卡自己带分享键：网页发链接，别的文件（pdf/pptx/图）照旧发文件。

**链接从哪来**：页面 `POST /api/assistant/share_link {path}` → 服务端拍快照 → 回 `/s/<口令>/<文件名>`，页面拼上 `location.origin`。
所以**经穿透（公网 https）打开工作台时点分享，链接就是公网地址**；在 `localhost` 或局域网 IP 上点，拼出来的地址别人打不开。

**为什么是快照，不直接指向工作区**：链接发出去就收不回来，拿到它的人能按文件名去试同目录的别的文件。
网页在工作区根上时，同目录就是你的 pptx、md 和 `inbox/`。所以点分享那一刻把网页拷进 `~/.ds4/assistant/shares/<口令>/`，链接只读这份拷贝：

| 网页在哪 | 拷什么 |
|---|---|
| 子目录（`site/index.html`） | 整个 `site/`，跳过 `.` 开头的文件和软链；多文件网页照样能跑 |
| 工作区根（`card.html`） | 它自己 + 它用 `src=` / `href=` / `poster=` / `url(` 按文件名引用到的根上文件；别的根上文件、子目录一概不拷 |

- 之后模型再改网页，已经发出去的链接内容不变；同一版本（文件清单 + 内容的哈希一样）再点分享，复用同一个口令，不重复拷。
- 网页所在目录超过 64 MiB 或 2000 个文件就拒绝分享，提示"把网页和它的文件单独放一个文件夹"。多半是网页被放进了 `inbox/` 这类大目录。
- 安全头和预览完全一样（§3 的表，同一个函数 `asst_web_serve`）。你自己点开链接时，网页脚本照样读不到工作台的登录票。
- 撤销：删掉 `shares/<口令>/` 目录，链接立刻 404。页面上目前没有撤销按钮。

**Safari 要点两下**：链接要等服务端拍完快照才有，点了再要。Chrome 的点击手势有 5 秒宽限，能直接弹出分享面板；
iPhone Safari 的手势当场过期，第一下会提示"链接已生成, 再点一次「分享」"，第二下直接弹出。
没有改成"画卡时就先要链接"，因为那样每张网页卡都会拍一份没人要的快照。

**代码**：`src/assistant/asst_share.c`（拍快照 / 复用口令 / `/s/` 路由），`asst_preview.c` 里抽出的 `asst_web_serve`（两条路共用的
解码、越界判定和安全头），`asst_api.c` 的 `share_link`，`studio_main.c` 的 takeover 钩子，`web/studio.html` 的 `sharePage`。

**门**：`ds4_asst_test` 的 preview 段 52/52（新加 17 项：快照范围、软链和 `.` 文件不拷、`..` 越界、同版本同链接、
改过出新链接且旧链接内容不变、根上网页只带引用到的文件、坏口令 404、POST 405）；
`make asst-e2e` 119/119（新加：卡上三个按钮、分享面板拿到的是 `/s/` 链接且不带文件、再点同一个链接、
没登录的新浏览器打开链接能看到分享那一刻的网页、同目录 js 与图都在、照样锁着）。

