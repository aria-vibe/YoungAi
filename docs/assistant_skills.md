# 助理技能(skills)方案稿

2026-10-14。读者 = 第一次接触这个项目的人。先说怎么装、怎么用、出错会怎样; 为什么这么设计放后面。
落地过程与判决进 `fable5.md`; 助理总纲 `docs/assistant_arch.md`、细节 `docs/assistant.md`。

## 0. 一句话

**技能 = 一个文件夹, 教助理"某类活怎么干"。** 放进 `~/.ds4/assistant/skills/<名>/`, 里面必有 `SKILL.md`,
旁边放模板、示例、参考资料。助理平时只看见每个技能的名字和一句描述; 任务对上了, 它自己调 `use_skill`
读全文, 再按里面说的步骤、用**它自己的工具**去干。

格式照 **Agent Skills 开放标准**(Anthropic 2025 年公开, Claude Code / 各家 agent 通用)的 `SKILL.md` frontmatter 约定。
**但有一条硬限制(10-14 用户定): 技能不许带脚本。** 文件夹里装不进脚本文件, `SKILL.md` 和随附的 md/txt 里也不许有
任何脚本语句(命令行、代码块)。所以别处的技能只有"纯说明"的那部分能进来; 像 Anthropic 自己的 `pptx` 技能(靠 pptxgenjs /
python 脚本干活)会在导入时整份被拒(§1.1 有实测)。仓库自带两份合规技能, **ds4-train 起来时默认装上**(§1.2):
`skills/pptx/` 走本机的 `write_document` 出 .pptx(六种版式 + 五套主题, 版式由内容形状决定, 见 docs/assistant_office.md §3.4);
`skills/finance/` 教助理用 `market_data` 工具查股票、指数、财报、公告、资讯(§7);
`skills/lottery/` 教助理用 `lottery_data` 工具查体彩福彩开奖、竞彩足球篮球的赔率和赛果(§8);
`skills/research/` 教助理做科研的活: 找文献做综述、读论文、复现方法、统计分析画图、写论文和审稿回复(§9, 不带新工具, 全靠 visit_page / run / render_image / write_document)。

## 1. 怎么装

两条路, 都在工作台"技能"tab(侧栏第三格, 手机上在"更多 → 助理"组):

| 路 | 操作 | 落盘 |
|---|---|---|
| 新建 | 点"新建", 填名字(小写字母/数字/连字符), 页面给一份带 frontmatter 的模板, 改完保存 | `skills/<名>/SKILL.md` |
| 导入文件夹 | 点"导入文件夹", 选一个**带 SKILL.md 的技能文件夹本身**(不是上一级) | 文件夹里每个文件逐个传上来, 技能名 = 文件夹名净化成小写连字符 |

直接 `cp -r` 到 `~/.ds4/assistant/skills/` 也行, 页面和模型读的都是盘。

`SKILL.md` 长这样:

```markdown
---
name: weekly-report
description: 写部门周报用这个: 按部门汇总本周 jira 导出的 csv, 产出 docx
---
# 周报

1. 读 reference/tpl.md 拿版式
2. run: python3 /abs/skills/weekly-report/scripts/sum.py inbox/jira.csv > summary.md
3. write_document 出 docx, share 交出去
```

**`description` 是助理决定"要不要用这个技能"的唯一依据**, 写清楚什么任务用它。正文写步骤、用哪些工具、
脚本怎么跑。没有 frontmatter 也行: 第一行非空文字当描述。

### 1.1 不许带脚本: 什么会被拒, 怎么拒

| 东西 | 判法(确定性, `asst_skill_check` / 扩展名白名单) | 结果 |
|---|---|---|
| 文件 `.py .sh .js .ts .rb .html .xml …` 或没扩展名 | 不在白名单 `md txt csv tsv json yaml yml png jpg jpeg gif webp pdf docx xlsx pptx` | 这个文件不收, 其余照装 |
| 任何文件以 `#!` 开头 | 读前两个字节 | 不收 |
| md/txt 里的围栏代码块(行首三个反引号或波浪线) | 逐行 | **整份文件拒**, 报 `line N: fenced code block` |
| md/txt 里以命令开头的行或行内代码: `python python3 bash sh node npm npx pip curl wget chmod rm sudo make gcc java ruby perl php go cargo exec eval source export soffice ffmpeg run ./…` | 去掉列表符/表格竖线/反引号后看第一个词(区分大小写: 句首的 Make/Run 是英文单词不算) | 整份拒, 报 `line N: command 'python'` |
| 行内代码里提到脚本文件 `… .py .sh .js .ps1 …` | 反引号段内 | 整份拒, 报 `script file reference` |
| `<script` | 不分大小写 | 整份拒 |

导入文件夹时 `SKILL.md` 先传: 它被拒, 整个技能不装(不留半拉子文件夹); 随附文件哪个被拒跳过哪个, 页面逐个提示。

**实测(10-14, Mac 上的 ds4-train 真服务)**: 把 Anthropic 公开的 `skills/pptx/SKILL.md`(20.8 KB)POST 进来 →
`400 {"error":"SKILL.md rejected: line 15: command 'python' in inline code; skills may not contain commands or scripts"}`
(第 15 行是它的"读内容"那格: `python scripts/thumbnail.py deck.pptx`)。仓库自带的 `skills/pptx/` 两个文件都过, 装进来后
`use_skill` 工具表里出现 `- pptx: 做 PPT / 幻灯片 / …`。

出错会怎样:
- 名字带大写/空格/下划线 → 保存报 "skill name must be 1-64 lowercase letters, digits or hyphens"; 导入时页面自动净化。
- 文件夹里没有 `SKILL.md` → 列表里不出现(拷到一半的也不算), 导入时提示"选的文件夹里没有 SKILL.md"。
- 正文里写了一句 `bash x.sh` → 保存整份被拒, 弹出 "SKILL.md rejected: line N: command 'bash' …"; 把步骤改成"用 write_document / read_file …"这类工具名就过。
- 技能改了, **进行中的对话看不到**: 工具表每轮重拼所以下一轮就带新描述, 但模型已经读过的旧正文不会自动刷新; 新对话最干净。

### 1.2 仓库自带的技能: 默认装上

不用做任何事: ds4-train 每次启动, 把 `<仓库根或发布包>/skills/` 下的每个技能装进 `~/.ds4/assistant/skills/`, 启动日志一行
`ds4-train: 自带技能 installed finance;`。装法和页面导入是同一个口(§1.1 的检查照过), 自带的也不许带命令。

账本 `~/.ds4/assistant/skills.seeded`(一行 "名字 指纹") 记"装的是哪一版", 决定以后怎么办:

| 情形 | 启动时做什么 |
|---|---|
| 没装过 | 装, 记账 |
| 装过、你没改过、仓库出了新版 | 旧的挪进 `trash/skills/<名>-<秒>-replaced`, 装新版 |
| 你在技能页改过 | 不动(你的版本优先, 仓库升级也不盖) |
| 你删了 | 不再装回来 |
| 账本里没有、但盘上已有同名(早先手动导入过) | 内容和仓库一样就补记账(以后跟着升级); 不一样就不动、不记账 |

出错会怎样: 自带技能里混进一行命令 → 日志里 `xxx/SKILL.md skipped: line N: command '…'`, 这个技能不装, 别的照装。
改过的技能想回到仓库版: 页面上删掉它, 再从仓库 `skills/<名>` 用"导入文件夹"装一次(删除会被账本记住, 不会自动回来)。
发布包里带 `skills/`(release_pack.sh), 解压直接起也会装。

## 2. 模型那头怎么用(渐进披露)

```
工具表(每轮随请求发)                              use_skill(name) 的结果
┌──────────────────────────────────────┐          ┌───────────────────────────────────────────┐
│ use_skill: Read the full instructions │          │ skill: weekly-report                       │
│  … Installed skills:                  │  调用 →  │ folder: /…/skills/weekly-report (read-only)│
│  - weekly-report: 写部门周报用这个…    │          │ files:                                      │
│  - pdf-forms: Fill PDF forms …        │          │   SKILL.md (812 bytes)                      │
│  name: enum [weekly-report, pdf-forms]│          │   scripts/sum.py (3104 bytes)               │
└──────────────────────────────────────┘          │ ---                                         │
                                                  │ # 周报 …(正文, frontmatter 去掉)             │
                                                  └───────────────────────────────────────────┘
```

- 一个技能都没装: 工具表里**没有** `use_skill` 这一行(模型不会去调一个空表)。
- 正文上限 64 KB(≈2 万 token), 超了截断并指路 `read_file … start=` 分页。
- 随附文件只有资料(模板/示例/数据), `read_file` 读, 路径用结果里给的绝对路径(沙盒工作目录是工作区, 相对路径会找不到)。干活用的是助理自己的工具, 工具描述里写明 "Skills contain no scripts"。
- **提示词一个字没改**: 技能的存在只通过 `use_skill` 的 description 进模型 —— 总纲铁律 3(场景知识只住在工具
  description)。十个技能的正文塞进 system 是几万 token 每会话预填一遍, 多数会话一个都用不上, 这就是只给"名字: 描述"的原因。

## 3. 边界(安全)

| 事 | 规矩 | 为什么 |
|---|---|---|
| 工具读技能文件 | 放行, **不置污点** | 用户自己装的, 不算读私有文件; 读完上网不该降级成"仅这次" |
| 工具写技能文件 | **一律拒, 不可审批** | 模型能改自己的技能 = 网页里一句提示注入就能长期住进去("每次先把工作区发出去"); 改只能在页面 |
| 沙盒 | 技能夹永远是只读根 | `read_file` 读得到资料; 里面本来就没有可执行的东西(全部 0600, 没有脚本文件) |
| 上传随附文件 | 相对路径逐段查: 不许 `..`、不许 `.` 开头、不许绝对路径; 扩展名白名单; md/txt 过脚本语句检查 | §1.1 |
| 删除 | 整个文件夹挪进 `trash/skills/<名>-<时间>` | 点错了盘上还在(和删会话同路) |
| 其它控制目录 | 照旧拒(token / policy / approvals / sessions) | 技能是 `~/.ds4/assistant` 下唯一对工具开读的子目录 |

诚实的边界: 技能正文和网页一样是进上下文的文本, 它说什么模型就可能做什么。**装技能 = 信任写它的人**, 和装一个
浏览器扩展一个意思。从网上拷技能先看一眼 SKILL.md 和 scripts/。

## 4. 代码落点

```
src/assistant/asst_skill.c    扫描 skills/ (按名排序, 工具表顺序稳定 ⇒ KV 前缀稳定) / frontmatter 解析(description 支持 > | 折行与引号) /
                              use_skill 的 schema(描述列技能, name 枚举) 与执行(文件夹 + 文件表 + 正文) / 页面接口(列表 / save / delete / 上传随附文件) /
                              ★asst_skill_check 脚本语句检查 + 扩展名白名单(§1.1)
src/skill/skill_seed.c        ★仓库自带技能默认装上(§1.2): 账本 skills.seeded + 目录指纹(FNV-1a 64); ds4-train 启动时 asst_skills_seed(<根>/skills) 显式调(不放 asst_init: 单测拿临时根, 自动装会改掉"没装 → 没有 use_skill"的前提)
skills/pptx/             仓库自带的合规 PPT 技能: SKILL.md(怎么把主题变成 .pptx, 走 write_document) + reference/example-deck.md(12 页示例稿, 六种版式、照片、图标、示意图、图表各出现至少一次)
skills/finance/          仓库自带的金融技能(§7): SKILL.md 一份, 认标的 → 按问题选查法 → 读数常识 → 回答怎么写
src/assistant/asst_tools.c    注册表一行 { "use_skill", NULL, MEM, …, asst_skill_build_tool }; exec 分发到 asst_skill_load
src/assistant/asst_guard.c    use_skill 放行(cls=skill); 技能夹读放行写拒; asst_guard_mark 读技能不置污点; asst_guard_spec 技能夹只读根
src/assistant/asst_store.c    asst_init 建 skills/ 并设 asst_skills(Landlock 只给存在的路径立规则)
src/assistant/asst_api.c      GET/POST /api/assistant/skills; POST /api/assistant/skill_file?name=&path=(原始体)
web/studio.html               "技能"tab(侧栏第三格; app 外壳在"更多 → 助理"): 列表 / 新建(模板) / 编辑保存 / 文件表 / 添加文件 / 导入文件夹(webkitdirectory) / 删除(ask 确认)
tests/asst/t_asst_logic.c     test_skills 52 项: 名字规则、脚本语句检查 10 例(代码块/#!/命令行/行内命令/脚本文件名/./x/curl/<script>/两例纯说明放行)、
                              保存带命令整份拒且不留文件夹、.py/无扩展名/#! txt/含 npm 的 md 全拒、pdf 收、自带 pptx 技能装得进并能 use_skill、frontmatter 折行、
                              没装无工具/装了有枚举、正文去 frontmatter、守门读放行写拒不置污点、沙盒只读根、沙盒里真读、三个接口、删除进 trash
tests/asst/e2e_device.mjs     技能页 8 项: 导航第三格、空表、模板、保存落盘、列表行、接口描述、删除进 trash、收起
```

接口:
```
GET  /api/assistant/skills                       → {root, skills:[{name, description, md, files:[{path,size}]}]}
POST /api/assistant/skills      {op:"save", name, md} | {op:"delete", name}   → 同 GET
POST /api/assistant/skill_file?name=<技能>&path=<相对路径>   原始体 = 文件字节 → {ok:true}
```

## 5. 门

- 离线(Mac, 10-14 已过): `ds4_asst_test` 216/216(含 test_skills 36) + `make asst-e2e` 全过(含技能页 8 项) + `make linecount`。
- 真模型(spark, 待跑): 装一个带脚本的技能 → 提一个它覆盖的任务 → 审计里第一条工具调用是 `use_skill`, 之后按 SKILL.md 的步骤走且 `run` 跑到了 scripts/ 里的脚本; 提一个它不覆盖的任务 → 没调 `use_skill`。原文进 fable5。

## 6. 明确不做的

- **技能带脚本 / 正文带命令**(10-14 用户定): 技能只说"怎么干", 手是助理自己的工具。要新能力就加注册表一行(总纲 §2.3), 不往技能里塞程序。
- 模型自己写技能(`skill-creator` 那类): 和 §3 第二行冲突; 要做也只能走"模型写到工作区 → 用户在技能页导入"。
- 技能里的 `allowed-tools` 等扩展字段: 守门是确定性的, 不按技能开关权限。
- 从网上一键安装(市场/URL): 外部资源必须用户先同意(铁律 10-13), 导入文件夹就是那个"同意"动作。

## 7. 金融技能 finance + 行情工具 market_data(10-18)

用户原话: "参考我的交易智能体, 实现一个金融 skill 默认加入当前项目的 skill, 先把股票财经这些基础查询的能力封装进去"。

**分工**: 技能不许带脚本, 取数这只手只能是工具。所以拆两半:
- `market_data` 工具(`src/assistant/tools/tool_market.c`, 注册表一行, 守门按 web): 取数、把东财的 f2/f3 编号字段翻成中文、金额换成亿/万、标币种、算好 K 线区间涨跌。
- `skills/finance/SKILL.md`: 认准标的、哪类问题用哪几种查法、读数常识(交易时间、手、涨跌停、前复权、财报累计口径)、回答格式。

数据源和交易智能体(`~/git/quant_trading_flow-main`)是同一家东方财富公开接口, 不要 key, 请求经火狐走:

| what | 接口 | 覆盖 | 一次给什么 |
|---|---|---|---|
| search | searchapi suggest | 全市场 | 名字/拼音/代码 → 最多 10 条候选 + secid |
| quote | push2 ulist(未登录会 302 到 push2delay) | A 股/港股/美股/指数/基金 | 最多 20 只: 最新价、涨跌、开高低收、量额、振幅换手量比、市盈率市净率市值、更新时间(北京时间) |
| history | push2his kline | 同上 | 日/周/月 K(前复权), 默认 30 根最多 250, 带区间涨跌和最高最低 |
| financials | datacenter `RPT_F10_FINANCE_MAINFINADATA` | 沪深京 A 股 | 默认 4 期最多 12: 营收/归母/扣非及累计同比、单季同比、EPS、BPS、每股经营现金流、ROE、毛利率、净利率、负债率 |
| news | search-api-web(JSONP, 按相关度) | 任意关键词 | 默认 10 条最多 30: 时间、标题、媒体、摘要、链接 |
| notices | np-anotice | 沪深京 A 股 | 默认 10 条: 日期、标题、栏目、原文链接 |
| market | push2 ulist 固定九个指数 | 上证/深证/创业板/沪深300/科创50/恒生/道指/纳指100/标普 | 一行一个, 沪深指数带上涨/下跌/平盘家数 |

和交易智能体不一样的两处(都是 10-18 实测撞出来的):
- 财务没用它那张 `RPT_PCF10_FINANCEMAINFINADATA`: 不管 pageSize 都只回最新一期, 看不了趋势。换成 F10 页面同源的 `RPT_F10_FINANCE_MAINFINADATA`, 一期一行, 还多了单季同比和净利率。
- 资讯按相关度不按时间排: 按时间排, 前几条全是顺带提一句"茅台"的别家新闻。

代码怎么认: `1.600519` 原样; `sh600519`/`sz000001`/`bj830799` 按前缀; 6 位纯数字 6/9/5 开头沪市其余深市(北交所在东财也是 `0.`)——
**所以 000001 是平安银行**, 要上证指数得给名字或 `sh000001`; 名字、港股 5 位、美股代码走代码搜索取第一条。几只一起查用逗号(中文逗号、顿号也认)。

出错会怎样:
- 名字对不上任何标的 → 结果末尾 `not resolved (left out above): xxx: no stock, index or fund matches`, 其余照出; 一只都没解析出来就是 error。
- 港美股查 financials/notices → error 里直说只有 A 股, 指路 news / web_search。
- 接口改版或被风控 → `error: unexpected response from Eastmoney …: <原文开头 200 字>`, 模型照实说取不到。
- 港股、美股的金额是港元、美元: 结果第一行写 `币种 港元/美元`。没标之前苹果"总市值 50125 亿"会被当成人民币讲。
- 行情可能有延时(港美股常见 15 分钟), 每只都带"更新于 … 北京时间"; 周末查到的就是上个交易日收盘。

验证:
- 离线: `./ds4_asst_test` 里 `asst market` 71 项(10-18 Mac 与 spark 都 71/71): 代码规则、六个解析器(夹具是当天真返回)、整条执行路(取数换成读夹具)、守门、技能账本五种情形、仓库 `skills/` 两个技能都过脚本检查。
- 真接口(要火狐和外网): `./ds4_asst_test --tool market_data '{"what":"quote","symbols":"贵州茅台,00700,AAPL"}'`, 七种 what 在 Mac 和 spark 上都跑通(一次 3~13 秒)。
- 真服务: ds4-train 起来日志 `自带技能 installed finance;`, `GET /api/assistant/skills` 列出 finance。
- 待跑: spark 真模型问"茅台最近怎么样" → 审计里先 `use_skill(finance)` 再 `market_data`, 终稿数字和工具结果逐个对得上。

## 8. 彩票技能 lottery + 彩票工具 lottery_data(10-18)

用户原话: "实现一个彩票 skill 默认加入当前项目的 skill, 先把近期查询封装了, 包含体彩, 福彩, 到各大赛事"。
拆法和 §7 一样: `lottery_data` 工具取数(`src/assistant/tools/tool_lottery.c` 开奖 + `tool_lottery_match.c` 竞彩), `skills/lottery/SKILL.md` 教用法。

怎么用(不经模型直接跑, 要火狐和**中国大陆出口**):

```
./ds4_asst_test --tool lottery_data '{"what":"draws","game":"ssq"}'                 # 双色球最近 5 期 + 最新一期奖级
./ds4_asst_test --tool lottery_data '{"what":"draws","game":"dlt","issue":"26118"}' # 大乐透某一期
./ds4_asst_test --tool lottery_data '{"what":"matches","league":"英超,西甲"}'        # 竞彩足球在售 + 赔率
./ds4_asst_test --tool lottery_data '{"what":"results","sport":"basketball"}'       # 竞彩篮球最近 3 天赛果
```

| what | 数据源(都是官方) | 覆盖 | 一次给什么 |
|---|---|---|---|
| draws | 体彩中心 `webapi.sporttery.cn` getHistoryPageListV1; 中彩网 `cwl.gov.cn` findDrawNotice | 福彩 ssq 双色球 / fc3d 3D / qlc 七乐彩 / kl8 快乐8; 体彩 dlt 大乐透 / pl3 / pl5 / qxc 七星彩 / sfc 胜负彩(含任九) / bqc 半全场 / jqc 进球彩 | 默认 5 期最多 30, 或 issue 指定一期: 号码按玩法分段(红蓝球、前后区、前六位+第七位)、销售额、奖池; 最新一期带全部奖级(注数 + 每注税前奖金, 写到元)、一等奖中奖地、公告链接; 足彩三种带每场对阵比分 |
| matches | `uniform/{football,basketball}/getMatchCalculatorV1` | 竞彩足球、篮球在售场次 | 场次编号、联赛、主客队与排名、开赛时间; 足球胜平负 + 让球, 篮球胜负 + 让分 + 大小分 |
| results | `uniform/{football,basketball}/getUniformMatchResultV1`(一页 100 场) | 同上, date 往前 days 天(默认 3 最多 7) | 比分(篮球翻成主:客)、半场或各节和加时、胜负、按比分算的让球/让分结果; 没完赛的写"还没有赛果" |

league 只在本地按中文简称/全称筛, 不发出去; 期号和日期拼进网址前校验成纯数字/`YYYY-MM-DD`。所以守门直接放行(cls=web), 读过私有文件后也不问。

为什么不用百度或 500 彩票网这类转载站: 它们给的是 HTML 卡片, 改版就挂, 也是二手数。官方接口给 JSON, 字段稳定, 号码就是开奖公告上的。

**出错会怎样**(都是 10-18 实测撞到的):
- **机器走海外出口 → 全部拿不到。** 两家防火墙只放中国大陆地址: Mac 那天出口在乌克兰, 连体彩主页都回 HTTP 567 "请求已被拦截"。
  工具认得出拦截页, 报 `error: sporttery.cn refused the request (…overseas proxy or VPN)`, 模型照实说取不到。spark(南京电信)全通。
- **直接打接口也会被拦**: curl 或火狐地址栏直开接口地址, 体彩回 "Access Restricted"、福彩回空页。所以工具先开官网首页, 在页面里 `fetch` 接口。
- 福彩接口带上网上常见的 `systemType=PC` 参数 → 防火墙直接断连, 页面里报 `NetworkError`。不要加。
- 篮球比分接口原文是**客队在前**(皇马主场 119:84 赢, 接口写 `"84:119"`), 工具已翻成主:客; 改解析时别丢了这一步, 不然主客输赢全反。
- 足球赛果行里那组 h/d/a 赔率没注明是哪个玩法的, 工具不输出, 免得模型拿错的赔率讲"爆冷"。
- 旧路径 `jc/football/getMatchResultV1` 在页面里也回"禁止访问", 只有 `uniform/` 下的赛果接口能用。
- 3D 官方不给奖级明细(固定奖金玩法), 结果里写明; 七乐彩一等奖没人中时不出"中奖地"一行。

验证(10-18):
- 离线: `./ds4_asst_test` 里 `asst lottery` 44 项(Mac 与 spark 都 44/44): 夹具是当晚 spark 从两家官方取的真返回(`tests/asst/fixtures/lottery/`), 覆盖七个解析器、期号 5↔7 位互换、数量上限、日期跨月、非法期号/日期/玩法在发请求前拒、拦截页识别、守门、技能过脚本检查并能 use_skill。
- 真接口: spark 上 11 个玩法 draws + 足球/篮球的 matches 与 results 全跑通, 一次 3~4 秒; 号码与当天官方公告一致(双色球 2026119 期 12 22 23 25 27 30 + 04)。
- 默认装上: spark 生产 `train_ui_spark.sh restart` 后日志 `ds4-train: 自带技能 installed lottery;`, 账本 `skills.seeded` 多一行 lottery。
- 待跑: 真模型问"今晚双色球开了什么""明天英超有哪些比赛" → 审计里先 `use_skill(lottery)` 再 `lottery_data`, 终稿号码与工具结果逐个对得上。

## 9. 科研技能 research(10-19)

用户原话: "设计一个科研skill, 辅助科研人员, 论文研究, 技术实现, 包括但是不限于数据等等常见学科, 然后默认装到spark上面"。

**和金融、彩票不一样: 没有新工具。** 科研要的手(查网页、读文件、跑程序、画图、出文档)助理都已经有了, 缺的是"怎么组合、什么能信、什么不能写"。
所以只有 `skills/research/SKILL.md`(三条底线 → 按任务选做法 → 找文献用哪个源 → 读论文 → 复现与实现 → 数据分析 → 写作 → 回答怎么写)
加 `reference/templates.md`(综述 / IMRaD 论文 / 实验记录 / 审稿回复 / 开题与基金立项依据 / 组会页次 / GB/T 7714、APA、IEEE 引用格式)。

**三条底线写在最前面**, 因为这是模型做科研最常犯的错: ① 引用必须是这次 `visit_page` 打开过的(凭记忆写的题目、作者、DOI 经常不存在);
② 数字只从数据来; ③ 没跑过的不说跑过。

**文献源是 10-19 在 spark(南京电信出口)用 `--tool visit_page` 逐个实测过的**, 技能里只写通的:

| 源 | 通不通 | 技能里怎么用 |
|---|---|---|
| arXiv 搜索页 / 摘要页 / HTML 全文 | 通(5~7 s) | 搜索页按公布日期倒序; 2023 年底以后的论文有 HTML 全文, 直接读正文 |
| OpenAlex `api.openalex.org/works` | 通, JSON 原样进结果 | 全学科 + 引用数 + 按年筛; 没 key 时和同一出口的人共用每日免费额度(返回里有提示), 够查 |
| Crossref `api.crossref.org/works` | 通 | DOI 查元数据、按题目找 DOI |
| Europe PMC REST | 通 | 生物医学(PubMed 全部 + 预印本), 有 PMC 号的能开全文 XML |
| Bing(`web_search`) | 通 | 定关键词、找 PDF 和代码仓库、中文文献 |
| arXiv Atom 接口 export.arxiv.org | **火狐里 20 s 超时**(curl 0.6 s 就回, 是火狐把 Atom 当下载不算加载完) | 技能明写别用, 走搜索页 |
| PubMed 网站搜索页 | 通但正文全是筛选器 | 不推荐, 走 Europe PMC |
| Google Scholar / Hugging Face papers / GitHub | 打不开或超时 | 技能明写 Scholar 别试; 代码仓库打不开就如实说 |
| Semantic Scholar 接口 | 429 限流 | 不写 |

**沙盒里能用什么(spark 实测)**: `run` 的 python3 只有标准库 —— numpy、sympy、networkx、torch 装在用户 `~/.local`, 沙盒看不见(`--tool run` 里导入 numpy 报 ModuleNotFoundError);
pdftotext、jq、sqlite3 在; pandoc、LaTeX、R 没有。所以技能里统计检验那段写明: 标准库没有检验函数, p 值自己按公式算或用置换检验 / 自助法(只要 random);
xlsx 没库时请用户另存 csv 或用 zipfile 解。要让沙盒看见科学库, 得装到系统目录(apt 的 python3-numpy 这类)再重启主进程(工具箱只探一次), 这是用户的事, 技能里让模型把缺的库写在回答里。

出错会怎样:
- 技能正文提了 pdftotext、网址、统计检验名, 都不在命令表里, 脚本检查放行; 但写作时不许把 `run` 放在行首或表格格首(`run` 在命令表里), 技能里一律写"用 `run` 工具"。
- 某个源在用户的出口打不开(海外出口开 arXiv 没问题, 开 OpenAlex 也没问题; 国内别的出口未测) → `visit_page` 报超时, 模型按技能换别的源或如实说。

验证(10-19):
- 离线: `./ds4_asst_test` 的 `asst lottery` 里加 2 项(技能没有专属工具, 测试住在彩票那份的自带技能段): 仓库 skills/ 四个技能全过脚本检查、use_skill 列出 research 带描述、正文装进来且列出 reference/templates.md。Mac 45/45, 全套绿, linecount 过。
- 真服务: spark 重启 ds4-train 后日志 `自带技能 installed research;`, `GET /api/assistant/skills` 列出 research。
- 待跑: 真模型问"帮我查一下滑动窗口注意力量化方向最近有什么论文" → 审计里先 `use_skill(research)` 再 `visit_page` 开 arXiv 搜索页, 终稿每篇都有打开过的链接。
