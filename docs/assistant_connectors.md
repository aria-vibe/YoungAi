# 助理连接器(connectors)方案稿

2026-10-19。读者 = 第一次接触这个项目的人。先说怎么用、出错会怎样; 为什么这么设计放后面。
用户原话: "在 skill 下面设计一个连接器的功能, 比如有个连接器名称, 账号, 密码, 地址, 比如填当前电脑里面交易 agent 的 MongoDB 的地址、账号、密码,
填好了有个测试连接的按钮, 也有开关按钮, 连接器可以是多个。这样我就可以问一个交易 agent 的数据库数据, 然后用 HTML 绘制出来";
写到一半追加: "不仅仅是 MongoDB, 主流数据库都要执行, 所以连接器最好支持下拉选项"。
技能方案 `docs/assistant_skills.md`; 助理总纲 `docs/assistant_arch.md`、细节 `docs/assistant.md`。

## 0. 一句话

**连接器 = 你在技能页登记的一个数据源(名字 / 类型 / 地址 / 端口 / 库名 / 账号 / 密码)。** 登记好、开关打开,
助理的工具表里就多一个**只读**的 `query_db` 工具, 描述里列着每个开着的连接器叫什么、是什么库、怎么查;
你问"交易 agent 最近推荐了什么", 它自己调 `query_db` 查, 把查到的数据写成一页 HTML(内联 SVG / canvas 画图), 回答下面直接渲染出来。

五种库, 类型下拉选: **MongoDB · PostgreSQL · MySQL/MariaDB · Redis · SQLite 文件**。线协议全是纯 C 自己写的(§5), 不链任何客户端库、不调 mongosh/psql/mysql 命令行
(spark 上没有, 本项目零依赖); SQLite 没有线协议, 在沙盒里跑工具箱已有的 `sqlite3 -readonly`。
密码只进 `secrets/`(0600), **页面拿不回, 模型永远看不到**; 只读是代码保证的(§3), 不是提示词里的一句"请不要改库"。

## 1. 怎么用

工作台"技能"tab(侧栏第三格)里, "已装技能"卡下面多一张卡**"连接器"**:

| 操作 | 怎么做 | 落盘 |
|---|---|---|
| 新建 | 点"新建", 选类型, 字段按类型出现(端口自动填默认值): MongoDB 要 地址/端口/库名/账号/密码/认证库; PostgreSQL/MySQL 要 地址/端口/库名/账号/密码; Redis 要 地址/端口/库编号/账号(ACL 用户, 可空)/密码; SQLite 只要文件路径 | `~/.ds4/assistant/connectors/<名>.json` + 密码 `secrets/conn_<名>` |
| 测试连接 | 表单下面的按钮, **不用先保存**: 拿表单当前值连一次, 结果一行写在按钮旁 | 不落盘 |
| 保存 | 密码框留空 = 不改已存的密码(已存的占位写"已保存, 留空不改"); 填了就覆盖 | 同上 |
| 开关 | 列表每行右边一个滑动开关, 和技能的一样 | json 里 `enabled` |
| 删除 | 确认后 json 挪进 `trash/connectors/<名>-<秒>.json`; **密码直接删不进回收站** | — |

名字规则同技能(小写字母/数字/连字符), 最多 24 位(密码的 secret 名是 `conn_<名>`, secrets 名字上限 32)。

测试连接成功长这样(10-19 Mac 实测): `✓ MongoDB 7.0.40 · quant_trading · 13 collections · 15 ms`。
MySQL 是 `MySQL 8.4.x · hpias · 11 tables`, PostgreSQL 是 `PostgreSQL 16.x · clouddesk · 11 tables`, SQLite 是 `SQLite 3.x · /路径 · 2 tables`。失败就是 §4 里的那些字, 和模型调工具看到的同一句。

交易 agent 这个例子填: 名字 `quant` · MongoDB · `127.0.0.1`(spark 上填 Mac 的局域网地址 `192.168.2.203`, 27017 本来就映射在所有网卡上, 10-19 从 spark 握手验过) · `27017` · `quant_trading` · 账号密码空(它的 mongod 没开鉴权, 空就不走 SCRAM)。

然后到聊天页问: "quant 库里最近 30 条推荐, 按日期画成图"。审计流水里会看到:
`query_db(connector=quant, what=collections)` → `query_db(what=fields, collection=recommendations)` → `query_db(what=find, …, sort={"date":-1}, limit=30)` → `write_file(推荐走势.html)`, 回答下面出渲染卡。

## 2. 模型那头: 一个工具 `query_db`, 描述随连接器变

工具表里这一行由 `asst_conn_build_tool` 按当时开着的连接器拼(和 `use_skill` 列技能一样的渐进披露); **一个开着的都没有, 工具表里就没有这一行**。
描述 = 通用做法("先列再看字段再查, 不猜名字; 结果截了去 db/ 读; 画图写内联 SVG 的 html") + 每个连接器一行 + 用到的每种库一句怎么查(`src/db/db.c` 的 `DB_KINDS[].hint`)。
`what` 是用到的种类的并集(枚举), 不同库各认自己那几个, 调错了回 `error: for a MongoDB connector what must be one of: collections,fields,find,count,distinct,aggregate`。

| 库 | what | 发给库的 | 给模型什么 |
|---|---|---|---|
| MongoDB | collections | listCollections + 逐个 count | 名字 · 条数 |
| | fields | find 抽样 20 条 | 键名 · 类型 · 在几条里出现 · 样例值(**不写死任何集合名或字段名**, 换一个库照样用) |
| | find / count / distinct | find+getMore / count / distinct, 都带 maxTimeMS 10 s | 一行一条 JSON; limit 默认 50 最多 500 |
| | aggregate | aggregate + 游标 | 同 find |
| PostgreSQL / MySQL / SQLite | tables | pg_class / information_schema.tables / sqlite_master | 表名 · 类型 · 行数估计 |
| | fields | information_schema.columns / pragma_table_info | 列名 · 类型 · 可空 · 默认 |
| | sql | 一条只读 SQL, 自动包成 `SELECT * FROM (<sql>) AS q LIMIT n` | 第一行列名, 之后一行一条 JSON |
| Redis | keys | SCAN MATCH pattern + 逐个 TYPE | key · 类型 |
| | get | 按 TYPE 选 GET / HGETALL / LRANGE / SMEMBERS / ZRANGE WITHSCORES / XRANGE | 值(JSON) |
| | info / command | INFO / 一条白名单里的文本命令 | 原文 / 回复 JSON |

Mongo 的 `filter` / `sort` / `projection` / `pipeline` 都是 **MongoDB 查询语法的 JSON**(字符串或对象都收); 结果里 ObjectId 写成 `{"$oid":"…"}`、日期写成 `{"$date":"2026-10-18T01:30:00Z"}`,
模型原样抄回 filter 就能匹配(写成裸字符串就对不上 `_id` 了)。`{"$numberLong":"…"}` 也认; JSON 里的整数编成 int32/int64, `2.0` 会变成 `2`(Mongo 数值比较跨类型, 不影响查询)。

结果给模型的部分截到 16 KB(和网页正文一页同量, 在整行处截), **全量落工作区 `db/<连接器>-<集合或 what>-<时刻>.json`**(一行一条), 末尾一句指路 `read_file … start=` 分页或 `run` 里处理。
画图不加任何新工具: `write_file` 一个 `.html`, 回答下面的预览卡(`asst_preview.c`, CSP sandbox, 无网)渲染。
没网 ⇒ **CDN 的 Chart.js 加载不了**, 模型得用内联 SVG / canvas 画(write_file 的描述里已经写了这条; 要现成图表库得 vendored 进仓库 —— 外部资源, 要用户点头, 第一版不装)。

## 3. 只读是代码保证的

| 库 | 第一道(发请求前) | 第二道(服务器自己拒) |
|---|---|---|
| MongoDB | `mongo_query.c` 里只有 listCollections / find / count / distinct / aggregate(+getMore / killCursors)的代码, 没有 insert/update/delete/drop; pipeline 里 `$out` `$merge` 拒, filter/pipeline 里 `$where` `$function` `$accumulator` 拒(服务器上跑 JavaScript) | 每条命令 `maxTimeMS: 10000` |
| PostgreSQL | `sql_guard.c` 白名单: 只放行 SELECT / WITH … SELECT / SHOW / EXPLAIN / DESCRIBE / VALUES / TABLE; 多条语句(分号)拒; WITH / EXPLAIN 后面藏 INSERT/UPDATE/DELETE/… 拒(可写 CTE、EXPLAIN ANALYZE 真执行) | 会话 `SET default_transaction_read_only = on`(连函数里的写也拒) + `statement_timeout = 10000` |
| MySQL / MariaDB | 同一份白名单 | `SET SESSION TRANSACTION READ ONLY`(InnoDB 拒一切写) + `max_execution_time = 10000`(MariaDB 叫 max_statement_time, 两个都试) |
| SQLite | 同一份白名单(不放 PRAGMA: writable_schema 能改状态) | `sqlite3 -readonly` 开文件 + 沙盒里只挂那一个文件为只读根 |
| Redis | 命令白名单(GET/MGET/HGET*/LRANGE/SMEMBERS/ZRANGE*/XRANGE/SCAN/KEYS/INFO/TYPE/TTL/DBSIZE…); CONFIG 只许 GET, CLIENT 只许 LIST/INFO | — |

共同的: limit 封顶 500; 密码只在握手那一刻从 `secrets/conn_<名>` 读进栈, 用完清零, 任何错误字符串里不带它。
守门(`sbx_guard.c`)新类别 `db`: **放行不审批**(连接器是用户自己填的, 命令只读), **执行后置 `taint_priv`**(库里是用户自己的数据, 和读工作区外的私有文件同级:
之后上网只能"仅这次", 单测验过); 不置 `taint_web`。连接器关着或删了, 模型凭旧名字硬调 → `error: this connector is turned off by the user` / `no such connector`。

## 4. 出错会怎样

都是具体的字, 页面"测试连接"和模型看到的是同一句(10-19 实测):

- 地址不通: `cannot connect to 127.0.0.1:27017 (connection refused)`; 5 秒连不上: `… (timed out after 5 s)`。
- 账号密码错: MongoDB `authentication failed for user cfo on admin (code 18; wrong password, or the user lives in another authSource)`; PostgreSQL `authentication failed for user x (wrong password, or pg_hba.conf does not allow this host)`; MySQL `… (code 1045: …)`; Redis `authentication failed (wrong password or user)`。
- 老 Mongo(< 4.0)只有 SCRAM-SHA-1: `server offers only SCRAM-SHA-1 for this user; enable SCRAM-SHA-256 on the server (MongoDB 4.0+)`。不做 SHA-1(MongoDB 4.0+ 默认就是 SHA-256)。
- PostgreSQL 还在用 md5: `server wants md5 authentication; set password_encryption = scram-sha-256 and the pg_hba.conf method to scram-sha-256`。
- MySQL 用别的插件(如 sha256_password): `server wants authentication plugin …, which is not supported (use mysql_native_password or caching_sha2_password)`。
- 填了 `mongodb+srv://` 或要 TLS: `TLS and +srv are not supported; use host:port on the local network`。第一版只做明文 TCP(本机 / 局域网)。
- 库名写错: Mongo 不报错, listCollections 为空 → 测试连接写 `database quant_tradin has no collections (check the name; the server has: admin, config, local, quant_trading)`。
- 查询语法错: 服务器的报错原样回给模型(`server: … (code N)` / sqlite `Error: in prepare, no such table: nope`), 它改了再调。
- 写请求: `$out is not allowed: this tool is read-only …` / `only read-only statements are allowed: SELECT, WITH … SELECT, SHOW, EXPLAIN, DESCRIBE` / `one statement at a time (no ';' between statements)` / Redis `DEL is not allowed: this tool is read-only (…)`。
- 结果太大: 末尾 `…[16384 of N bytes shown; the full result is saved as db/quant-news_items-20261019T153000.json: read_file it with start= to page, or process it with run]`。
- sqlite3 没装: `sqlite3 is not installed on this machine (the sandbox toolbox has no sqlite3)`。
- 名字不合规: `connector name must be 1-24 lowercase letters, digits or hyphens`。

## 5. 协议: 纯 C, 不链客户端库, 不调命令行

需要的密码学原语仓库本来就有(`src/common/ds4_sha256`: SHA-256 / HMAC / PBKDF2; Web Push 已经在用), 只补了 MySQL 要的 SHA-1 和 RSA 公钥加密(在 `mysql_auth.c` 里, 全仓只有它用):

```
MongoDB   TCP → hello(问 saslSupportedMechs) → 有账号: saslStart(SCRAM-SHA-256 client-first) → saslContinue(client-final, 校验 server signature)
          → OP_MSG(opcode 2013, 一个 kind-0 section = 命令 BSON 带 $db) → 解回包 ok / cursor.firstBatch / cursor.id → getMore 到 limit → killCursors
PostgreSQL StartupMessage(v3, user/database) → 'R' 鉴权: SASL(同一份 SCRAM 代码, 用户名留空) 或 cleartext → 'Q' 简单查询 → 'T' 列定义 'D' 行 'Z' 就绪; 'E' 报错
MySQL     握手 v10(nonce 8+12, 插件名) → HandshakeResponse41 → 插件应答: mysql_native_password = SHA1(pw) XOR SHA1(nonce‖SHA1(SHA1(pw)));
          caching_sha2_password = SHA256(pw) XOR SHA256(SHA256(SHA256(pw))‖nonce) 先试快路, 服务器回 04 要全量 → 请公钥(02) → RSA-OAEP(SHA-1) 加密 (pw‖0) XOR nonce 发回;
          AuthSwitchRequest(FE) 换插件重算 → COM_QUERY(03) → 列数 / 列定义 / EOF / 行(lenenc) / EOF; 不开 DEPRECATE_EOF
Redis     RESP2: *N $len … → + - : $ * 五种回复; AUTH [user] pw; SELECT n
SQLite    沙盒子进程 execve sqlite3 -readonly -batch -json <文件> <SQL>(SQL 走 argv, 不经 bash 不用转义); HOME=/ (不设会往输出插一行警告)
```

RSA 只有 c = m^e mod n, e = 65537: 32 位 limb 大数, 逐位"乘加模"(每步减一次模), 2048 位一次加密毫秒级, 不值得写蒙哥马利。
BSON 编解码(`bson.c`)解码每一步先查长度再读: 回包是网络上来的字节, 一个错的长度字段不能让我们读越界。
全部明文 TCP, 不做 TLS / +srv / 压缩 / 连接池(一次调用连一次、用完关; 本机/局域网握手几毫秒)。

单测不打库也能过: BSON 用规范文档里的两条字节序列逐字节对; SCRAM-SHA-256 用 RFC 7677 §3 的完整向量(user/pencil → ClientProof、ServerSignature 逐字节对);
SHA-1 用 FIPS "abc" 和 1000 个 a; 大数用 4^13 mod 497 = 445 和 2^65537 mod 1e9+7 = 947173645; RSA-OAEP 对一把 2048 位测试公钥出 256 字节且两次不同。

## 6. 代码落点(每文件 ≤500 行, `make linecount` 守)

```
src/db/db.h / db.c              种类表 DB_KINDS(kind / 显示名 / 默认端口 / 表单字段 / what 枚举 / 工具描述里那句) + 分发 db_query / db_probe; 加一种库 = 这里一行 + 一份协议文件
src/db/db_net.c                 共用 TCP: 5 s 连接超时 + 20 s 收发超时, 具体错误字
src/db/bson.h / bson.c          BSON 编解码, JSON ↔ BSON($oid / $date / $numberLong), 日期 ↔ ISO
src/db/mongo.h                  MongoDB 三文件的接口
src/db/mongo_wire.c             OP_MSG 收发 · hello · SCRAM 握手(saslStart/saslContinue) · 密码用完清零
src/db/mongo_scram.c            SCRAM-SHA-256 纯算术(client-first / client-final / 校验 server signature / 标准 base64); PostgreSQL 共用
src/db/mongo_query.c            六种 what + 游标 + 白名单扫描($out/$merge/$where/$function/$accumulator) + maxTimeMS + 结果 JSON 行 + 测试连接
src/db/sql_guard.c              三家 SQL 共用: 只读白名单 / 包 LIMIT / 结果表 → 文本
src/db/pg_wire.c                PostgreSQL v3 + SCRAM/cleartext + 只读会话 + tables/fields/sql + 测试连接
src/db/mysql_auth.c             SHA-1 / 两种 scramble / DER 公钥解析 / 大数 / RSA-OAEP
src/db/mysql_wire.c             MySQL 握手 + 鉴权(含插件切换、sha2 全量) + COM_QUERY 文本结果集 + 只读会话 + 测试连接
src/db/redis.c                  RESP2 + AUTH/SELECT + keys/get/info/command + 命令白名单 + 测试连接
src/assistant/asst_conn.c       连接器: 名字规则 / <名>.json 读写 / 密码走 asst_secret_set("conn_<名>") / 开关 / 删除进 trash / 测试连接 / query_db 的 schema build / 页面接口
src/assistant/tools/tool_db.c   query_db 执行: 参数 → 连接器开着吗 → what 合不合这种库 → 密码进栈 → db_query 或 sqlite(沙盒子进程) → 头 16 KB 给模型 + 全量落 db/
src/assistant/asst_tools.c      注册表一行 { "query_db", NULL, DB, …, asst_conn_build_tool }; 新执行点 DB
src/sandbox/sbx_guard.c         query_db: cls=db, 放行; asst_guard_mark 置 taint_priv
src/sandbox/sbx_toolbox.c       asst_toolbox_has("sqlite3")
src/assistant/asst_store.c      asst_init 建 connectors/
src/assistant/asst_api.c        GET/POST /api/assistant/connectors
web/studio.html                 技能页第二张卡: 列表(名字 · 类型 地址:端口/库 · 开关) / 表单(类型下拉 + 按类型显隐字段 + 端口默认值) / 测试连接 / 保存 / 删除
tests/asst/t_asst_db.c          §8 的离线项 + 本机有 Mongo 才跑的 live 段
tests/asst/e2e_device.mjs       连接器页 17 项
```

接口:
```
GET  /api/assistant/connectors   → {kinds:[{kind,label,port,fields}], connectors:[{name, kind, host, port, db, user, auth_db, path, has_password, enabled, updated}]}   ← 没有 password 字段
POST /api/assistant/connectors   {op:"save", name, kind, host, port, db, user, auth_db, path, password?}   ← password 缺省 = 不改
                                 {op:"delete", name} | {op:"enable", name, on} → 同 GET
                                 {op:"test", name?, kind, host, port, db, user, auth_db, path, password?} → {ok:true, line, ms} | {error}
                                   ← 带 name 且 password 缺省时用已存的密码测
```

## 7. 以后

- ClickHouse: HTTP 接口 + `readonly=1` 参数(服务器端只读), 几十行; SQL Server(TDS)/ Oracle(TNS)协议太大, 不做。
- TLS: 要自己写 TLS 1.3 或引库, 都不小; 局域网/本机先不做。
- Chart.js vendored 进工作区可引用的路径(外部资源, 要用户点头)。

## 8. 门(10-19 Mac 全过)

1. 离线: `./ds4_asst_test` 新增 `asst db` **80/80**: BSON 两条规范金标 + 往返 + 坏文档; SCRAM RFC 7677 向量(含错 nonce、错签名); SHA-1 / scramble / 大数 / RSA-OAEP; SQL 白名单 12 例 + 包 LIMIT; Mongo 禁键 4 例;
   连接器 json/密码分离(GET 不带 password、0600、留空不改)、种类校验、工具表有/无/枚举/what 并集、关着硬调拒、what 按种类校验、禁键在连接前拒、守门 cls=db + taint_priv + 之后上网要问、sqlite 要 path、删除进 trash 且密码没了;
   live(本机 127.0.0.1:27017 有 Mongo 才跑): 握手 + 集合表 + `$out` 拒 + 错账号报 authentication failed。全套 `ds4_asst_test` 其余套件不变, `make linecount` 过。
2. 真库(Mac, 不经模型, `./ds4_asst_test --tool query_db …`):
   - MongoDB 7.0.40(交易 agent 的容器): collections 13 个集合条数与 mongosh 一致(news_items 25495 / daily_bars 11199); fields 抽 recommendations 21 个字段带类型样例; find 带 sort/projection/limit; count 带 filter(603619 419 条); aggregate 按日期分组; distinct; `$out` / `$where` 在连接前被拒。
   - MySQL 8.4(本机 hpias 容器, caching_sha2_password **走了全量鉴权的 RSA 路**): tables 11 张; fields; sql 带 ORDER BY + limit; `WITH … REPLACE()` 函数名不误伤; UPDATE 拒。
   - PostgreSQL 16(本机 clouddesk 容器, SCRAM-SHA-256): tables 11 张; fields; sql count/min/max; 带 LIMIT 的 SELECT 再包一层也对; `EXPLAIN ANALYZE DELETE` 拒。
   - SQLite(临时建的 lite.db): tables 含视图; fields 5 列; sql 中文与单引号转义; DELETE 拒; `PRAGMA …; INSERT` 拒; 跑完条数没变。
   - Redis: 本机没有 Redis 服务, **没有真跑过**(协议按 RESP2 规范写, 只有离线单测); 要验得起一个 redis(docker 镜像 = 外部资源, 待用户点头)。
3. 页面(Mac, `make asst-e2e` 真 ds4-train + Chromium): 连接器 17 项全过 —— 下拉五种; 选 PostgreSQL 端口 5432 且认证库/文件字段藏起; 选 MongoDB 回 27017; 测试连接真连上 `✓ MongoDB 7.0.40 · quant_trading · 13 collections · 15 ms`;
   保存落盘 + 密码 0600 + 接口无 password; 留空再保存密码没动; 列表行; 开关; 删除进 trash 密码没了。
4. 真模型(spark, 10-19 20:5x, 连接器 quant 指向 Mac 的 `192.168.2.203:27017`, `tests/asst/asst_send_probe.sh` 走页面同一条路): 问"查一下 quant 库里最近的推荐记录, 按日期统计每天推荐了几只、平均预期收益多少, 画成一张图给我"
   → 176 s, 工具序列 `query_db(collections)` → `query_db(fields, recommendations)` → `query_db(aggregate, $group 带非法累加器)`(服务器报错原文回去) → `query_db(aggregate, $group n/$sum, avgExpected/$avg)` → `render_image`(内联 SVG 双图, 1600×900 PNG);
   终稿表 20260903/04/08 各 10 只、平均预期收益 5.76% / 7.39% / 5.18%, 与 Mac 上不经模型的 aggregate(0.05762 / 0.07393 / 0.05175)逐个对得上; 没猜集合名字段名, 没编数。原文在 fable5。

## 9. 明确不做的

- 写库(insert/update/delete/drop)和"审批后可写": 交易 agent 的库是生产数据, 助理只看不碰。真要改, 用户在交易 agent 里改。
- 连接器带进技能文件夹 / 技能里声明"我用哪个连接器": 技能是可导入的文件, 密码不能跟着文件走; 技能正文可以写"用 query_db 查 quant"(名字而已)。
- 为交易 agent 写死集合名、字段含义的提示词或技能(铁律 10-11: 禁案例定制代码)。模型用 `fields` 自己摸, 用户想教它就自己写一份技能。
- 调 mongosh / psql / mysql 命令行当后端: spark 上没有, 输出要解析, 沙盒里又没网。
