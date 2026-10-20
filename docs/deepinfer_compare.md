# 对照 DeepInfer: ds4 还有哪些可优化的地方(方案稿, 2026-10-20)

读者: 第一次看这份稿的人。先说结论, 再说依据, 最后是每一项怎么做、怎么验、不这么做会怎样。
本稿只是方案, **没有改任何代码, 没有发车**。

## 0. 结论

DeepInfer(github.com/deepseek-ai/DeepInfer, 2026-10 开源)是 DeepSeek 自己线上跑 V4.1 Flash 的推理框架。
它的主体是**集群**手段(64~128 卡 Wide-EP、专家负载均衡、P/D 分离、3FS 存 KV), 这些在一台 128 GB 的 Spark 上没有对象。
把单机能用的机制逐条对照, ds4 已经有 **12 项里的 9 项**(有 3 项还比它细), 真正能搬的只有 4 件事, 按收益排:

| # | 事 | 场景 | 预计收益 | 风险 | 工作量 |
|---|---|---|---|---|---|
| 1 | indexer 打分核默认走张量核(`--idx-mma` 转正) | 长上下文(106k~1M), 三路 | 106k 预填 133 → 96 s(−28%); 1M 三路解码外推 90 → 72 ms/步 | 四块相加顺序不同, 不保证逐字节同(12k/106k 实测逐字节同) | 小: 改默认 + 跑门 |
| 2 | 预填最后一块的解码器段只算尾部 K 行(官方"解码器有界回放"的回放宽度) | 短提示 TTFT, 工作台多轮 | 单块提示(≤2048)预填 −35~40%; 12k 提示 −7% | 09-29 试过 K=128 判负(投机接受 1.87 → 1.41); 官方线上 K=140 也是近似 | 中: 一处分块规则 + 窗口可见域 |
| 3 | 三路合批的投机改成全批统一预算(DeepInfer 的置信排序算法) | 3 路并发 | 让 N=3 也能开投机; 收益要量, 估每路 +10~20% | 无正确性风险(k 只影响快慢) | 小: 调度器 100 行 |
| 4 | 合批路发/等拆分(GPU 跑下一步时主机吐上一步) | 3 路并发 | 每步 −2~3 ms(48 ms 里的空转 3.2) | 无 | 小 |

不搬的(有对象但已判负, 或没对象): 专家负载均衡 / EP / CP / DP、Mega MoE、FP4 专家原件、KV 存储集群、提示查表草稿(10-07 判负)、草稿器蒸馏(10-07 判负)、
结构化输出(JSON schema 约束采样, 是功能不是优化, 见 §4)。

**单路 48 t/s 这档, DeepInfer 没有任何能再往上推的手段**: 它的投机(DSpark + 置信调度)与 ds4 同款, ds4 的调度器还多了每请求自量成本与在线校准 ρ。
10-20 的账(memory `decode_speed_account_0922`)不变: 单路剩下的只有每 token 字节 / 第二台 Spark / 多路合批。

## 1. DeepInfer 是什么(只写与 ds4 有关的)

- 服务对象: `deepseek-ai/DeepSeek-V4.1-Flash-1017`(与 ds4 现役底座同一份权重)。硬件: NVIDIA SM100 ×4(预填)/ ×64(解码), 昇腾 950。
- 精度(Q&A 文档"低精度计算"): 主 KV = per-16 e4m3; indexer K = MXFP4(per-32 ue8m0); MoE 专家权重 = MXFP4。
  ds4 对应: 压缩 KV e4m3 scale/16、索引键 fp4、窗口 KV fp8(32 块) —— **KV 精度与官方一致**; 专家是 12/13 位 VQ(为装进 128 GB), 比官方更小, 这是 ds4 的选择不是差距。
- 线上数(Q&A): 单集群 155 万亿 token/天; EP64 下 context 131k、每 token 接受 75% 时 τ=3.25。
- 它开源的部分**不含** KV 存储、P/D 分离、负载均衡(README 明写)。

## 2. 逐机制对照

| DeepInfer 机制 | 在哪 | ds4 现状 | 判 |
|---|---|---|---|
| DSpark 草稿器(3 塔, 目标层 37/38/39 的注意力输入, markov 偏置头, confidence 头) | `models/mtp/dflash.py` | 同款: `core_v41_draft.c`, 目标层/markov/confidence 全从 GGUF 元数据来; 10-07 加了 markov 偏置缓存(命中 59%) | 持平 |
| **置信调度的变长验证**: 每轮按 confidence 的累乘(前缀存活率)排序, 全批共享一个 total_q 预算, 预算由"吞吐曲线 + 滑动中位数拟合的时间"选 | `csrc/engine/mtp_varlen.hpp`, `schedule_mtp_varlen.py` | 单路有: `core_draft_sched.c` 按 产出/成本 选 k, 成本每请求自量(最小二乘), 采样下 ρ 在线校准 —— **比它细**; 合批路没有: `allow_spec = n <= 2`, 行数按 8 封顶从 k 最大的路削 | 单路持平; **合批路可搬(§3.3)** |
| 验证 = 拒绝采样(min(1, p/q) + 残差) | `tilelang_ops/sampler.py` | 同: `cuda_v41_sample.inc.cu` 设备采样核(09-29) | 持平 |
| 零开销调度: GPU 跑第 N 步时 CPU 做第 N+1 步的调度与第 N−1 步的后处理 | `csrc/engine/zos.cpp`, `worker/model_runner.py` | 单路有: `v41_graph_launch/wait`、`v41_draft_launch/wait`(10-07); 合批路没有: `server_sched_v41.c` 的 `ds4_v41_multi_round` 同步跑完才 emit | **合批路可搬(§3.4)** |
| 解码整步 CUDA graph(主模型 + DFlash 各一张, 按批大小分档) | `worker/cuda_graph_runner.py` | 同: 纯解码 n=1 图、验证批 n=2.. 各一张、草稿图按补窗口行数分档、合批整步图 | 持平 |
| **解码器 SWA 有界回放**: consumer 层(L20~39)预填只算末尾 keep_tokens=140 个 query(≥ 窗口 128 + 回退 6+); 它们的全局 KV 由边界层投影全长 hidden 得到 | `models/v4_modules/yoco.py`, `input_metadata/yoco.py`, `prepare_tensor_input_v4.cpp:203` | 半有: CED(`core_v41_forward.c` 中间块到 L20 收工), 但**最后一块 2048 行跑满 40 层**; 09-29 试 K=128 判负 | **可搬, 要重新量(§3.2)** |
| 编码器 SWA 有界回放(前缀缓存命中但窗口 KV 丢了时只重算 128 个) | Q&A "KVCache 重算与命中" | 不需要: `core_v41_kvreuse.c` 检查点带整份窗口环 | 无对象 |
| 跨层共享: kv 源层 {2,8,14,20} / index 源层 {2,8,14,20,24,28,32,36} / 候选块源 L20(2048 块 × 8) | `cross_layer.py` | 同: `is_kv_source[]`、C2 候选紧凑(09-30) | 持平 |
| indexer 打分走张量核(DeepGEMM `fp8_fp4_paged_mqa_logits`, fp4 K) | `index_attn.py` | 有但默认关: `--idx-mma`(`cuda_v41_indexer_mma.inc.cu`), 12k/106k 门逐字节同 | **转正(§3.1)** |
| topk 用 DeepSelect(专用 radix 选择, 2~20× torch.topk) | `deep_select.interface.topk` | 09-16 换成 256 桶 radix select, 单 block/行; 09-30 账: 解码态 1M 的剩余税在 topk/候选块这类"一行一个 block"的核 | 多 block topk 是 ds4 自己的下一刀, 不是 DeepInfer 独有(§3.1 一起量) |
| 批不变性(同一请求在不同批里结果逐字节同) | `DEEPINFER_BATCH_INVARIANT` | 天然有: 三路与单路逐字节同(09-30 md5 b55e8ce4) | 持平 |
| 结构化输出(xgrammar: regex / JSON schema / EBNF → 每步 token 位掩码, 草稿位也掩) | `structure_output/`, `sampler.py:_get_so_mask` | 没有(`src/server` 无 response_format / json_schema) | 功能差距, 不排进本稿(§4) |
| 分词: special token 之间的普通文本段并行编码 | Q&A "Tokenizer 处理与加速" | 单线程; 没量过 1M 提示的分词耗时 | 先量再说(§4) |
| Engram 表放 CPU 内存/显存, DeepEP 按哈希跨卡取 | `engram.py` | 盘上分片 + io_uring 并行读(128 GB 装不下表) | 无对象 |

## 3. 四件事怎么做

### 3.1 indexer 张量核打分转正(`--idx-mma` 改默认开)

**现状**: 核在 `src/cuda/cuda_v41_indexer_mma.inc.cu`(200 行), 开关在 `cuda_v41_indexer.inc.cu:98`(默认 0), CLI `--idx-mma`。
09-30 的门(fable5 8864~8896 行): 12k 单路 64 步 md5 与标量核同; 106k 单路/三路 md5 同; 106k 预填 133.5 → 96.3 s; 106k 三路 57.5 → 50.2 ms/步。
当时没转正的唯一理由: 块内 s8 mma 精确, 但**四块相加顺序**与标量核的蝶形树不同, 理论上不保证逐字节同, 等用户点头。

**DeepInfer 给的新依据**: 官方线上就是张量核打分(`deep_gemm.fp8_fp4_paged_mqa_logits`), 且 indexer K 是 MXFP4。
也就是说"标量核的累加顺序"并不比张量核的更接近官方, 两者都只是某一种舍入顺序; 逐字节同这把尺在这里量的是"和我们自己的旧核同", 不是"和官方同"。

**做法**: `g_v41_idx_mma` 默认 1, `--idx-mma` 改成 `--no-idx-mma`(留判决用)。元数据形状闸不动(32 头 × 128 维之外直接拒)。
**已做(10-20 晚)**: A/B 逐字节同(1397 字节), 9210 token 预填 8.7 → 8.3 s; 判决份 15,360 位学生 logits 张量核 vs 标量核 cmp 逐字节同, 五指标逐字相同, 打分 45.1 → 42.1 s。记录在 fable5 10-20 晚节。

**门(10 分钟级)**: ① `speed-bench/decode_prefill_agree_spark.sh`(解码路 vs 预填路逐位一致率, 10-20 新尺)两档各跑一次, 必须与现役一样高;
② 判决份五指标(`v41_judge.sh`, 现役编程侧车)Same top / KLD 不退; ③ `kernel_ab_spark.sh` 12k 请求 512 token 逐字节同(09-30 已同, 复证)。
长门: 106k 三路 `multi_probe_spark.sh "1 3" 32 …`, 读数对 09-30 的 50.2 ms。

**不这么做会怎样**: 1M 三路每步 ~90 ms(~33 t/s 总), 预填 106k 多付 37 s/路。这是用户定的 3×1M 场景里最大的一刀, 已经做完只差转正。

### 3.2 预填最后一块的解码器段只算尾部 K 行

**官方口径(Q&A "KVCache 重算与命中")**: "精确重建要 20 个 decoder 层跑最后 2560 个 token; 改为每次 prefill 在 decoder 层只重算 prompt 末尾 128 个 token,
产出仅供 decode、不进前缀缓存, 总 prefill 计算量近乎减半。" DeepInfer 默认 keep_tokens=140(≥ 窗口 128 + 回退 6), 可见域 = 滑窗 ∩ 末尾 140 的矩形
(`prepare_tensor_input_v4.cpp:222~274`), 并提供 mask/compact 两种模式的逐位等价测试(`examples/yoco/`)。

**ds4 现状**: `v41_prefill_chunk` 让最后一块 ≥ 128 行、≤ 2048 行, 40 层跑满。所以 ds4 已经是近似(不到 2560), 只是比官方宽。
09-29 试过"最后一块正好 128 行"(`core_v41_forward.c:363` 判负存档): CFO 14103 token 预填 15.1 → 13.8 s, 但投机接受 1.87 → 1.41/5、解码 43.5 → 38.3, 文本第 1 字就变。
**为什么掉**: 草稿器目标层是 37/38/39(全在解码器段), 它补窗口要最近 128 个位置在这三层的注意力输入; 回放只有 128 行时, 这 128 行自己的窗口被截在块头, 三层隐态与训练时的分布差得远, 草稿器就看不准。
官方的草稿器同样吃这段近似隐态(DeepInfer `dflash_targets_in_consumer` 时 dflash_input 也切成 compact 行), 线上就是这么跑的, 接受率 75% 是在这个口径下量的。

**做法**(不动 CED, 只动最后一块): 最后一块仍整块跑编码器段(L0~19)与 L20 的 KV 投影; L20 之后只对末尾 K 行继续跑(hc/x 等按行偏移取视图, 窗口环只写这 K 行, 注意力核的 `win_from[l]` 钳到块尾 − K)。
K 不写死: 从 `K = 窗口 + 草稿块 + 1`(= 134)起, 按 128 的倍数向上扫 134 / 256 / 512 / 1024 / 2048(2048 = 现状), 每档量三样:
首 token 的 logits 对现状的 KL、投机接受率、预填秒数。选"接受率与首 token KL 在噪声内的最小 K"。
收益账: 解码器段 20 层 ≈ 预填一块的 45%(专家 + 注意力 + hc 各半), 单块提示省 ~40%; 12k 提示六块只省最后一块 ⇒ ~7%; 工作台多轮(KV 复用后每轮只填增量, 通常 < 2048)每轮 TTFT 都吃到 ~40%。

**门**: ① `decode_prefill_agree_spark.sh`(解码路与预填路一致率)不退; ② `kernel_ab_spark.sh` 同请求投机接受直方图与 t/s(接受率是这刀的主判, 不是副作用);
③ 判决份五指标不退(前缀是打分路, 不受影响, 作为对照); ④ 三趟鹈鹕真实请求收口位 Σp(`think_close_probe_spark.sh`)不退。
**不这么做会怎样**: 工作台每轮多等 ~0.3~0.8 s(一块 2048 行 ≈ 2 s 的 40%); 长提示不痛。
**风险**: 这是用"官方线上同款近似"换速度, 与 09-29 的判决相反; 发车要用户点头(铁律: 质量门不为速度让步 / 新方案不被旧判决否决, 两条都适用, 所以由用户裁)。
**已做(10-20 晚, 用户定"K 用官方配置" ⇒ 默认 140)**: 机制 + `--decoder-replay K` + 扫尺落地; 三条真实请求上 K=512 输出逐字节同、接受同, 预填 −2.5%(9210 token)/ −12~20%(≈1k token); K=134 短提示 −30% 但翻了 2/3 条文本、接受 −0.2。
默认 140 的 A/B(9210 token): 预填 8.3 → 7.7 s, 接受 2.46 → 2.44, 文本第 25 行起分叉。`--decoder-replay 0` 退回整块。全表在 fable5 10-20 晚节。

### 3.3 三路合批的投机: 全批统一预算(搬 DeepInfer `MTPVarlen::schedule`)

**DeepInfer 的算法**(`mtp_varlen.hpp:150~267`): 把每路每位的 confidence 累乘成"前缀存活率", 全批所有 (路, 位) 排成一条队列按存活率降序;
从"只验真 token"(total_q = 路数)起, 每多纳入一位: 期望产出 += 存活率, 行数 += 1, 查"行数 → 每步毫秒"曲线(启动时实测, 滑动中位数在线修斜率)算吞吐, 取吞吐最大的行数;
再按同一条队列把多出来的行分给各路。两轮账: 定预算用 N−2 轮的置信(异步调度时 N−1 轮的还没回), 分行用 N−1 轮的。

**ds4 合批路现状**(`core_v41_req.c:210~224`): 每路各自 `v41_draft_pick_k`(按本请求成本), 行数超 8 从 k 最大的路削; 09-30 金融提示 N=3 投机总 41.8 < 纯合批 65.7, 于是 `allow_spec = n <= 2`。
当时的账: 合批里骨架已摊掉, 每多一行只剩专家字节 + 注意力 ~7 ms, 一行验证按接受直方图只值 0.43 个 token, 而给别的请求一行值 1 个。

**为什么 DeepInfer 的办法能改善**: 它不是"每路各选 k 再削", 是"全批只验最有把握的那几位" —— N=3 时总预算可能只有 4~5 行(三路各 1 + 最有把握的 1~2 位), 验证批 5 行 ≈ 纯合批 3 行 + 2 行专家字节,
产出 3 + 2×0.75 = 4.5 对 3。ds4 自己的账(每多一行 ~7 ms, 48 ms 一步)也支持: 一位存活率 ≥ 0.15 就值得(7/48)。
**做法**: `v41_multi_round` 里把各路 `host_conf` 的累乘(乘各路自己的 ρ)合成一条队列; 成本曲线复用每路 `v41_sched` 的最小二乘(合批时按总行数记 `v41_sched_cost`);
预算 = argmax 产出/成本; 分行按队列。`allow_spec` 的 2 路限制删掉, 由预算自己退化(没把握就是 total_q = 路数 = 纯合批)。
**门**: `multi_probe_spark.sh "3" 64 …` 真实 CFO 请求(`req_render_ids.sh`)三路投机 vs 纯合批, 读总 t/s; 三路答案与单路逐字节同(k 只影响快慢, 贪心下必须同)。
**不这么做会怎样**: 三路场景永远走纯合批, 每路 20.8(12k)。

### 3.4 合批路的发/等拆分

`v41_sched_run` 每步: `ds4_v41_multi_round`(发图 + 等 + 取 token)→ 逐路 `v41_emit`(渲染/写 socket/监控)→ 下一步。emit 期间 GPU 空着。
10-01 三路一步 48 ms 的独占表里"空转 3.2 ms"(memory `conc3_byte_account_1001`)大半是这段。
**做法**: 与单路 `v41_graph_launch/wait` 同构: multi_round 拆成 launch(写槽 + 发图)/ wait(等 + 取 want); 主循环先 launch 下一步再 emit 上一步。草稿已经是 launch/wait 两半(10-07)。
**门**: 三路 md5 逐字节同 + `multi_probe` 中位 ms。预计 48 → 45~46 ms。

### 执行顺序

3.1(改默认 + 门, 半天)→ 3.4(小, 无风险)→ 3.3(调度器)→ 3.2(要用户先裁)。每一步按 10 分钟验证铁律先跑短门, 长门只在提交前。
**结果(10-20 晚)**: 3.1 落地; 3.2 落地(默认 140, 用户定); 3.3 / 3.4 复核后不做(见下)。
**10-20 晚复核: 3.3 与 3.4 不做**。3.3 漏算了三路串行草稿 22 ms/轮(两路并道 10-01 判负), 最优预算也只有 ~57 t/s < 纯合批 62.5; 3.4 的"空转 3.2 ms"来自没有 emit 的 CLI 探针, 是 sync→pick→预取→cudaGraphLaunch 的主机串行, 拆 launch/wait 盖不住, 服务端能盖的只有 ~1 ms。账在 fable5 10-20 晚节。

## 4. 不排进本稿但要记下的

- **结构化输出**: DeepInfer 用 xgrammar 把 JSON schema / regex / EBNF 编成自动机, 每步给采样核一个 vocab 位掩码(草稿位也逐位掩, 拒绝时 rollback)。
  ds4 的 OpenAI 兼容层没有 `response_format`。纯 C 实现一个 JSON-schema 自动机 + 位掩码进采样核是独立的一期功能(估 2~3 千行), 不是速度优化; 助理工具调用走 DSML 不需要它。
- **分词并行**: 官方按 special token 切段并行编码。ds4 1M 提示的分词耗时没量过; 先量(`./ds4 --dump-tokens` 1M 文本计时), 超过预填的 1% 再做。
- **DeepInfer 的参考分数**(`eval/README.md`): AIME 2026 99.9% / GPQA Diamond 93.4% / HLE 42.6%; `tests/models/offline_decode_ppl.json` 是它自己的回归 PPL(短/长各 10 条)。
  可作 ds4 量化质量门的外部锚, 但它是温 1.0 采样 32 次的 Pass@1, 与 ds4 的参考前向尺不是一个口径, 只作旁证。
- DeepInfer 对 FlashMLA 的说明: 开源版比内部少一处 SASS 手改, 解码核慢 5~10%, 端到端最多差 5%。与 ds4 无关, 记一笔。
