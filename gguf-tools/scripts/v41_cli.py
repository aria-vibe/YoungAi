"""v41_cli — v41_teacher.py 的命令行定义(2026-09-12 从主文件拆出, 主文件超 500 行仓规)。
只有参数声明, 零逻辑。每个开关的"为什么"写在 help 里, 主文件的用法段不再重复。"""
import argparse


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf", help="模型目录: FP 的 HF 目录, 或 v41_quantize 产出的量化目录(学生从文件读回)")
    ap.add_argument("--ids", default="", help="一条序列(一行一个 id); 与 --seqs 二选一")
    ap.add_argument("--seqs", default="",
                    help="★多序列 top-K 档(2026-10-22, 后训练 ③ 的 FP 教师)★: 一行一条序列(空格隔开的 id, ds4 --ptrain dumpids= 的产物), "
                         "按 batch 维一次前向(各条独立从 0 起, 右侧补齐), 每条每位只落 top-K + 余量 → --out(格式见 v41_teacher_seqs.py)。"
                         "为什么不拼成一条: 教师看到的上下文必须与学生训练时一样(一题一条); 为什么不一条一趟: 流式装一遍原件 ≈ 25 分钟")
    ap.add_argument("--topk", type=int, default=64, help="配 --seqs: 每位保留的 top-K(与 ds4 --ptrain 的 topk= 必须相同)")
    ap.add_argument("--ntok", type=int, default=32)
    ap.add_argument("--layers", type=int, default=0, help="只跑前 N 层(0=全部)")
    ap.add_argument("--out", default="")
    ap.add_argument("--qnbit", type=int, default=0,
                    help="主干专家量化位宽(0=不量化, 走 FP 教师); 落地 bpw = nbit + 8/blk")
    ap.add_argument("--qblk", type=int, default=32, help="每几个元素一个 ue8m0 scale")
    ap.add_argument("--dump-act", default="",
                    help="把每层 MoE 输入的 E[x^2] 存成 f32[nlayer][dim](金融语料校准用)")
    ap.add_argument("--dump-mainh", default="",
                    help="★DSpark 夹具★: 把官方 Transformer.forward 返回的 main_hidden(目标层注意力输入的 hc 均值拼接, "
                         "f32[S][目标层数×dim])落盘 —— 给 v41_dspark_fixture.py --fp-mainh 用, 量'FP 隐态 vs 量化隐态'对草稿器的影响")
    ap.add_argument("--dump-mainh-variants", default="",
                    help="★DSpark 夹具·取法验证★: 目标层的四种候选 main_hidden 各落一份 <prefix>.{inmean,inpre,outmean,outpre}.bin "
                         "(层输入 hc 均值 = 官方 model.py 的取法 / 层输入按 pre_mix 折叠 = 注意力真正吃的那份 / 层输出的两种)。"
                         "用途: 官方参考实现的取法在 PPL 2 的文本上也只让草稿器中 0.6, 怀疑 h.mean(dim=2) 不是训练时的口径, 逐个喂夹具判")
    ap.add_argument("--vq-nc", type=int, default=0,
                    help="VQ 码字数(0=不走 VQ); 配 --vq-dim, bpw = ceil(log2 nc)/dim + 行增益")
    ap.add_argument("--vq-dim", type=int, default=8)
    ap.add_argument("--vq-iters", type=int, default=8)
    ap.add_argument("--vq-stride", type=int, default=8, help="训练码本的采样步长(最终 assign 仍全量)")
    ap.add_argument("--dump-moe", default="",
                    help="dump 每层 MoE 的输入 x 与输出 y(f32) 到目录: 解放大器的原料")
    ap.add_argument("--force-x", default="",
                    help="每层 MoE 的输入强制替换成该目录里教师的 x(层内解算, 隔断上游误差传播)")
    ap.add_argument("--amp-online", default="",
                    help="★在线序贯反修★: 一次前向里逐层解出放大器并立刻应用, 放大器存到该目录。"
                         "靶当场重算(关量化开关再跑一遍 ffn), 不需要预先 dump 的教师 y")
    ap.add_argument("--amp-k", type=int, default=128)
    ap.add_argument("--amp-lam", type=float, default=10.0,
                    help="★ridge 强度★: held-out K×λ 扫描的峰值是 10~30。曾硬编码 1e-3"
                         "(近乎无正则) ⇒ 泛化 −66.5%, 端到端 Σmin 0.6711→0.2431")
    ap.add_argument("--amp-scan-split", type=int, default=6144,
                    help="K 扫描的拟合/评估切分(无 <ids>.layout 时的前缀切): 前 N 个 token 解, 其余只评估(必须 > 5120)")
    ap.add_argument("--amp-scan-k", action="store_true",
                    help="★K 扫描诊断★: 每层对一串候选秩各算一次残差, 给 K↔质量曲线"
                         "(末位=完整最小二乘解=低秩这条路的天花板); 只诊断不应用修正")
    ap.add_argument("--amp", default="",
                    help="放大器目录(v41_amp_solve 产物): 每层 MoE 输出加 B·(A·x)")
    ap.add_argument("--amp-layers", default="",
                    help="配 --amp: 只挂这些层的放大器, 如 '0' / '0-9' / '0,5,7-9'(消融用); 空=全挂")
    ap.add_argument("--amp-select", default="",
                    help="★在线序贯反修·逐层择优★: 每层先 K×λ 分层 held-out 扫描, 按规则选 (λ,K) 再解、"
                         "再应用; 放大器 + manifest 存到该目录。held-out 按 <ids>.layout 的窗分层")
    ap.add_argument("--amp-rowdiag", default="",
                    help="★逐行诊断★ 形如 K:λ(如 256:3): 每层按 (λ,K) 在拟合行解一次, 报能量增益 vs 逐行增益、"
                         "顶 1% 行能量份额 —— 查'层内能量正 / 端到端负'的分叉在哪。不应用修正")
    ap.add_argument("--amp-t2diag", default="",
                    help="★靶口径诊断★ 形如 K:λ:<教师dump目录>: 每层解同 x 放大器, 对照教师 --dump-moe 的"
                         " y_fp(x_fp), 报 T2/T1 能量比、cos(T1,T2)、修掉 T1 后总误差的涨跌。不应用修正")
    ap.add_argument("--exact-weights", action="store_true",
                    help="★对拍夹具★: fp4/fp8/VQ 解出的权重留 f32 不舍 bf16(官方 fp4/fp8 kernel 就是用精确权重算的;"
                         " 舍 bf16 是本夹具为了喂官方 bf16 模型声明做的近似)。引擎用的是文件精确值, 对拍用这个口径")
    ap.add_argument("--index-topk", type=int, default=0,
                    help="★诊断夹具★: 覆盖 config 的 index_topk(每个 query 可见的压缩位数, 1017 是 512)。给大于序列长的值 = 索引器不做选择、"
                         "全部压缩位都看 —— 10-25 查教师 500 位之后回看失灵(抄 1500 位的第二份错块、工具 schema 段 NLL 5 对引擎 1.3)时, 用它分辨是索引选择坏还是别处坏。不是生产配置")
    ap.add_argument("--dump-idx", default="",
                    help="★诊断夹具(10-25)★: 把每个索引源层(L2/8/14/20/24/28/32/36)索引器选出的压缩位落到该目录 idx_Lnn.bin(int32 [s][topk], 未加 offset, "
                         "不可见位 -1)。用途: FP 路(出厂权重)与 file 路(量化目录)同一序列各落一份, 逐查询比 top-512 集合的重叠 —— 教师 600 位后回看失灵, 先定是不是选错了位")
    ap.add_argument("--verify-load", action="store_true",
                    help="★诊断夹具(10-25)★: 每个非 routed 专家的张量装到 GPU 后, 再从 mmap 用 CPU 独立解一遍逐位比, 不同就打出来。"
                         "为什么: GB10 内存吃紧时 NVRM 报 _memdescAllocInternal OOM, 10-22 引擎撞过'GPU 读主机页静默读坏'; 教师 FP 路 + engram 开是整机最紧的组合, "
                         "症状(确定性复现、关 engram 就好、只有长程选择坏)像装载时某些张量被读坏。抓到哪个张量不同, 病根就在装载而不在算法")
    ap.add_argument("--ffn-from", default="",
                    help="★诊断夹具(10-25)★: layers.N.ffn.* (routed/shared 专家 + gate)改从这个目录装, 其余(注意力/索引器/压缩器/engram/头)仍从主目录装 —— "
                         "FP 路(精确权重)长程失灵而量化目录不失灵, 用它做 2×2 二分: 量化目录 + --ffn-from FP = q4_K 骨架 + 精确专家; FP + --ffn-from 量化目录 = 精确骨架 + VQ 专家。"
                         "与 --fp-dir 不同: --fp-dir 只在反修算靶那一遍(qmode 关)才接管 ffn, 平常前向不生效")
    ap.add_argument("--act-quant", action="store_true",
                    help="★诊断夹具(10-25)★: 量化权重(fp4/fp8/VQ/q4_K)的 linear 先按官方 linear() 口径把激活压 fp8(per-32 ue8m0, 量化再反量化)再算。"
                         "教师默认不压(fp32 精确激活); 精确 FP4 专家下长程失灵而 VQ 专家不失灵, 剩下的唯一差别就是这步 —— 模型可能是靠它校准的")
    ap.add_argument("--kernel-check", action="store_true",
                    help="★诊断夹具(10-25)★: 保留的三个官方 tilelang 核(act_quant 原地 fp8 / fp4_act_quant 原地 fp4 / hc_split_sinkhorn)每次调用都在旁边用 torch 参考"
                         "(v41_kernel_rowprobe.py 同一套公式)再算一遍, 记最大差、不同元素数、输入幅度; 前向仍用核的结果。rowprobe 只在随机小幅输入上对过, "
                         "这里对的是真实激活(巨值通道)。有差 ⇒ 核在 GB10 上对某些幅度算错; 无差 ⇒ 核排除")
    ap.add_argument("--ffn-part", default="", choices=["", "routed", "shared"],
                    help="配 --ffn-from: 只换 routed 专家(layers.N.ffn.experts.*, FP 侧是 FP4)或只换 shared 专家(layers.N.ffn.shared_experts.*, FP 侧是 FP8, 量化目录里是 Q4_K); 空 = 整个 ffn")
    ap.add_argument("--balloon-gb", type=float, default=0.0,
                    help="★诊断夹具(10-25)★: 建模后先在 GPU 上占住这么多 GB 不放(填 1), 人为把整机内存压到 FP 路的水位。给干净的 file 档加它: "
                         "坏了 ⇒ 病是内存压力下的静默读坏(NVRM _memdescAllocInternal OOM 那条路), 不是权重也不是算法")
    ap.add_argument("--force-idx", default="",
                    help="★诊断夹具(10-25)★: 索引器不自己选, 直接用该目录里 --dump-idx 落的 idx_Lnn.bin(同一序列、另一条路如 file 档跑出来的选择)。"
                         "FP 路换上 file 路的选择后能抄 ⇒ 病在索引选择; 还不能 ⇒ 病在注意力值/分数。不是生产配置")
    ap.add_argument("--index-fp32", action="store_true",
                    help="★诊断夹具(10-25)★: 索引分数(q·k → relu → ×权重 → 按头求和)全程 fp32, 不按官方在 bf16 里算。官方/引擎都是 bf16 口径, 这个旗只用来分辨"
                         "'选错位'是不是 bf16 精度/并列造成的。不是生产配置")
    ap.add_argument("--no-engram", action="store_true",
                    help="★对拍夹具★: 摘掉 engram 层(L1/L14 的 n-gram 查表加法), 给引擎 P2 分段对拍用 —— 引擎先不带"
                         " engram 跑通其余链路, 两边同口径比 logits。不是生产配置")
    ap.add_argument("--amp-whiten", type=int, default=0,
                    help="★输出通道白化★ 0=原始(按能量截断) 1=按靶通道 RMS 白化 2=按 y_fp 通道 RMS 白化。"
                         "只改 SVD 截断保留哪些方向(满秩解不变): 防止秩全花在巨值通道上。"
                         "扫描/择优/解/逐行诊断统一吃这个开关")
    ap.add_argument("--fp-dir", default="",
                    help="★量化目录当学生时必给★: FP 原始 HF 目录。反修的靶 y_fp(x_q) 要在同一个 x 上"
                         "重跑 FP 的 ffn, 权重从这里装(量化目录里只有 VQ/FP4, 没有 FP)")
    ap.add_argument("--act", default="",
                    help="校准激活(--dump-act 的产物): 给 VQ 的最近邻加列权 E[x^2], 误差优先落在低能量通道")
    ap.add_argument("--qcb", default="",
                    help="码本文件(v41_codebook 产出); 不给则拒绝量化 —— 码本必须匹配源分布")
    ap.add_argument("--qpow2", type=int, default=1,
                    help="1=scale 取 2 的幂(真格式 ue8m0); 0=精确 f32 σ(仅诊断, 不可落地)")
    a = ap.parse_args()
    if bool(a.ids) == bool(a.seqs):
        ap.error("--ids 与 --seqs 必须给且只给一个")
    return a
