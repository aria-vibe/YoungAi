"""v41_patches — 官方 inference/model.py 在 GB10 教师端要打的三处补丁 + meta 建模后的 buffer 修复。
从 v41_teacher.py 拆出(2026-09-12, 主文件超 500 行仓规)。逐字搬家, 语义零改动。

① linear 走 dequant + fp32 matmul(不做激活量化) —— 教师与学生必须同一条前向路
② sparse_attn 换 torch 版 —— 官方 kernel 要 141312 B 动态 shared memory, GB10 上限 101376
③ engram 表按行流式取 —— 98 GB 不能整层驻留, 每 token 只查 24 行
④ meta 建模后修 buffer —— 非零常量(freqs_cis / ngram 哈希表)清零不报错只静默出错
"""
import numpy as np
import torch
import torch.nn.functional as F

from v41_hf_io import e8m0_to_f32


ACT_QUANT = False    # --act-quant(诊断, 10-25): 量化权重的 linear 先把激活按官方 linear() 的口径压 fp8(per-32 ue8m0)再算


def patch_linear(M):
    """权重在 materialize 时已 dequant 成 bf16, 这里只负责【用 fp32 算】。
    ★为什么不图省事用 bf16 GEMM★: 本仓实测过 fp16 层缓存让判决 KL 从 0.4742 劣化到
    0.4775(fable5 09-03), 教师端掺这种损失会把它算进"量化代价"里。官方 kernel 更低
    (激活压 fp8), 那是推理优化, 不该进判决。
    ★ACT_QUANT(10-25 诊断)★: 官方 linear() 对 fp4/fp8 权重一律先 act_quant(x, 32, "ue8m0") 再 GEMM —— 模型是带着这步训出来的;
    教师去掉它在 VQ/q4_K 权重上没事(file 档对得上引擎), 在精确 FP4 专家上长程失灵, 这个旗把它按原口径加回去(inplace 量化再反量化,
    fp8 × 2 的幂 scale 在 bf16 里无损, 与 kernel 里 fp8 激活 × scale 同值)。"""
    def linear_dq(x, weight, bias=None):
        assert bias is None
        if ACT_QUANT and getattr(weight, "_was_quant", False):
            xb = x.to(torch.bfloat16).contiguous()
            M.act_quant(xb, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
            x = xb
        y = F.linear(x.float(), weight.float())
        # 两个分支的返回 dtype 必须跟官方对齐: 官方对量化权重走 fp4/fp8_gemm, kernel 出 bf16
        # (下游 act_quant 就是按 bf16 收的); 对普通权重走 F.linear 保持输入 dtype -- Gate 正是
        # 靠后者拿 fp32 分数做 topk, 一刀切成 bf16 会改变专家选择。
        return y.to(torch.bfloat16) if getattr(weight, "_was_quant", False) else y.to(x.dtype)
    M.linear = linear_dq


def patch_sparse_attn(M):
    """★不是嫌它慢, 是 GB10 上物理跑不了★: 官方 kernel 要 141312 字节动态 shared memory
    (q_shared 64KB + o_shared 64KB, h=64/d=512 定死), 而 GB10 的 opt-in 上限是 101376
    (实测 shared_memory_per_block_optin)。那 kernel 是照数据中心 Blackwell(227KB)写的。
    逐条对应 kernel 语义, 一条都不能差:
      ① idx == -1 ⇒ 该位置 kv 取 0 且 score 取 -inf
      ② scores_max 初值 -1e30(有限下界) ⇒ 全无效的行得到全零输出而不是 NaN
      ③ ★attn_sink 只进分母, 不贡献 value★
    数值上本版全程 fp32(kernel 是 bf16 gemm + fp32 累加), 教师端取更准的一侧。"""
    def sparse_attn_torch(q, kv, attn_sink, topk_idxs, scale):
        b, m, h, d = q.shape
        idx = topk_idxs.long()
        valid = idx >= 0
        safe = idx.clamp_min(0)
        out = torch.empty(b, m, h, d, dtype=torch.float32, device=q.device)
        step = max(1, int(2e8 // (b * idx.shape[-1] * d * 4)))   # 控 gather 的峰值内存(按 b 摊: 多序列档 b=64 时每位 ~270 MB, 不摊就是 ×64; b=1 与原式同)
        for s0 in range(0, m, step):
            s1 = min(m, s0 + step)
            si, sv = safe[:, s0:s1], valid[:, s0:s1]
            k = kv.gather(1, si.reshape(b, -1, 1).expand(b, si.shape[1] * si.shape[2], d))
            k = k.view(b, s1 - s0, -1, d).float() * sv.unsqueeze(-1)
            sc = torch.einsum("bmhd,bmkd->bmhk", q[:, s0:s1].float(), k) * scale
            sc = sc.masked_fill(~sv.unsqueeze(2), float("-inf"))
            mx = sc.amax(-1, keepdim=True).clamp_min(-1e30)
            p = (sc - mx).exp()
            den = p.sum(-1) + (attn_sink.view(1, 1, h).float() - mx.squeeze(-1)).exp()
            out[:, s0:s1] = torch.einsum("bmhk,bmkd->bmhd", p, k) / den.unsqueeze(-1)
        return out.to(q.dtype)
    M.sparse_attn = sparse_attn_torch


class StreamEngramEmbed(torch.nn.Module):
    """engram 表换成按行 memmap 取(在 meta 上建好后替换, 它自带真实数据)。
    取到的行与 C 读器逐位对拍过(v41_dequant_parity.py 全绿)。"""
    def __init__(self, idx, dev, name_w, name_s, num_emb, dim, blk):
        super().__init__()
        pw, ow, _, _ = idx[name_w]
        ps, os_, _, _ = idx[name_s]
        self.w = np.memmap(pw, dtype=np.uint8, mode="r", offset=ow, shape=(num_emb, dim))
        self.s = np.memmap(ps, dtype=np.uint8, mode="r", offset=os_, shape=(num_emb, dim // blk))
        self.blk = blk
        self.dev = dev

    def forward(self, indices):
        flat = indices.reshape(-1).cpu().numpy()
        wc = torch.from_numpy(np.ascontiguousarray(self.w[flat]))
        sc_ = torch.from_numpy(np.ascontiguousarray(self.s[flat]))
        w = wc.to(self.dev)
        s = sc_.to(self.dev)
        v = w.view(torch.float8_e4m3fn).float().unflatten(-1, (-1, self.blk))
        v = (v * e8m0_to_f32(s).unsqueeze(-1)).flatten(-2)
        out = v.to(torch.bfloat16).view(*indices.shape, -1)
        if VERIFY_ROWS is not None:   # --verify-load 诊断: 同一批行在 CPU 上独立解一遍, 与 GPU 上那份逐位比
            vc = wc.view(torch.float8_e4m3fn).float().unflatten(-1, (-1, self.blk))
            vc = (vc * e8m0_to_f32(sc_).unsqueeze(-1)).flatten(-2).to(torch.bfloat16).view(*indices.shape, -1)
            nd = int((out.cpu() != vc).sum().item())
            VERIFY_ROWS[0] += 1
            if nd:
                VERIFY_ROWS[1] += 1
                if VERIFY_ROWS[1] <= 5:
                    print(f"[★verify★] engram 行 GPU≠CPU 元素 {nd}/{out.numel()}", flush=True)
        return out


def fix_buffers(M, net, args, tok, dev):
    """【为什么必须修】参数可以留在 meta(hook 按需装), 但 buffer 不行: 它们是**算出来的**,
    在 meta context 下建等于值当场丢失。分两类:
      ① KV cache 类(window_kv_cache/compress_kv_cache/kv_state/...): 本来就是 zeros, 重建即可
      ② ★非零常量★: Attention.freqs_cis(YaRN 频率, 还分层不同) 和 NgramHashState 的
         token_map/primes/multipliers —— 这类清零【不会报错, 只会静默出错】, 必须重算。
    返回 (修的 buffer 数, 其中重算的 freqs_cis 数)。"""
    M.precompute_freqs_cis.cache_clear()   # ★lru_cache 在 meta 下已缓存过 meta 结果, 不清会拿回 meta★
    nfix = nfreq = 0
    with torch.device(dev):
        for mod in net.modules():
            for bname, buf in list(mod.named_buffers(recurse=False)):
                if buf is None or buf.device.type != "meta":
                    continue
                if bname == "freqs_cis":
                    # 复原 model.py Attention.__init__ 的分支: 有压缩比才开 YaRN
                    if getattr(mod, "compress_ratio", 0):
                        osl, theta = args.original_seq_len, args.compress_rope_theta
                    else:
                        osl, theta = 0, args.rope_theta
                    nb = M.precompute_freqs_cis(mod.rope_head_dim, args.max_seq_len, osl, theta,
                                                args.rope_factor, args.beta_fast, args.beta_slow)
                    nfreq += 1
                else:
                    nb = torch.zeros(buf.shape, dtype=buf.dtype)
                mod.register_buffer(bname, nb, persistent=False)
                nfix += 1
    # NgramHashState 整个重建: 它全是算出来的 buffer(token_map 要遍历整个词表), 没有参数
    if net.engram_hash is not None:
        with torch.device(dev):
            net.engram_hash = M.NgramHashState(args, net.engram_layout, tok)
    return nfix, nfreq


IDX_DUMP = None      # --dump-idx 目录(Path), v41_teacher.py 设; None = 不落
IDX_FP32 = False     # --index-fp32: 索引分数用 fp32 算(诊断, 见 v41_cli.py)
IDX_FORCE = None     # --force-idx 目录(Path): 用别处落的 idx_Lnn.bin 代替自己的选择(诊断)
VERIFY_ROWS = None   # --verify-load: [取行次数, 不同次数], engram 流式行 GPU vs CPU 逐位比(诊断)
KCHECK = None        # --kernel-check: {站点: [调用数, 最大差, 不同元素数, 总元素数, 输入最大|x|]}, v41_teacher.py 前向后打印


# ---- --kernel-check 的 torch 参考(与 v41_kernel_rowprobe.py 同一套公式, 照 kernel.py 逐行写) ----
def _pow2_ceil(a):
    return torch.exp2(torch.ceil(torch.log2(a)))


def _e2m1_rne(v):
    """e2m1 的 RNE: 网格 0 .5 1 1.5 2 3 4 6, 平手取尾数位为 0 的那个(0 1 2 4 6)。"""
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=v.device)
    even = torch.tensor([1, 0, 1, 0, 1, 0, 1, 1], device=v.device, dtype=torch.bool)
    a = v.abs().clamp(max=6).unsqueeze(-1)
    d = (a - grid).abs()
    dmin = d.amin(-1, keepdim=True)
    cand = (d == dmin)
    tie = cand.sum(-1, keepdim=True) > 1
    pick = torch.where(tie, cand & even, cand).float().argmax(-1)
    return grid[pick] * v.sign()


def ref_fp8_inplace(x, blk):
    xf = x.float().unflatten(-1, (-1, blk))
    amax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    s = _pow2_ceil(amax / 448.0)                      # ue8m0: 2 的幂, 向上取
    q = (xf / s).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (q * s).flatten(-2).to(x.dtype)


def ref_fp4_inplace(x, blk, e4m3_scale):
    xf = x.float().unflatten(-1, (-1, blk))
    amax = xf.abs().amax(-1, keepdim=True)
    if e4m3_scale:
        amax = amax.clamp(min=6 * 2 ** -9)
        s = (amax / 6).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp(min=6 * 2 ** -126)
        s = _pow2_ceil(amax / 6)
    return (_e2m1_rne(xf / s) * s).flatten(-2).to(x.dtype)


def ref_hc_split_sinkhorn(mixes, sc, base, hc, iters, eps):
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * sc[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * sc[1] + base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * sc[2] + base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def _kc_note(site, kern, ref, amax_in):
    d = (kern.float() - ref.float()).abs()
    st = KCHECK.setdefault(site, [0, 0.0, 0, 0, 0.0])
    st[0] += 1
    st[1] = max(st[1], d.max().item())
    st[2] += int((d > 0).sum().item())
    st[3] += d.numel()
    st[4] = max(st[4], amax_in)


def patch_kernel_check(M):
    """--kernel-check: 三个保留的 tilelang 核每次调用都旁跑 torch 参考, 记差; 前向仍用核的结果(只观测不改路)。
    为什么: rowprobe 只在 randn×0.7~2 的输入上对过, 真实激活有巨值通道(千级), e4m3 scale 在 amax/6>448 会饱和、
    fast_log2_ceil 走位运算 —— 这些只有真数据才试得出来。"""
    k_act, k_fp4, k_hc = M.act_quant, M.fp4_act_quant, M.hc_split_sinkhorn

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        if not (inplace and scale_fmt):
            return k_act(x, block_size, scale_fmt, scale_dtype, inplace)
        amax = x.float().abs().max().item()
        ref = ref_fp8_inplace(x.clone(), block_size)
        out = k_act(x, block_size, scale_fmt, scale_dtype, True)
        _kc_note(f"act_quant fp8 blk{block_size} N={x.shape[-1]}", out, ref, amax)
        return out

    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        if not inplace:
            return k_fp4(x, block_size, inplace, scale_dtype)
        amax = x.float().abs().max().item()
        e4 = scale_dtype == torch.float8_e4m3fn
        ref = ref_fp4_inplace(x.clone(), block_size, e4)
        out = k_fp4(x, block_size, True, scale_dtype)
        _kc_note(f"fp4_act_quant blk{block_size} {'e4m3' if e4 else 'e8m0'} N={x.shape[-1]}", out, ref, amax)
        return out

    def hc_split_sinkhorn(mixes, sc, base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        amax = mixes.float().abs().max().item()
        pre, post, comb = k_hc(mixes, sc, base, hc_mult, sinkhorn_iters, eps)
        rp, ro, rc = ref_hc_split_sinkhorn(mixes, sc, base, hc_mult, sinkhorn_iters, eps)
        _kc_note("hc_split_sinkhorn pre", pre, rp, amax)
        _kc_note("hc_split_sinkhorn post", post, ro, amax)
        _kc_note("hc_split_sinkhorn comb", comb, rc, amax)
        return pre, post, comb

    M.act_quant, M.fp4_act_quant, M.hc_split_sinkhorn = act_quant, fp4_act_quant, hc_split_sinkhorn


def patch_indexer(M, block=1024):
    """⑤ 索引分数按 query 块算(2026-09-22)。官方 Indexer.forward 一次性物化 [b, s, h, t] 的 bf16 分数:
    24400 token 的 ratio-1 层 = 24400×32×24400×2 B = 38 GB, 加 relu/乘权重的临时副本翻倍 —— 09-22 实撞
    L20 一到就把 121 GB 吃干, 看门狗停车(8192 token 时只有 4 GB, 所以以前没撞)。
    这里逐块做 einsum → relu → ×权重 → 按头求和, 只留 [b, s, t](24400² × 2 B = 1.2 GB); 每个元素的算式与
    官方逐字相同(元素之间互不依赖, 分块不改值), 后面的 mask / 候选块 / topk 原样照抄官方。"""
    def forward(self, x, qr, latent, start_pos, offset):
        assert self.freqs_cis is not None
        bsz, seqlen, _ = x.size()
        ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen
        if self.owns_k and latent is not None:
            freqs = (self.freqs_cis[: seqlen - seqlen % ratio : ratio] if start_pos == 0
                     else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0))
            k = self.k_norm(self.wk(latent))
            M.apply_rotary_emb(k[..., -rd:], freqs)
            M.fp4_act_quant(k, M.fp4_block_size, True)
            self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
            M.shared_attn.index_k = self.k_cache
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
        M.apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
        M.fp4_act_quant(q, M.fp4_block_size, True)
        index_k = M.shared_attn.index_k[:bsz, : end_pos // ratio]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        sdt = torch.float32 if IDX_FP32 else q.dtype   # 诊断: fp32 分数(官方/引擎都是 bf16 口径)
        index_score = torch.empty(bsz, seqlen, index_k.size(1), dtype=sdt, device=x.device)
        for s0 in range(0, seqlen, block):
            s1 = min(seqlen, s0 + block)
            sc = torch.einsum("bshd,btd->bsht", q[:, s0:s1].to(sdt), index_k.to(sdt))
            index_score[:, s0:s1] = (sc.relu_() * weights[:, s0:s1].unsqueeze(-1).to(sdt)).sum(dim=2)
            del sc
        if M.world_size > 1:
            M.dist.all_reduce(index_score)
        if start_pos == 0:
            compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
            index_score.masked_fill_(torch.arange(seqlen // ratio, device=x.device) >= compress_lens, -torch.inf)
        else:
            compress_lens = end_pos // ratio
        if self.is_candidate_source:
            M.shared_attn.candidates = M.select_candidate_blocks(
                index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size)
        elif self.uses_candidates:
            index_score = index_score.masked_fill(~M.shared_attn.candidates, -torch.inf)
        topk = min(self.index_topk, end_pos // ratio)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        li = getattr(self, "layer_id", -1)
        if IDX_DUMP is not None:   # 诊断: 选中的压缩位(未加 offset, 不可见 -1), 一个索引源层一个文件
            torch.where(idxs < compress_lens, idxs, -1).int().cpu().numpy().tofile(IDX_DUMP / f"idx_L{li:02d}.bin")
        if IDX_FORCE is not None:  # 诊断: 换上别处落的选择(形状必须同: 同序列同配置)
            f = np.fromfile(IDX_FORCE / f"idx_L{li:02d}.bin", dtype=np.int32).reshape(bsz, seqlen, -1)
            f = torch.from_numpy(f).to(x.device)
            assert f.shape == idxs.shape, f"force-idx L{li} 形状 {tuple(f.shape)} != {tuple(idxs.shape)}"
            return torch.where(f >= 0, f + offset, -1).int()
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()
    M.Indexer.forward = forward
