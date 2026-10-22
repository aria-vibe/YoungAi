#!/usr/bin/env python3
"""v41_hf_io.py — V4.1 HF 权重的索引/读取/解量化, 供教师前向复用(2026-09-11)。

【定位】金标/教师夹具的 IO 层, 不是数值链的一部分(全仓零 Python 数值链: 算法一律 C)。
这里只做三件事: 扫 safetensors 头建索引、按索引 memmap 零复制取张量、按 V4.1 的格式
解量化(E4M3 + ue8m0 块 scale / FP4 E2M1 两两打包)。

【为什么从 v41_teacher.py 拆出来】单文件 ≤500 行是仓规。拆点选在这里是因为 IO 层与
前向编排层本来就无耦合 —— 它只认"张量名 → 字节", 不认模型结构。

【正确性】本文件的解量化与 C 读器(gguf-tools/quantize/st_locate.h)逐位对拍过
(v41_dequant_parity.py, 四种块格式全绿)。★改这里必须重跑那个对拍★ —— V4.1 的格式
解错不会报错, 只会静默出一整套好看的假数。
"""
import json
import mmap
import os
import struct
import threading
import time
from pathlib import Path

import numpy as np
import torch

FP4_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


# ---------------- safetensors 索引 ----------------
def build_index(hf: Path):
    """扫一遍全部分片, 建 张量名 → (文件, 数据绝对偏移, dtype, shape)。
    只扫一次(96085 个张量), 之后所有加载都查这张表。
    ★有 model.safetensors.index.json 就以它的 weight_map 为准★: 量化目录(v41_quantize 产物)把
    engram 所在的出厂分片软链进来, 同目录下同名张量只认 map 指定的那个文件 —— 靠 glob 顺序
    "后扫的覆盖先扫的"是在赌文件名排序, 赌错不报错只出假数。"""
    idx = {}
    files = sorted(hf.glob("*.safetensors"))
    wm = None
    ij = hf / "model.safetensors.index.json"
    if ij.exists():
        wm = json.loads(ij.read_text())["weight_map"]
        files = sorted({hf / f for f in wm.values()})
    for p in files:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        base = 8 + n
        for name, e in hdr.items():
            if name == "__metadata__":
                continue
            if wm is not None and wm.get(name) != p.name:
                continue
            idx[name] = (p, base + e["data_offsets"][0], e["dtype"], tuple(e["shape"]))
    return idx


DT = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.uint8, "I8": torch.uint8, "U8": torch.uint8,
      "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "Q4_K": torch.uint8}

# 每个逻辑元素占几字节。Q4_K 是块格式(144 B / 256 元素), shape 记的是【逻辑形状】——
# 字节数必须按块算, 按 shape 算会短 4.5 倍(读出来是能跑的垃圾, 不报错)。
ELEM_BYTES = {"BF16": 2, "F16": 2, "F32": 4}


_MM = {}


def _mm(path):
    """每个分片一份 memmap, 复用。★不要每次 open+read★: 那样每个张量要多两次内存拷贝
    (read 出 bytes, 再 bytearray 复制一份), 一层 1152 个专家张量 × 40 层 = 46080 次,
    实撞的表现是 GPU 利用率 0% 而前向迟迟不动。"""
    if path not in _MM:
        _MM[path] = np.memmap(path, dtype=np.uint8, mode="r")
    return _MM[path]


_VIEW = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn}


def tensor_nbytes(dt, shp):
    """一个张量在文件里占的字节数。Q4_K 按块算(见 ELEM_BYTES 的注释), 其余 = 元素数 × 元素字节。"""
    if dt == "Q4_K":
        rows, cols = shp
        return rows * (cols // 256) * 144
    return int(np.prod(shp)) * ELEM_BYTES.get(dt, 1)


# ---------------- 整层预读(2026-10-22) ----------------
# 为什么: 逐张量 mmap 切片是缺页驱动的小块读 —— 一层专家 2304 个张量 7.64 GB 实测 22.6 s(0.34 GB/s), 占教师一层 48 s
#   的近半, 而盘本身 O_DIRECT 顺序读 8.4 GB/s、GPU 只有四成在干活。层 pre-hook 时把【下一层】全部张量所在的字节段在后台
#   线程一次顺序读进内存, 本层到时 load_raw 直接从内存切: 字节一个不变 ⇒ 数值与 mmap 路逐位同(字节级对拍过才换上)。
# 段怎么切: 本层张量按文件偏移排序, 相邻间隙 ≤ _PF_GAP 的并成一段连续读(间隙里夹的别层/别的张量白读一点, 比上千次小读便宜);
#   engram 表(98 GB, 名字 layers.N.engram.embed.*)不在内, 它按行流式(StreamEngramEmbed)。
# 内存: 最多两层在内存(当前层 + 预读中的下一层) ≈ 2 × 7.7 GB; 层 post-hook 丢掉本层。
# 出错会怎样: O_DIRECT 要求偏移/长度/缓冲都 4 KB 对齐 —— 段两端向外对齐, 缓冲用匿名 mmap(页对齐), 多读的边角不用;
#   文件尾不足一块 preadv 返回短读, 读到 0 字节就停。某层没预读到(比如别的索引 idx_fp 里的张量)就走原来的 mmap 路, 只慢不错。
_PF = {}                 # 层号 → _Prefetch
_PF_GAP = 16 << 20
_PF_ALIGN = 4096
_PF_CHUNK = 16 << 20


class _Prefetch:
    def __init__(self, idx, li):
        pre, skip = f"layers.{li}.", f"layers.{li}.engram.embed."
        spans = {}
        for name, (p, off, dt, shp) in idx.items():
            if name.startswith(pre) and not name.startswith(skip):
                spans.setdefault(p, []).append((off, off + tensor_nbytes(dt, shp)))
        self.runs = []       # [path, lo, hi, alo, buf]: 张量覆盖 [lo,hi), 实际读 [alo, 对齐(hi)), buf 从 alo 起
        for p, sp in spans.items():
            sp.sort()
            lo, hi = sp[0]
            for l2, h2 in sp[1:]:
                if l2 - hi > _PF_GAP:
                    self.runs.append([p, lo, hi, 0, None])
                    lo, hi = l2, h2
                else:
                    hi = max(hi, h2)
            self.runs.append([p, lo, hi, 0, None])
        self.nbytes = sum(r[2] - r[1] for r in self.runs)
        self.ready = False
        self.t0 = time.time()
        self.secs = 0.0
        self.th = threading.Thread(target=self._read, daemon=True)
        self.th.start()

    def _read(self):
        for run in self.runs:
            p, lo, hi = run[0], run[1], run[2]
            alo = lo & ~(_PF_ALIGN - 1)
            ahi = (hi + _PF_ALIGN - 1) & ~(_PF_ALIGN - 1)
            buf = mmap.mmap(-1, ahi - alo)
            mv = memoryview(buf)
            fd = os.open(p, os.O_RDONLY | os.O_DIRECT)
            try:
                pos = 0
                while pos < ahi - alo:
                    n = os.preadv(fd, [mv[pos:pos + _PF_CHUNK]], alo + pos)
                    if n <= 0:
                        break
                    pos += n
            finally:
                os.close(fd)
            run[3], run[4] = alo, buf
        self.secs = time.time() - self.t0

    def slice(self, p, off, n):
        for path, lo, hi, alo, buf in self.runs:
            if path == p and lo <= off and off + n <= hi:
                return np.frombuffer(buf, dtype=np.uint8, count=n, offset=off - alo)
        return None


def prefetch_start(idx, li, nlayers):
    """发第 li 层的后台预读(层号越界或已发过就不动)。"""
    if 0 <= li < nlayers and li not in _PF:
        _PF[li] = _Prefetch(idx, li)


def prefetch_ready(li):
    """等第 li 层的字节到内存; 返回 (等了几秒, 段字节数, 读了几秒)。没发过预读返回 None —— 这层照旧走 mmap。"""
    pf = _PF.get(li)
    if pf is None:
        return None
    t = time.time()
    pf.th.join()
    pf.ready = True
    return time.time() - t, pf.nbytes, pf.secs


def prefetch_drop(li):
    _PF.pop(li, None)        # 匿名映射随对象释放解除; 切出去的 numpy 视图都已拷到设备上, 没人再引用


def _pf_slice(p, off, n):
    for pf in _PF.values():
        if pf.ready:
            r = pf.slice(p, off, n)
            if r is not None:
                return r
    return None


def load_raw(idx, name, device):
    """按索引把一个张量读上来(不解量化)。E8M0/I8/U8 一律当 uint8 收 —— 解释权在用它的地方。
    Q4_K 按块字节读成一维 uint8(shape 是逻辑形状, 塞不进去), 解释权在 load_q4k。
    字节来源: 这层预读到内存了就从内存切(_pf_slice), 否则 memmap —— 两条路给的是同一段文件字节。"""
    p, off, dt, shp = idx[name]
    nbytes = tensor_nbytes(dt, shp)
    raw = _pf_slice(p, off, nbytes)
    if raw is None:
        raw = _mm(p)[off:off + nbytes]
    if dt == "Q4_K":
        return torch.from_numpy(raw).to(device, non_blocking=True)
    t = torch.from_numpy(raw)
    if dt in _VIEW:
        t = t.view(_VIEW[dt])
    return t.view(*shp).to(device, non_blocking=True)


def is_q4k(idx, name):
    return name in idx and idx[name][2] == "Q4_K"


def load_q4k(idx, name, device):
    """q4_K 块流 → f32 [rows, cols]。数值全在 C/CUDA(libv41vq.so 的 v41_q4k_decode_gpu),
    这里只递指针 —— 那个核与 src/common/ds4_deq_q4_K(金标)逐式同源。"""
    rows, cols = idx[name][3]
    blocks = load_raw(idx, name, device).contiguous()
    out = torch.empty(rows, cols, dtype=torch.float32, device=device)
    rc = vq_lib().v41_q4k_decode_gpu(blocks.data_ptr(), rows, cols, out.data_ptr())
    if rc != 0:
        raise RuntimeError(f"q4_K 解码失败 rc={rc} @{name}")
    return out


# ---------------- VQ 落盘产物(v41_quantize) ----------------
_VQLIB = None


def vq_lib():
    """libv41vq.so: 与量化器链的是同一个 v41_vq.o, 解码核只有一份(数值在 C, 这里只递指针)。"""
    global _VQLIB
    if _VQLIB is None:
        import ctypes
        so = Path(__file__).resolve().parent.parent / "quantize" / "libv41vq.so"
        _VQLIB = ctypes.CDLL(str(so))
        _VQLIB.v41_vq_decode_gpu.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        _VQLIB.v41_vq_decode_gpu.restype = ctypes.c_int
        _VQLIB.v41_q4k_decode_gpu.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        _VQLIB.v41_q4k_decode_gpu.restype = ctypes.c_int
    return _VQLIB


def has_vq(idx, wname):
    return wname.endswith(".weight") and (wname[:-7] + ".vq.idx") in idx


def load_vq(idx, wname, device):
    """layers.N.ffn.experts.M.w1.weight → 用 .vq.idx / .vq.gain + 专家共享 .vq.cb 解成 f32 [rows, cols]。
    索引流逐行字节对齐 LSB 先, 位宽 = ceil(log2 nc); 列数从 (每行字节数, 位宽, dim) 反推。"""
    base = wname[:-7]
    ex = base.rsplit(".", 1)[0]
    pk = load_raw(idx, base + ".vq.idx", device).contiguous()
    gain = load_raw(idx, base + ".vq.gain", device).contiguous()
    cb = load_raw(idx, ex + ".vq.cb", device).contiguous()
    nc, dim = cb.shape
    bits = max(1, int(np.ceil(np.log2(nc))))
    rows, bytes_row = pk.shape
    cols = (bytes_row * 8 // bits) * dim
    out = torch.empty(rows, cols, dtype=torch.float32, device=device)
    rc = vq_lib().v41_vq_decode_gpu(pk.data_ptr(), cb.data_ptr(), gain.data_ptr(),
                                    rows, cols, dim, nc, out.data_ptr())
    if rc != 0:
        raise RuntimeError(f"VQ 解码失败 rc={rc} @{wname}")
    return out


# ---------------- dequant(与 C 读器逐位对拍过) ----------------
def e8m0_to_f32(b: torch.Tensor) -> torch.Tensor:
    v = torch.exp2(b.float() - 127.0)
    return torch.where(b == 255, torch.zeros_like(v), v)   # 255 = NaN 槽, 当 0 处理


def dq_fp8(w: torch.Tensor, s_u8: torch.Tensor) -> torch.Tensor:
    """E4M3 + ue8m0 块 scale。块高/宽由形状反推(专家与 engram 的块高是 1, 其余 32)。"""
    rows, cols = w.shape
    s = e8m0_to_f32(s_u8)
    bh, bw = rows // s.shape[0], cols // s.shape[1]
    s = s.repeat_interleave(bh, 0).repeat_interleave(bw, 1)
    return w.float() * s


def dq_fp4(w_u8: torch.Tensor, s_u8: torch.Tensor, tbl: torch.Tensor) -> torch.Tensor:
    """FP4 E2M1: [out, in/2] uint8, 低 nibble 偶数列 / 高 nibble 奇数列(官方 convert.py 口径)。"""
    lo = tbl[(w_u8 & 0x0F).long()]
    hi = tbl[((w_u8 >> 4) & 0x0F).long()]
    x = torch.stack([lo, hi], dim=-1).flatten(1)
    s = e8m0_to_f32(s_u8)
    return x * s.repeat_interleave(x.shape[1] // s.shape[1], 1)


