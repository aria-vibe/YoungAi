#!/usr/bin/env python3
"""dsh_capture_proxy.py — 本机 HTTP 代理: dsh 发给官方 Anthropic 接口的每个请求体原样落盘, 响应流式透传(2026-10-26 晚, agent 多轮料)。

为什么: ③ 要学"看了工具结果之后, 交还是接着改"这个多轮决策, 料里一条多轮 agent 会话都没有; dsh 的 events.jsonl 把思考/写入截在 8~9k 字,
~/.dsh/sessions 也不是请求体 —— 只有抓到 dsh 真发出去的每轮请求体(system + tools + 全部历史), 才能①让官方 API 在同一前缀上重采拿 top-20
(gguf-tools/scripts/api_agent.py resample) ②用 ds4-server --render-anthropic 按部署同一套渲染成学生看见的那串字节。
用法: tests/asst/dsh_capture_proxy.py --port 18433 --upstream https://api.deepseek.com/anthropic --dir <抓包目录>
  dsh_compare.sh 第 4 个参数给抓包目录时自动起它、跑完杀; 产物 <目录>/req_NNN.json(请求体) + resp_NNN.sse(上游原响应) + meta_NNN.json(路径/状态码/用时)。
  key 由 dsh 放在请求头里经这里透传, 不落盘(头只转发, meta 只记名字不记值)。
出错会怎样: 上游 4xx/5xx 原样回给 dsh(状态码 + 正文), 也落盘; 本地连接异常打到 stderr, 不吞。"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP = {"host", "content-length", "connection", "transfer-encoding", "accept-encoding", "keep-alive"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seq = 0; lock = threading.Lock(); upstream = ""; outdir = ""

    def log_message(self, fmt, *args):   # 默认日志进 stderr 太吵, 只留我们自己的一行
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); body = self.rfile.read(n) if n else b""
        with Handler.lock: Handler.seq += 1; k = Handler.seq
        tag = f"{k:03d}"; t0 = time.time()
        with open(os.path.join(Handler.outdir, f"req_{tag}.json"), "wb") as f: f.write(body)
        hdr = {h: v for h, v in self.headers.items() if h.lower() not in HOP}
        hdr["Accept-Encoding"] = "identity"   # 上游别压缩, 落盘的 .sse 才是明文
        req = urllib.request.Request(Handler.upstream + self.path, data=body, method="POST", headers=hdr)
        try:
            up = urllib.request.urlopen(req, timeout=3600); status = up.status
        except urllib.error.HTTPError as e:
            up = e; status = e.code
        self.send_response(status)
        ctype = up.headers.get("Content-Type", "application/json")
        self.send_header("Content-Type", ctype); self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        total = 0; dropped = ""
        with open(os.path.join(Handler.outdir, f"resp_{tag}.sse"), "wb") as f:
            try:
                while True:
                    chunk = up.read(4096)
                    if not chunk: break
                    f.write(chunk); total += len(chunk)
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk)); self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError) as e:   # dsh 中途挂断(收尾后顺手发的请求它自己不等), 上游剩下的继续落盘到此为止
                dropped = type(e).__name__
        json.dump({"path": self.path, "status": status, "req_bytes": len(body), "resp_bytes": total, "secs": round(time.time() - t0, 1),
                   "client_dropped": dropped, "headers": sorted(h for h in hdr)}, open(os.path.join(Handler.outdir, f"meta_{tag}.json"), "w"))
        print(f"[capture] #{tag} {self.path} {status} 请求 {len(body)} B 响应 {total} B {time.time() - t0:.1f} s{(' 客户端中途挂断 ' + dropped) if dropped else ''}", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, required=True); ap.add_argument("--upstream", required=True); ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    Handler.upstream = a.upstream.rstrip("/"); Handler.outdir = a.dir
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"[capture] 127.0.0.1:{a.port} → {Handler.upstream}, 落盘 {a.dir}", file=sys.stderr, flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
