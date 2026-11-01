#!/usr/bin/env python3
"""api_record_proxy.py — 录请求的 HTTP 转发器: 客户端(aider 容器)打到本机 --listen 口, 原样转给 --upstream, 把每次
/v1/chat/completions 的请求体与响应体逐字存成一行 jsonl(id / time / path / status / request / response / secs)。

为什么: 段 B 要拿真实 agent 请求当料, 训练前缀必须和服务端收到的请求逐字节一致(系统提示 + 示例轮 + 文件内容 + 题面, aider 一共 10 条消息);
aider 的 --verbose 打印是逐行加角色前缀的, 空白丢了就对不上。响应体里带 reasoning_content, 就是学生自己的思考轨迹, 一并存下。

用法: python3 gguf-tools/scripts/api_record_proxy.py --listen 18002 --upstream http://127.0.0.1:18001 --out <jsonl>
只做转发不改字节(不支持流式: aider 跑分是 stream=False; 来了流式请求照转, 响应原样回但不解析)。Ctrl-C 退。
"""
import argparse, http.client, json, os, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    up = urllib.parse.urlparse(a.upstream); lock = threading.Lock(); seq = [0]

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *x):
            pass

        def _fwd(self):
            n = int(self.headers.get("Content-Length") or 0); body = self.rfile.read(n) if n else b""
            t0 = time.time(); c = http.client.HTTPConnection(up.hostname, up.port, timeout=7200)
            hdr = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "accept-encoding", "connection")}
            c.request(self.command, self.path, body=body, headers=hdr); r = c.getresponse(); data = r.read(); secs = time.time() - t0
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() in ("transfer-encoding", "connection", "content-length"): continue
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            if self.path.endswith("/chat/completions") and self.command == "POST":
                try: req = json.loads(body.decode("utf-8")); resp = json.loads(data.decode("utf-8"))
                except ValueError: req = body.decode("utf-8", "replace"); resp = data.decode("utf-8", "replace")
                with lock:
                    seq[0] += 1; rec = {"id": f"rec{seq[0]:03d}", "time": round(t0, 3), "path": self.path, "status": r.status, "secs": round(secs, 1),
                                        "request": req, "response": resp}
                    with open(a.out, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    u = (resp.get("usage") or {}) if isinstance(resp, dict) else {}
                    print(f"[record] {rec['id']} {r.status} {len(req.get('messages', [])) if isinstance(req, dict) else '?'} 条消息 "
                          f"prompt {u.get('prompt_tokens')} completion {u.get('completion_tokens')} {secs:.0f} s → {a.out}", flush=True)

        do_POST = do_GET = _fwd

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(f"[record] 监听 {a.listen} → {a.upstream}, 写 {a.out}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.listen), H).serve_forever()


if __name__ == "__main__":
    main()
