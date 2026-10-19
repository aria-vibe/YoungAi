/* db_net.c — 各数据库协议共用的 TCP 小工具: 带超时的 connect + 收发到满。
 * 连不上 5 秒就报(本机/局域网一毫秒就通); 收发 20 秒(比查询的 maxTimeMS 10 秒长, 让服务器先报超时而不是我们先断线)。
 * 错误是具体的字("connection refused" / "timed out after 5 s"), 页面"测试连接"和模型看到的是同一句。 */
#include "db.h"
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

int db_tcp_connect(const char *host, int port, char *err, size_t errn) {
    if (!host || !host[0]) { snprintf(err, errn, "no host"); return -1; }
    if (strstr(host, "://")) { snprintf(err, errn, "give a host name or address, not a URL"); return -1; }
    char ps[16]; snprintf(ps, sizeof ps, "%d", port);
    struct addrinfo hints = { .ai_socktype = SOCK_STREAM, .ai_family = AF_UNSPEC }, *res = NULL;
    const int gai = getaddrinfo(host, ps, &hints, &res);
    if (gai) { snprintf(err, errn, "cannot resolve %s (%s)", host, gai_strerror(gai)); return -1; }
    int last = 0;
    for (struct addrinfo *a = res; a; a = a->ai_next) {
        const int fd = socket(a->ai_family, SOCK_STREAM, 0);
        if (fd < 0) { last = errno; continue; }
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
        int rc = connect(fd, a->ai_addr, a->ai_addrlen);
        if (rc < 0 && errno == EINPROGRESS) {
            struct pollfd p = { fd, POLLOUT, 0 };
            rc = poll(&p, 1, MONGO_CONNECT_S * 1000);
            if (rc > 0) { int e = 0; socklen_t el = sizeof e; getsockopt(fd, SOL_SOCKET, SO_ERROR, &e, &el); if (e) { errno = e; rc = -1; } else rc = 0; }
            else if (rc == 0) { errno = ETIMEDOUT; rc = -1; }
        }
        if (rc == 0) {
            fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) & ~O_NONBLOCK);
            struct timeval tv = { MONGO_IO_S, 0 };
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv); setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
            freeaddrinfo(res);
            return fd;
        }
        last = errno; close(fd);
    }
    freeaddrinfo(res);
    snprintf(err, errn, "cannot connect to %s:%d (%s)", host, port, last == ETIMEDOUT ? "timed out after 5 s" : last == ECONNREFUSED ? "connection refused" : strerror(last));
    return -1;
}

bool db_io_all(int fd, void *buf, size_t n, bool write_) {
    uint8_t *p = buf;
    while (n) {
        const ssize_t k = write_ ? write(fd, p, n) : read(fd, p, n);
        if (k < 0) { if (errno == EINTR) continue; return false; }
        if (k == 0) return false;
        p += k; n -= (size_t)k;
    }
    return true;
}
