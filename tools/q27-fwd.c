/*
 * q27-fwd: zero-fork, zero-copy-path epoll TCP <-> UDS/TCP proxy for q27-engine.
 * Replaces `socat TCP-LISTEN:8000,fork` (backlog=5, fork-per-conn, Nagle on).
 *
 * Features:
 *   - Single-process epoll event loop (zero fork, zero thread creation)
 *   - Listen backlog 1024, SO_REUSEADDR, TCP_NODELAY + TCP_QUICKACK on all TCP sockets
 *   - Supports TCP -> Unix Domain Socket (unix:/path) OR TCP -> TCP (ip:port)
 *   - 64 KiB per-direction ring buffers, clean TCP half-close (SHUT_WR) for HTTP/1.1 keep-alive & SSE
 *
 * Usage:
 *   q27-fwd <bind_ip>:<port> unix:<socket_path>
 *   q27-fwd <bind_ip>:<port> <dst_ip>:<dst_port>
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define MAX_EVENTS 128
#define BUF_SIZE   65536
#define BACKLOG    1024

typedef struct Conn Conn;

typedef struct {
    int fd;
    bool is_tcp;
    bool eof_in;
    bool shut_out;
    bool connecting;
    uint32_t events;
    size_t head, len;
    uint8_t buf[BUF_SIZE]; /* data read FROM this side, to be written TO peer */
    Conn *conn;
} Side;

struct Conn {
    Side cli;
    Side srv;
    bool closing;
    Conn *next_dead;
};

static int g_ep = -1;
static int g_listen = -1;
static volatile sig_atomic_t g_stop = 0;
static Conn *g_dead = NULL;

static bool g_dst_unix = false;
static struct sockaddr_un g_dst_un;
static struct sockaddr_in g_dst_in;

static void on_sig(int sig) {
    (void)sig;
    g_stop = 1;
}

static void set_tcp_fast(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef TCP_QUICKACK
    setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));
#endif
}

static void ep_mod(Side *s, uint32_t ev) {
    if (s->fd < 0 || s->events == ev) return;
    struct epoll_event e = { .events = ev, .data.ptr = s };
    if (epoll_ctl(g_ep, EPOLL_CTL_MOD, s->fd, &e) == 0) {
        s->events = ev;
    }
}

static void close_conn(Conn *c) {
    if (c->closing) return;
    c->closing = true;
    if (c->cli.fd >= 0) {
        epoll_ctl(g_ep, EPOLL_CTL_DEL, c->cli.fd, NULL);
        close(c->cli.fd);
        c->cli.fd = -1;
    }
    if (c->srv.fd >= 0) {
        epoll_ctl(g_ep, EPOLL_CTL_DEL, c->srv.fd, NULL);
        close(c->srv.fd);
        c->srv.fd = -1;
    }
    c->next_dead = g_dead;
    g_dead = c;
}

static void free_dead(void) {
    while (g_dead) {
        Conn *c = g_dead;
        g_dead = c->next_dead;
        free(c);
    }
}

static void update_interests(Conn *c) {
    if (c->closing) return;
    Side *a = &c->cli, *b = &c->srv;
    if ((a->eof_in && a->len == 0 && b->eof_in && b->len == 0) ||
        (a->shut_out && b->shut_out)) {
        close_conn(c);
        return;
    }
    uint32_t ea = EPOLLRDHUP, eb = EPOLLRDHUP;
    if (!a->eof_in && a->len < BUF_SIZE) ea |= EPOLLIN;
    if (b->len > 0 && !a->shut_out) ea |= EPOLLOUT;

    if (b->connecting) {
        eb |= EPOLLOUT;
    } else {
        if (!b->eof_in && b->len < BUF_SIZE) eb |= EPOLLIN;
        if (a->len > 0 && !b->shut_out) eb |= EPOLLOUT;
    }
    ep_mod(a, ea);
    ep_mod(b, eb);
}

static bool flush_to(Side *src, Side *dst) {
    while (src->len > 0 && !dst->shut_out && !dst->connecting) {
        size_t avail = BUF_SIZE - src->head;
        size_t chunk = src->len < avail ? src->len : avail;
        ssize_t n = send(dst->fd, src->buf + src->head, chunk, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (n > 0) {
            src->head = (src->head + (size_t)n) % BUF_SIZE;
            src->len -= (size_t)n;
            if (src->len == 0) src->head = 0;
        } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            break;
        } else if (n < 0 && errno == EINTR) {
            continue;
        } else {
            return false;
        }
    }
    if (src->eof_in && src->len == 0 && !dst->shut_out && !dst->connecting) {
        shutdown(dst->fd, SHUT_WR);
        dst->shut_out = true;
    }
    return true;
}

static bool read_from(Side *src, Side *dst) {
    while (!src->eof_in && src->len < BUF_SIZE) {
        size_t tail = (src->head + src->len) % BUF_SIZE;
        size_t space = BUF_SIZE - src->len;
        size_t cont = BUF_SIZE - tail;
        size_t chunk = space < cont ? space : cont;
        ssize_t n = recv(src->fd, src->buf + tail, chunk, MSG_DONTWAIT);
        if (n > 0) {
            src->len += (size_t)n;
            /* Immediately push to peer to minimize latency (zero extra epoll turn!) */
            if (!flush_to(src, dst)) return false;
        } else if (n == 0) {
            src->eof_in = true;
            if (!flush_to(src, dst)) return false;
            break;
        } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
            break;
        } else if (errno == EINTR) {
            continue;
        } else {
            return false;
        }
    }
#ifdef TCP_QUICKACK
    if (src->is_tcp) {
        int one = 1;
        setsockopt(src->fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one));
    }
#endif
    return true;
}

static void accept_all(void) {
    for (;;) {
        int cfd = accept4(g_listen, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (cfd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            if (errno == EINTR) continue;
            break;
        }
        set_tcp_fast(cfd);

        int sfd = socket(g_dst_unix ? AF_UNIX : AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
        if (sfd < 0) {
            close(cfd);
            continue;
        }
        if (!g_dst_unix) set_tcp_fast(sfd);

        int rc = g_dst_unix
            ? connect(sfd, (struct sockaddr *)&g_dst_un, sizeof(g_dst_un))
            : connect(sfd, (struct sockaddr *)&g_dst_in, sizeof(g_dst_in));
        bool connecting = false;
        if (rc < 0) {
            if (errno == EINPROGRESS) {
                connecting = true;
            } else {
                close(sfd);
                close(cfd);
                continue;
            }
        }

        Conn *c = calloc(1, sizeof(Conn));
        if (!c) {
            close(sfd);
            close(cfd);
            continue;
        }
        c->cli.fd = cfd;
        c->cli.is_tcp = true;
        c->cli.conn = c;
        c->srv.fd = sfd;
        c->srv.is_tcp = !g_dst_unix;
        c->srv.connecting = connecting;
        c->srv.conn = c;

        struct epoll_event ec = { .events = EPOLLIN | EPOLLRDHUP, .data.ptr = &c->cli };
        struct epoll_event es = { .events = (connecting ? EPOLLOUT : EPOLLIN) | EPOLLRDHUP, .data.ptr = &c->srv };
        c->cli.events = ec.events;
        c->srv.events = es.events;
        if (epoll_ctl(g_ep, EPOLL_CTL_ADD, cfd, &ec) < 0 ||
            epoll_ctl(g_ep, EPOLL_CTL_ADD, sfd, &es) < 0) {
            close_conn(c);
            continue;
        }
    }
}

static bool parse_hostport(const char *s, struct sockaddr_in *out) {
    const char *colon = strrchr(s, ':');
    if (!colon) return false;
    char ip[64];
    size_t n = (size_t)(colon - s);
    if (n == 0 || n >= sizeof(ip)) return false;
    memcpy(ip, s, n);
    ip[n] = '\0';
    int port = atoi(colon + 1);
    if (port <= 0 || port > 65535) return false;
    memset(out, 0, sizeof(*out));
    out->sin_family = AF_INET;
    out->sin_port = htons((uint16_t)port);
    return inet_pton(AF_INET, ip, &out->sin_addr) == 1;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <bind_ip>:<port> <unix:/path | dst_ip:port>\n", argv[0]);
        return 2;
    }
    struct sockaddr_in bind_addr;
    if (!parse_hostport(argv[1], &bind_addr)) {
        fprintf(stderr, "Invalid bind address: %s\n", argv[1]);
        return 2;
    }
    if (strncmp(argv[2], "unix:", 5) == 0) {
        g_dst_unix = true;
        memset(&g_dst_un, 0, sizeof(g_dst_un));
        g_dst_un.sun_family = AF_UNIX;
        strncpy(g_dst_un.sun_path, argv[2] + 5, sizeof(g_dst_un.sun_path) - 1);
    } else if (!parse_hostport(argv[2], &g_dst_in)) {
        fprintf(stderr, "Invalid target address: %s\n", argv[2]);
        return 2;
    }

    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    g_listen = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (g_listen < 0) { perror("socket"); return 1; }
    int one = 1;
    setsockopt(g_listen, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    set_tcp_fast(g_listen);
    if (bind(g_listen, (struct sockaddr *)&bind_addr, sizeof(bind_addr)) < 0) { perror("bind"); return 1; }
    if (listen(g_listen, BACKLOG) < 0) { perror("listen"); return 1; }

    g_ep = epoll_create1(EPOLL_CLOEXEC);
    if (g_ep < 0) { perror("epoll_create1"); return 1; }
    struct epoll_event el = { .events = EPOLLIN, .data.ptr = NULL };
    if (epoll_ctl(g_ep, EPOLL_CTL_ADD, g_listen, &el) < 0) { perror("epoll_ctl"); return 1; }

    fprintf(stderr, "[q27-fwd] listening on %s -> %s (epoll, backlog=%d, TCP_NODELAY=1)\n", argv[1], argv[2], BACKLOG);

    struct epoll_event evs[MAX_EVENTS];
    while (!g_stop) {
        int nfds = epoll_wait(g_ep, evs, MAX_EVENTS, 1000);
        if (nfds < 0) {
            if (errno == EINTR) continue;
            break;
        }
        for (int i = 0; i < nfds; i++) {
            if (evs[i].data.ptr == NULL) {
                accept_all();
                continue;
            }
            Side *s = (Side *)evs[i].data.ptr;
            Conn *c = s->conn;
            if (c->closing) continue;
            Side *peer = (s == &c->cli) ? &c->srv : &c->cli;
            uint32_t rev = evs[i].events;

            if (rev & EPOLLERR) {
                close_conn(c);
                continue;
            }
            if (s->connecting && (rev & EPOLLOUT)) {
                int err = 0;
                socklen_t elen = sizeof(err);
                if (getsockopt(s->fd, SOL_SOCKET, SO_ERROR, &err, &elen) < 0 || err != 0) {
                    close_conn(c);
                    continue;
                }
                s->connecting = false;
                if (!flush_to(peer, s)) { close_conn(c); continue; }
            }
            if (!c->closing && (rev & (EPOLLIN | EPOLLHUP | EPOLLRDHUP))) {
                if (!read_from(s, peer)) { close_conn(c); continue; }
            }
            if (!c->closing && (rev & EPOLLOUT)) {
                if (!flush_to(peer, s)) { close_conn(c); continue; }
            }
            if (!c->closing) update_interests(c);
        }
        free_dead();
    }
    free_dead();
    close(g_listen);
    close(g_ep);
    return 0;
}
