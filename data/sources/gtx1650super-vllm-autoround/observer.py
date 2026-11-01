#!/usr/bin/env python3
"""Transparent TCP observer for the lab run (task tool, not part of the lab). Listens on LISTEN_HOST:LISTEN_PORT and
copies every byte unchanged to UPSTREAM_HOST:UPSTREAM_PORT and back; it parses nothing while relaying. Each connection's
two directions are also written verbatim to OUTDIR/conn-<n>.req and conn-<n>.resp for later inspection (request bodies,
streamed finish_reason). Usage: observer.py LISTEN_HOST LISTEN_PORT UPSTREAM_HOST UPSTREAM_PORT OUTDIR"""
import itertools, os, socket, sys, threading

def pipe(src, dst, log):
    try:
        while True:
            b = src.recv(65536)
            if not b:
                break
            dst.sendall(b)
            log.write(b)
            log.flush()
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass

def handle(client, n, up, out):
    server = socket.create_connection(up)
    with open(f"{out}/conn-{n:04d}.req", "wb") as rq, open(f"{out}/conn-{n:04d}.resp", "wb") as rs:
        t = threading.Thread(target=pipe, args=(client, server, rq), daemon=True)
        t.start()
        pipe(server, client, rs)
        t.join()
    client.close()
    server.close()

def main():
    lh, lp, uh, upp, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
    os.makedirs(out, exist_ok=True)
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((lh, lp))
    s.listen(16)
    for n in itertools.count():
        c, _ = s.accept()
        threading.Thread(target=handle, args=(c, n, (uh, upp), out), daemon=True).start()

if __name__ == "__main__":
    main()
