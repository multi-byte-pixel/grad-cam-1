#!/usr/bin/env python3
"""Parallel range downloader for CIFAR-10 (no external deps).

Toronto throttles ~45 kB/s per connection but not per IP, so many parallel
range requests scale nearly linearly.  Targets the resolved cave.* URL
directly to avoid the 301 redirect dropping the Range header.
"""
import concurrent.futures
import hashlib
import os
import sys
import urllib.request

URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
OUT = os.path.expanduser("~/training/data/cifar-10-python.tar.gz")
MD5 = "c58f30108f718f92721af3b95e74349a"
N = 32


def total_size():
    req = urllib.request.Request(URL, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range")
        if not cr:
            raise RuntimeError("server does not support range requests")
        return int(cr.split("/")[-1])


def fetch(args):
    i, start, end = args
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    last = None
    for _ in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return i, r.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise last


def main():
    size = total_size()
    chunk = size // N
    jobs = []
    for i in range(N):
        start = i * chunk
        end = size - 1 if i == N - 1 else start + chunk - 1
        jobs.append((i, start, end))

    parts = [None] * N
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
        for i, data in ex.map(fetch, jobs):
            parts[i] = data

    with open(OUT, "wb") as f:
        for p in parts:
            f.write(p)

    h = hashlib.md5(open(OUT, "rb").read()).hexdigest()
    print(f"downloaded {size} bytes, md5={h}", flush=True)
    if h != MD5:
        print("MD5 MISMATCH", flush=True)
        sys.exit(1)
    print("MD5 OK", flush=True)


if __name__ == "__main__":
    main()
