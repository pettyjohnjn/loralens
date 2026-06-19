#!/usr/bin/env python3
"""Run a command and report peak resident memory for the process tree."""

import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Set


def _children(pid: int) -> List[int]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        text = path.read_text().strip()
    except OSError:
        return []
    if not text:
        return []
    return [int(value) for value in text.split()]


def _rss_kb(pid: int) -> int:
    status = Path(f"/proc/{pid}/status")
    try:
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        return 0
    return 0


def _process_tree(root_pid: int) -> Set[int]:
    seen = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        if not Path(f"/proc/{pid}").exists():
            continue
        seen.add(pid)
        stack.extend(_children(pid))
    return seen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-sec", type=float, default=0.5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command")

    start = time.time()
    proc = subprocess.Popen(command)
    peak_kb = 0
    peak_pids = []

    while proc.poll() is None:
        pids = sorted(_process_tree(proc.pid))
        total_kb = sum(_rss_kb(pid) for pid in pids)
        if total_kb > peak_kb:
            peak_kb = total_kb
            peak_pids = pids
        time.sleep(args.interval_sec)

    pids = sorted(_process_tree(proc.pid))
    total_kb = sum(_rss_kb(pid) for pid in pids)
    if total_kb > peak_kb:
        peak_kb = total_kb
        peak_pids = pids

    elapsed = time.time() - start
    print(f"[rss] max_rss_kb={peak_kb}")
    print(f"[rss] max_rss_gb={peak_kb / (1024 ** 2):.3f}")
    print(f"[rss] max_rss_pid_count={len(peak_pids)}")
    print(f"[rss] elapsed_sec={elapsed:.1f}")
    return int(proc.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
