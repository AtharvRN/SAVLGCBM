#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log system, process, and GPU telemetry to CSV.")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    return parser.parse_args()


def read_proc_stat() -> List[int]:
    with open("/proc/stat", "r", encoding="utf-8") as handle:
        parts = handle.readline().strip().split()[1:]
    return [int(part) for part in parts]


def read_process_stat(pid: int) -> Tuple[int, int, int]:
    with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
        parts = handle.read().split()
    utime = int(parts[13])
    stime = int(parts[14])
    rss_pages = int(parts[23])
    return utime, stime, rss_pages


def read_meminfo() -> Dict[str, int]:
    payload: Dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            payload[key] = int(value.strip().split()[0])
    return payload


def read_loadavg() -> Tuple[float, float, float]:
    with open("/proc/loadavg", "r", encoding="utf-8") as handle:
        a, b, c, *_ = handle.read().strip().split()
    return float(a), float(b), float(c)


def read_cgroup_memory() -> Tuple[int, int]:
    current_candidates = [
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ]
    limit_candidates = [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]

    def _read_first(paths: List[str]) -> int:
        for path in paths:
            if os.path.exists(path):
                raw = Path(path).read_text().strip()
                if raw == "max":
                    return -1
                return int(raw)
        return -1

    return _read_first(current_candidates), _read_first(limit_candidates)


def query_gpu_rows() -> List[Dict[str, str]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    rows: List[Dict[str, str]] = []
    for line in output.strip().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 8:
            continue
        rows.append(
            {
                "gpu_index": fields[0],
                "gpu_name": fields[1],
                "gpu_util": fields[2],
                "gpu_mem_util": fields[3],
                "gpu_mem_used_mb": fields[4],
                "gpu_mem_total_mb": fields[5],
                "gpu_power_w": fields[6],
                "gpu_temp_c": fields[7],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_size = os.sysconf("SC_PAGE_SIZE")
    cpu_count = max(1, os.cpu_count() or 1)

    fieldnames = [
        "timestamp",
        "pid",
        "proc_cpu_pct",
        "proc_rss_mb",
        "mem_total_mb",
        "mem_available_mb",
        "mem_used_mb",
        "swap_free_mb",
        "cgroup_memory_current_mb",
        "cgroup_memory_limit_mb",
        "load1",
        "load5",
        "load15",
        "gpu_index",
        "gpu_name",
        "gpu_util",
        "gpu_mem_util",
        "gpu_mem_used_mb",
        "gpu_mem_total_mb",
        "gpu_power_w",
        "gpu_temp_c",
    ]

    prev_total = sum(read_proc_stat())
    prev_proc_user, prev_proc_sys, _ = read_process_stat(args.pid)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        handle.flush()

        while os.path.exists(f"/proc/{args.pid}"):
            time.sleep(max(0.2, args.interval))

            total_ticks = sum(read_proc_stat())
            proc_user, proc_sys, rss_pages = read_process_stat(args.pid)
            delta_total = max(1, total_ticks - prev_total)
            delta_proc = max(0, (proc_user + proc_sys) - (prev_proc_user + prev_proc_sys))
            proc_cpu_pct = 100.0 * cpu_count * float(delta_proc) / float(delta_total)

            prev_total = total_ticks
            prev_proc_user, prev_proc_sys = proc_user, proc_sys

            meminfo = read_meminfo()
            mem_total_mb = meminfo.get("MemTotal", 0) / 1024.0
            mem_available_mb = meminfo.get("MemAvailable", 0) / 1024.0
            mem_used_mb = max(0.0, mem_total_mb - mem_available_mb)
            swap_free_mb = meminfo.get("SwapFree", 0) / 1024.0
            cgroup_current, cgroup_limit = read_cgroup_memory()
            load1, load5, load15 = read_loadavg()
            gpu_rows = query_gpu_rows() or [{}]

            base = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pid": args.pid,
                "proc_cpu_pct": f"{proc_cpu_pct:.2f}",
                "proc_rss_mb": f"{(rss_pages * page_size) / (1024 ** 2):.2f}",
                "mem_total_mb": f"{mem_total_mb:.2f}",
                "mem_available_mb": f"{mem_available_mb:.2f}",
                "mem_used_mb": f"{mem_used_mb:.2f}",
                "swap_free_mb": f"{swap_free_mb:.2f}",
                "cgroup_memory_current_mb": f"{(cgroup_current / (1024 ** 2)) if cgroup_current >= 0 else -1:.2f}",
                "cgroup_memory_limit_mb": f"{(cgroup_limit / (1024 ** 2)) if cgroup_limit >= 0 else -1:.2f}",
                "load1": f"{load1:.2f}",
                "load5": f"{load5:.2f}",
                "load15": f"{load15:.2f}",
            }

            for gpu in gpu_rows:
                row = dict(base)
                row.update(gpu)
                writer.writerow(row)
            handle.flush()


if __name__ == "__main__":
    main()
