#!/usr/bin/env python3
"""
ATB Training Profiler
Samples GPU + CPU stats every second for 10 seconds.
Run WHILE atb-train is running in another terminal:

    python profile_training.py

Paste the output here for analysis.
"""

import json
import subprocess
import sys
import time

try:
    import psutil
except ImportError:
    print("pip install psutil first:  pip install psutil")
    sys.exit(1)

DURATION = 10
INTERVAL = 1.0


def _detect_gpu_query() -> str:
    """Probe which utilization flag this driver version accepts."""
    for util_flag in ("utilization.gpu", "gpu_util"):
        query = (
            f"{util_flag},memory.used,memory.total,power.draw,"
            "temperature.gpu,clocks.current.graphics,clocks.current.memory"
        )
        try:
            subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            return query
        except subprocess.SubprocessError:
            continue
        except OSError:
            break
    return ""


GPU_QUERY = _detect_gpu_query()


def query_gpu() -> dict:
    if not GPU_QUERY:
        return {"error": "nvidia-smi not found or no supported query flags"}
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip()
        parts = [p.strip() for p in raw.split(",")]
        return {
            "gpu_util_pct":  int(parts[0]),
            "vram_used_mib": int(parts[1]),
            "vram_total_mib": int(parts[2]),
            "power_w":       float(parts[3]),
            "temp_c":        int(parts[4]),
            "core_mhz":      int(parts[5]),
            "mem_mhz":       int(parts[6]),
        }
    except subprocess.TimeoutExpired:
        return {"error": "nvidia-smi timeout"}
    except (ValueError, IndexError) as exc:
        return {"error": f"parse error: {exc}"}


def query_cpu() -> dict:
    per_core: list[float] = psutil.cpu_percent(percpu=True)  # type: ignore[assignment]
    mem = psutil.virtual_memory()
    return {
        "cpu_avg_pct":    round(sum(per_core) / len(per_core), 1),
        "cpu_max_pct":    round(max(per_core), 1),
        "cores_busy":     sum(1 for c in per_core if c > 50),
        "n_cores":        len(per_core),
        "ram_used_gib":   round(mem.used / 1024**3, 2),
        "ram_total_gib":  round(mem.total / 1024**3, 2),
        "per_core_pct":   [round(c, 1) for c in per_core],
    }


def print_sample(ts: int, gpu: dict, cpu: dict) -> None:
    cpu_str = (
        f"CPU {cpu['cpu_avg_pct']:>4}% avg "
        f"({cpu['cores_busy']}/{cpu['n_cores']} cores >50%) | "
        f"RAM {cpu['ram_used_gib']:.1f}/{cpu['ram_total_gib']:.0f} GiB"
    )
    if "error" not in gpu:
        gpu_str = (
            f"GPU {gpu['gpu_util_pct']:>3}% | "
            f"VRAM {gpu['vram_used_mib']:>4}/{gpu['vram_total_mib']} MiB | "
            f"{gpu['power_w']:>5.1f}W | "
            f"{gpu['temp_c']}°C | "
        )
    else:
        gpu_str = f"GPU n/a ({gpu['error']}) | "
    print(f"  [{ts:>2}s]  {gpu_str}{cpu_str}")


def print_summary(samples: list[dict]) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  SUMMARY")
    print(sep)

    gpu_samples = [s["gpu"] for s in samples if "error" not in s["gpu"]]
    cpu_samples = [s["cpu"] for s in samples]

    if gpu_samples:
        n = len(gpu_samples)
        avg_util = sum(g["gpu_util_pct"] for g in gpu_samples) / n
        max_util = max(g["gpu_util_pct"] for g in gpu_samples)
        avg_vram = sum(g["vram_used_mib"] for g in gpu_samples) / n
        peak_vram = max(g["vram_used_mib"] for g in gpu_samples)
        avg_pow = sum(g["power_w"] for g in gpu_samples) / n
        max_pow = max(g["power_w"] for g in gpu_samples)
        print(f"  GPU util    avg={avg_util:.0f}%  max={max_util}%")
        print(f"  VRAM        avg={avg_vram:.0f} MiB  peak={peak_vram} MiB")
        print(f"  Power       avg={avg_pow:.1f}W  max={max_pow:.1f}W")
    else:
        print("  GPU: no data (nvidia-smi not available)")

    avg_cpu = sum(s["cpu_avg_pct"] for s in cpu_samples) / len(cpu_samples)
    peak_cpu = max(s["cpu_avg_pct"] for s in cpu_samples)
    peak_core = max(s["cpu_max_pct"] for s in cpu_samples)
    peak_ram = max(s["ram_used_gib"] for s in cpu_samples)
    print(f"  CPU avg     avg={avg_cpu:.0f}%  peak={peak_cpu:.0f}%")
    print(f"  CPU max     peak single core: {peak_core:.0f}%")
    print(f"  RAM         peak={peak_ram:.2f} GiB")

    last_cores: list[float] = cpu_samples[-1]["per_core_pct"]
    print("\n  Per-core % (last sample):")
    for i in range(0, len(last_cores), 4):
        row = last_cores[i:i + 4]
        line = "  ".join(f"Core{i + j:>2}: {v:>5.1f}%" for j, v in enumerate(row))
        print(f"    {line}")

    out_path = "profile_result.json"
    with open(out_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"\n  Raw data saved → {out_path}")
    print(f"{sep}\n")


def main() -> None:
    # Warm-up psutil (first call always returns 0).
    psutil.cpu_percent(percpu=True)
    time.sleep(0.5)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  ATB Training Profiler — {DURATION}s @ {INTERVAL}s interval")
    print(sep)
    print("  Make sure atb-train is running in another terminal!\n")

    samples: list[dict] = []

    for i in range(DURATION):
        t0 = time.perf_counter()
        gpu = query_gpu()
        cpu = query_cpu()
        ts = i + 1
        samples.append({"t_s": ts, "gpu": gpu, "cpu": cpu})
        print_sample(ts, gpu, cpu)
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, INTERVAL - elapsed))

    print_summary(samples)


if __name__ == "__main__":
    main()