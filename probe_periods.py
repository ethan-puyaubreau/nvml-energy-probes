"""Measure how often each NVML power source changes value.

The four sources are polled in a tight loop. For each one the script prints
how many times the value changed and the intervals between changes.
"""
import argparse
import statistics
import subprocess
import sys
import time

import pynvml

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=20)
parser.add_argument("--load", action="store_true", help="run an 8192 x 8192 matmul in another process")
args = parser.parse_args()

pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
print("GPU:", pynvml.nvmlDeviceGetName(gpu), "| driver:", pynvml.nvmlSystemGetDriverVersion())


def field(f):
    r = pynvml.nvmlDeviceGetFieldValues(gpu, [f])[0]
    if r.nvmlReturn:
        raise pynvml.NVMLError(r.nvmlReturn)
    return r.value.uiVal


sources = {
    "usage": lambda: pynvml.nvmlDeviceGetPowerUsage(gpu),
    "instant": lambda: field(pynvml.NVML_FI_DEV_POWER_INSTANT),
    "average": lambda: field(pynvml.NVML_FI_DEV_POWER_AVERAGE),
    "energy": lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu),
}

load = None
if args.load:
    load = subprocess.Popen([sys.executable, "-c",
        "import torch, time\n"
        "x = torch.randn(8192, 8192, device='cuda'); t = time.time()\n"
        f"while time.time() - t < {args.seconds + 3}:\n"
        "    y = x @ x; torch.cuda.synchronize()\n"])
    time.sleep(3)

last = {name: None for name in sources}
changes = {name: [] for name in sources}
rounds = 0
end = time.perf_counter() + args.seconds
while time.perf_counter() < end:
    now = time.perf_counter()
    for name, read in sources.items():
        value = read()
        if value != last[name]:
            if last[name] is not None:
                changes[name].append(now)
            last[name] = value
    rounds += 1
if load:
    load.terminate()

print(f"{rounds} rounds in {args.seconds} s (about {args.seconds / rounds * 1e3:.2f} ms per round)\n")
print(f"{'source':8} {'changes':>8} {'median ms':>10} {'p10 ms':>8} {'p90 ms':>8} {'min ms':>8}")
for name, times in changes.items():
    gaps = [(b - a) * 1e3 for a, b in zip(times, times[1:])]
    if len(gaps) < 3:
        print(f"{name:8} {len(times):>8}   too few changes")
        continue
    q = statistics.quantiles(gaps, n=10)
    print(f"{name:8} {len(times):>8} {statistics.median(gaps):>10.1f} {q[0]:>8.1f} {q[-1]:>8.1f} {min(gaps):>8.1f}")
