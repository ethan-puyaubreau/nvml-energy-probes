"""Split board power between kernel types using CUPTI occupancy.

Blocks of 2 s run one kernel type at a 25% or 50% duty cycle, in shuffled
order. The script fits idle power plus one term per type on the occupancy that
CUPTI reports, and compares each block with the energy counter read at its
borders.
"""
import argparse
import json
import os
import random
import tempfile
import threading
import time

import numpy as np
import pynvml
import torch
from torch.profiler import ProfilerActivity, profile, record_function

DUTIES = (0.25, 0.5)
REPS = 3  # blocks per type and duty cycle
BLOCK_SECONDS = 2.0
TYPES = ("fp32", "membound", "fp16")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=2, help="seed of the block order")
args = parser.parse_args()

pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
A = torch.randn(8192, 8192, device="cuda")
H = A.half()
V = torch.randn(128_000_000, device="cuda")  # 512 MB
kernels_of = {"fp32": lambda: torch.mm(A, A), "membound": lambda: V.mul_(1.0001), "fp16": lambda: torch.mm(H, H)}

random.seed(args.seed)
order = [(kind, duty) for kind in TYPES for duty in DUTIES] * REPS + [("idle", 0.0)] * REPS
random.shuffle(order)

samples, stop = [], threading.Event()


def sampler():
    while not stop.is_set():
        r = pynvml.nvmlDeviceGetFieldValues(gpu, [pynvml.NVML_FI_DEV_POWER_INSTANT])[0]
        samples.append((time.perf_counter(), r.value.uiVal / 1000,
                        pynvml.nvmlDeviceGetClockInfo(gpu, pynvml.NVML_CLOCK_SM)))
        time.sleep(0.01)


for run in kernels_of.values():
    for _ in range(3):
        run()
torch.cuda.synchronize()

blocks = []  # (kind, start, end, energy at start in mJ, energy at end in mJ, duty)
thread = threading.Thread(target=sampler, daemon=True)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    thread.start()
    with record_function("anchor"):
        anchor = time.perf_counter()
    for kind, duty in order:
        torch.cuda.synchronize()
        e0, t0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu), time.perf_counter()
        while time.perf_counter() - t0 < BLOCK_SECONDS:
            if kind == "idle":
                time.sleep(0.01)
                continue
            start = time.perf_counter()  # 100 ms period, busy for the first duty * 100 ms
            while time.perf_counter() - start < 0.1 * duty:
                kernels_of[kind]()
                torch.cuda.synchronize()
            rest = 0.1 - (time.perf_counter() - start)
            if rest > 0:
                time.sleep(rest)
        e1, t1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu), time.perf_counter()
        blocks.append((kind, t0, t1, e0, e1, duty))
    stop.set()
    thread.join()

trace = os.path.join(tempfile.gettempdir(), "attribution_kinds.json")
prof.export_chrome_trace(trace)
events = json.load(open(trace))["traceEvents"]
offset = [e for e in events if e.get("name") == "anchor"][0]["ts"] / 1e6 - anchor  # trace s = perf s + offset
kernels = [(e["ts"] / 1e6 - offset, (e["ts"] + e["dur"]) / 1e6 - offset)
           for e in events if e.get("cat") == "kernel"]
print(f"{len(kernels)} CUPTI kernels, {len(samples)} NVML samples, clock offset {offset:.3f} s")


def block_of(t):
    for b in blocks:
        if b[1] - 0.02 <= t < b[2] + 0.02:
            return b[0]


by_kind = {kind: [] for kind in TYPES}  # a kernel belongs to the block in which it starts
for start, end in kernels:
    kind = block_of(start)
    if kind in by_kind:
        by_kind[kind].append((start, end))


def occupancy(intervals, w0, w1):
    return sum(max(0, min(e, w1) - max(s, w0)) for s, e in intervals) / (w1 - w0)


ts = np.array([s[0] for s in samples])
watts = np.array([s[1] for s in samples])
best = None
for lag in (0.0, 0.25, 0.5, 0.75):  # the power reading trails the activity
    X, y = [], []
    t = blocks[0][1] + 0.5
    while t + 0.5 <= blocks[-1][2]:
        window = (ts >= t + lag) & (ts < t + lag + 0.5)
        if window.sum() > 5:
            X.append([1.0] + [occupancy(by_kind[kind], t, t + 0.5) for kind in TYPES])
            y.append(watts[window].mean())
        t += 0.25
    X, y = np.array(X), np.array(y)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    if best is None or r2 > best[0]:
        best = (r2, lag, coef, len(y))
r2, lag, coef, n = best
print(f"\nbest lag {lag} s, R2 {r2:.3f}, {n} windows")
print(f"idle {coef[0]:.0f} W | " + " | ".join(f"{kind} {c:.0f} W" for kind, c in zip(TYPES, coef[1:]))
      + " per unit of occupancy")

print("\nblock     duty  occupancy  counter W  model W  SM clock MHz")
for kind in TYPES + ("idle",):
    for duty in sorted({b[5] for b in blocks if b[0] == kind}):
        group = [b for b in blocks if b[0] == kind and b[5] == duty]
        counter = np.mean([(b[4] - b[3]) / 1e3 / (b[2] - b[1]) for b in group])
        occ = np.mean([occupancy(by_kind[kind], b[1], b[2]) for b in group]) if kind != "idle" else 0.0
        model = coef[0] + (coef[TYPES.index(kind) + 1] * occ if kind != "idle" else 0.0)
        clock = np.mean([s[2] for s in samples for b in group if b[1] + 0.5 < s[0] < b[2]])
        print(f"{kind:8} {duty:5.2f} {occ:10.2f} {counter:10.1f} {model:8.1f} {clock:13.0f}")
