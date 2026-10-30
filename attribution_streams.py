"""Predict the power of two streams from a model fitted on one stream at a time.

Stream 1 runs an fp32 matmul and stream 2 an fp16 matmul, alone and together,
in blocks of 2 s. The model (idle power plus one term per type, on CUPTI
occupancy) is fitted on the single-type blocks, then predicts the blocks where
both run. The energy counter, read at the borders of each block, is the
reference.

By default each stream gets bursts of 40 small (2048 x 2048) matmuls, which
overlap. With --large each stream gets one 8192 x 8192 matmul at a time, at a
25% duty cycle; these fill the GPU and do not overlap.
"""
import argparse
import json
import os
import random
import tempfile
import threading
import time
from collections import Counter

import numpy as np
import pynvml
import torch
from torch.profiler import ProfilerActivity, profile, record_function

REPS = 5
BLOCK_SECONDS = 2.0
BURST = 40
DUTY = 0.25  # with --large only

parser = argparse.ArgumentParser()
parser.add_argument("--large", action="store_true", help="one 8192 x 8192 matmul per stream at a time")
parser.add_argument("--seed", type=int, default=3, help="seed of the block order")
args = parser.parse_args()

pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
size = 8192 if args.large else 2048
A = torch.randn(size, size, device="cuda")
H = A.half()
s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()


def launch(kind):
    if kind in ("fp32", "both"):
        with torch.cuda.stream(s1):
            torch.mm(A, A)
    if kind in ("fp16", "both"):
        with torch.cuda.stream(s2):
            torch.mm(H, H)


for kind in ("fp32", "fp16", "both"):
    launch(kind)
torch.cuda.synchronize()
random.seed(args.seed)
order = ["fp32", "fp16", "both"] * REPS + ["idle"] * 3
random.shuffle(order)

samples, stop = [], threading.Event()


def sampler():
    while not stop.is_set():
        r = pynvml.nvmlDeviceGetFieldValues(gpu, [pynvml.NVML_FI_DEV_POWER_INSTANT])[0]
        samples.append((time.perf_counter(), r.value.uiVal / 1000))
        time.sleep(0.01)


blocks = []  # (kind, start, end, energy at start in mJ, energy at end in mJ)
thread = threading.Thread(target=sampler, daemon=True)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    thread.start()
    with record_function("anchor"):
        anchor = time.perf_counter()
    for kind in order:
        torch.cuda.synchronize()
        e0, t0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu), time.perf_counter()
        while time.perf_counter() - t0 < BLOCK_SECONDS:
            if kind == "idle":
                time.sleep(0.01)
                continue
            start = time.perf_counter()  # 100 ms period
            if args.large:
                while time.perf_counter() - start < 0.1 * DUTY:
                    launch(kind)
                    torch.cuda.synchronize()
            else:
                for _ in range(BURST):
                    launch(kind)
                torch.cuda.synchronize()
            rest = 0.1 - (time.perf_counter() - start)
            if rest > 0:
                time.sleep(rest)
        e1, t1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu), time.perf_counter()
        blocks.append((kind, t0, t1, e0, e1))
    stop.set()
    thread.join()

trace = os.path.join(tempfile.gettempdir(), "attribution_streams.json")
prof.export_chrome_trace(trace)
events = json.load(open(trace))["traceEvents"]
offset = [e for e in events if e.get("name") == "anchor"][0]["ts"] / 1e6 - anchor
kernels = [(e["ts"] / 1e6 - offset, (e["ts"] + e["dur"]) / 1e6 - offset, e["args"].get("stream"))
           for e in events if e.get("cat") == "kernel"]


def block_of(t):
    for b in blocks:
        if b[1] - 0.02 <= t < b[2] + 0.02:
            return b[0]


# the trace numbers streams its own way: learn them from the single-type blocks
seen = {"fp32": Counter(), "fp16": Counter()}
for start, end, stream in kernels:
    kind = block_of(start)
    if kind in seen:
        seen[kind][stream] += 1
stream_of = {kind: seen[kind].most_common(1)[0][0] for kind in seen}
print(f"{len(kernels)} CUPTI kernels; trace stream {stream_of['fp32']} runs fp32, {stream_of['fp16']} runs fp16")
by_type = {kind: [(s, e) for s, e, st in kernels if st == stream_of[kind]] for kind in seen}


def occupancy(intervals, w0, w1):
    return sum(max(0, min(e, w1) - max(s, w0)) for s, e in intervals) / (w1 - w0)


def union_occupancy(w0, w1):
    spans = sorted((max(s, w0), min(e, w1)) for s, e in by_type["fp32"] + by_type["fp16"] if e > w0 and s < w1)
    total, current = 0.0, None
    for s, e in spans:
        if current is None or s > current[1]:
            if current:
                total += current[1] - current[0]
            current = [s, e]
        else:
            current[1] = max(current[1], e)
    if current:
        total += current[1] - current[0]
    return total / (w1 - w0)


ts = np.array([s[0] for s in samples])
watts = np.array([s[1] for s in samples])
LAG, WINDOW = 0.5, 0.5  # the power reading trails the activity by about 0.5 s
X, y = [], []
for b in blocks:
    if b[0] == "both":
        continue
    t = b[1]
    while t + LAG + WINDOW <= b[2]:
        window = (ts >= t + LAG) & (ts < t + LAG + WINDOW)
        if window.sum() > 5:
            X.append([1, occupancy(by_type["fp32"], t, t + WINDOW), occupancy(by_type["fp16"], t, t + WINDOW)])
            y.append(watts[window].mean())
        t += 0.25
X, y = np.array(X), np.array(y)
coef = np.linalg.lstsq(X, y, rcond=None)[0]
r2 = 1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"\nmodel fitted on single-type blocks ({len(y)} windows, R2 {r2:.3f}):"
      f" idle {coef[0]:.0f} W, fp32 {coef[1]:.0f} W, fp16 {coef[2]:.0f} W per unit of occupancy")
print("\nblock  occ fp32  occ fp16  union  counter W        model W  model - counter")
for kind in ("fp32", "fp16", "both", "idle"):
    rows = []
    for b in [b for b in blocks if b[0] == kind]:
        occ32, occ16 = occupancy(by_type["fp32"], b[1], b[2]), occupancy(by_type["fp16"], b[1], b[2])
        counter = (b[4] - b[3]) / 1e3 / (b[2] - b[1])
        rows.append((occ32, occ16, union_occupancy(b[1], b[2]), counter, coef[0] + coef[1] * occ32 + coef[2] * occ16))
    m = np.array(rows).mean(0)
    print(f"{kind:5} {m[0]:9.2f} {m[1]:9.2f} {m[2]:6.2f} {m[3]:7.1f} +/-{np.array(rows)[:, 3].std():4.1f}"
          f" {m[4]:8.1f} {m[4] - m[3]:+10.1f}")
