"""Read the energy counter at several rates under a constant matmul load.

The counter is cumulative, so the power derived from its first and last value
should not depend on how often it is read in between.
"""
import argparse
import subprocess
import sys
import time

import pynvml

RATES = (1, 2, 5, 10, 15, 20, 50, 150, 400, 0)  # Hz, 0 reads as fast as possible

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=8, help="time spent at each rate")
args = parser.parse_args()

load = subprocess.Popen([sys.executable, "-c",
    "import torch, time\n"
    "x = torch.randn(8192, 8192, device='cuda'); t = time.time()\n"
    f"while time.time() - t < {args.seconds * len(RATES) + 10}:\n"
    "    y = x @ x; torch.cuda.synchronize()\n"])
time.sleep(5)

pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
for hz in RATES:
    e0, t0, reads = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu), time.perf_counter(), 0
    while time.perf_counter() - t0 < args.seconds:
        pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu)
        reads += 1
        if hz:
            time.sleep(1 / hz)
    e1, elapsed = pynvml.nvmlDeviceGetTotalEnergyConsumption(gpu), time.perf_counter() - t0
    print(f"{reads / elapsed:7.1f} Hz: counter {(e1 - e0) / 1e3 / elapsed:6.1f} W"
          f" (usage {pynvml.nvmlDeviceGetPowerUsage(gpu) / 1e3:.0f} W)")
load.terminate()
