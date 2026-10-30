"""Print the three NVML power readings every 0.5 s across a load step.

The load (an 8192 x 8192 matmul) is off for 6 s, on for 10 s, then off for 8 s.
"""
import threading
import time

import pynvml
import torch

pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)


def watts(f):
    return pynvml.nvmlDeviceGetFieldValues(gpu, [f])[0].value.uiVal / 1000


x = torch.randn(8192, 8192, device="cuda")
torch.cuda.synchronize()
stop, on = threading.Event(), threading.Event()


def work():
    while not stop.is_set():
        if on.is_set():
            x @ x
            torch.cuda.synchronize()
        else:
            time.sleep(0.01)


threading.Thread(target=work, daemon=True).start()

start = time.perf_counter()
next_print = 0.0
while (t := time.perf_counter() - start) < 24:
    if 6 <= t < 16:
        on.set()
    else:
        on.clear()
    if t >= next_print:
        print(f"t={t:5.2f} load={'on ' if on.is_set() else 'off'}"
              f" usage={pynvml.nvmlDeviceGetPowerUsage(gpu) / 1000:6.1f}"
              f" instant={watts(pynvml.NVML_FI_DEV_POWER_INSTANT):6.1f}"
              f" average={watts(pynvml.NVML_FI_DEV_POWER_AVERAGE):6.1f}")
        next_print += 0.5
    time.sleep(0.002)
stop.set()
