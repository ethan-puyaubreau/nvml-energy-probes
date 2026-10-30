"""Print the mean and standard deviation of the outputs in results/."""
import glob
import re
import statistics

import numpy as np

NUM = r"([-+]?\d+(?:\.\d+)?)"
TYPES = ("fp32", "membound", "fp16")
BLOCKS = ("fp32", "fp16", "both", "idle")


def mean_sd(values):
    return float(np.mean(values)), (statistics.stdev(values) if len(values) > 1 else 0.0)


def kinds_coefficients():
    """Return {name: [value per run]} for the fit of attribution_kinds.py."""
    out = {name: [] for name in ("idle",) + TYPES + ("r2",)}
    for path in sorted(glob.glob("results/attribution_kinds_run*.txt")):
        text = open(path).read()
        out["r2"].append(float(re.search(rf"R2 {NUM}", text)[1]))
        for name in ("idle",) + TYPES:
            out[name].append(float(re.search(rf"\b{name} {NUM} W", text)[1]))
    return out


def streams_rows(variant):
    """Return {block: {field: [value per run]}} for attribution_streams.py."""
    fields = ("occ_fp32", "occ_fp16", "union", "counter", "counter_sd", "model", "error")
    rows = {}
    for path in sorted(glob.glob(f"results/attribution_streams_{variant}_run*.txt")):
        for line in open(path):
            m = re.match(rf"({'|'.join(BLOCKS)})\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM} \+/-\s*{NUM}\s+{NUM}\s+{NUM}", line)
            if m:
                for name, value in zip(fields, m.groups()[1:]):
                    rows.setdefault(m[1], {}).setdefault(name, []).append(float(value))
    return rows


def poll_rates():
    """Return [(read rate in Hz, power from the counter in W, power usage in W)]."""
    found = re.findall(rf"{NUM} Hz: counter\s+{NUM} W \(usage {NUM} W\)", open("results/counter_poll_rate.txt").read())
    return [tuple(float(v) for v in row) for row in found]


if __name__ == "__main__":
    k = kinds_coefficients()
    print(f"attribution_kinds.py, {len(k['r2'])} runs (power added per unit of occupancy, W)")
    for name in k:
        m, sd = mean_sd(k[name])
        print(f"  {name:8} {m:7.2f} +/- {sd:5.2f}  min {min(k[name]):6.2f}  max {max(k[name]):6.2f}")
    for variant in ("burst", "large"):
        rows = streams_rows(variant)
        print(f"\nattribution_streams.py ({variant}), {len(rows['fp32']['counter'])} runs")
        for block in BLOCKS:
            r = rows[block]
            error, error_sd = mean_sd(r["error"])
            print(f"  {block:4} occ fp32 {np.mean(r['occ_fp32']):.2f} occ fp16 {np.mean(r['occ_fp16']):.2f}"
                  f" union {np.mean(r['union']):.2f} counter {np.mean(r['counter']):6.1f} W"
                  f" model {np.mean(r['model']):6.1f} W error {error:+5.1f} +/- {error_sd:4.1f}")
    print("\ncounter_poll_rate.py")
    for hz, counter, usage in poll_rates():
        print(f"  {hz:6.1f} Hz {counter:6.1f} W (usage {usage:.0f} W)")
