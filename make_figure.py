"""Draw figure.png from the outputs in results/."""
import re

import matplotlib
import numpy as np

import summarize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

# median update period of each source under load
text = open("results/probe_periods_load.txt").read()
period = dict(re.findall(r"^(usage|energy)\s+\d+\s+([\d.]+)", text, re.M))
values = [float(period["usage"]), float(period["energy"])]
ax[0].bar(["power usage /\ninstant / average", "energy counter"], values, color=["C3", "C0"])
for i, v in enumerate(values):
    ax[0].text(i, v + 10, f"{v:.0f} ms", ha="center")
ax[0].set_ylabel("median update period (ms)")
ax[0].set_title("NVML update periods under load")
ax[0].set_ylim(0, max(values) * 1.15)

# power derived from the energy counter against its read rate
rates = summarize.poll_rates()
usage = np.median([r[2] for r in rates])
ax[1].semilogx([r[0] for r in rates], [r[1] for r in rates], "o-", color="C0")
ax[1].axhline(usage, ls="--", color="gray")
ax[1].text(rates[0][0], usage + 8, f"power usage: {usage:.0f} W", color="gray")
ax[1].set_xlabel("energy counter read rate (Hz)")
ax[1].set_ylabel("power derived from the counter (W)")
ax[1].set_title("The counter under-counts when read fast")
ax[1].set_ylim(0, usage * 1.15)

# model against energy counter, two streams of small kernels, mean of the runs
rows = summarize.streams_rows("burst")
blocks = ["fp32", "fp16", "both", "idle"]
names = ["fp32 alone", "fp16 alone", "fp32 + fp16\n(2 streams)", "idle"]
x = np.arange(len(blocks))
ax[2].bar(x - 0.2, [np.mean(rows[b]["counter"]) for b in blocks], 0.4,
          yerr=[np.std(rows[b]["counter"]) for b in blocks], capsize=4, color="C0", label="energy counter")
ax[2].bar(x + 0.2, [np.mean(rows[b]["model"]) for b in blocks], 0.4, color="C1", label="CUPTI occupancy model")
ax[2].set_xticks(x)
ax[2].set_xticklabels(names)
ax[2].set_ylabel("mean board power over a 2 s block (W)")
ax[2].set_title(f"Two streams of small kernels ({len(rows['fp32']['counter'])} runs)")
ax[2].legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig("figure.png", dpi=150)
