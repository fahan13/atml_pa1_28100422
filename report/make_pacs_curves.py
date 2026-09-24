"""
One-row training-curve figure for Tasks 2 and 3 (report Figure 4).
Panels: T2 classification loss | T2 alignment/domain loss | T2 discriminator
accuracy | T3 classification loss | T3 DAN-DG pairwise MMD^2.
Reads only committed result JSONs. Run from the repo root:
    python report/make_pacs_curves.py
Writes report/figures/pacs_training_curves.png
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures" / "pacs_training_curves.png"


def hist(path):
    with open(ROOT / path) as f:
        return json.load(f)["history"]


T2 = {"Source-only": ("task2/results/source_only.json", "k"),
      "DAN": ("task2/results/dan.json", "tab:blue"),
      "DANN": ("task2/results/dann.json", "tab:orange"),
      "CDAN": ("task2/results/cdan.json", "tab:red")}
T3 = {"ERM": ("task2/results/source_only.json", "k"),
      "DAN-DG": ("task3/results/dan_dg.json", "tab:purple"),
      "SAM": ("task3/results/sam.json", "tab:green")}
LN7 = math.log(7)
ep = lambda h: [e["epoch"] for e in h]

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.3,
                     "xtick.labelsize": 7, "ytick.labelsize": 7})
fig, ax = plt.subplots(1, 5, figsize=(13, 2.55))

for lab, (p, c) in T2.items():
    h = hist(p)
    ax[0].plot(ep(h), [e["loss_cls"] for e in h], "-o", ms=2.3, c=c, label=lab)
    if lab == "DAN":
        ax[1].plot(ep(h), [e["loss_mmd"] for e in h], "-o", ms=2.3, c=c, label="DAN: MMD$^2$")
    if lab in ("DANN", "CDAN"):
        ax[1].plot(ep(h), [e["loss_dom"] for e in h], "-s", ms=2.3, c=c, label=f"{lab}: domain CE")
        ax[2].plot(ep(h), [e["domain_acc"] for e in h], "-s", ms=2.3, c=c, label=lab)
for lab, (p, c) in T3.items():
    h = hist(p)
    ax[3].plot(ep(h), [e["loss_cls"] for e in h], "-o", ms=2.3, c=c, label=lab)
    if lab == "DAN-DG":
        for k, name in (("mmd_pa", "P-A"), ("mmd_pc", "P-C"), ("mmd_ac", "A-C")):
            ax[4].plot(ep(h), [e[k] for e in h], "-o", ms=2.3, label=name)

for a in (ax[0], ax[1], ax[3]):
    a.set_yscale("log")
ax[0].axhline(LN7, ls=":", c="grey", lw=0.8); ax[3].axhline(LN7, ls=":", c="grey", lw=0.8)
ax[1].axhline(math.log(2), ls=":", c="grey", lw=0.8)
ax[2].axhline(0.5, ls=":", c="grey", lw=0.8); ax[2].set_ylim(0.25, 0.85)
ax[0].set_title("T2: classification loss"); ax[1].set_title("T2: alignment / domain loss")
ax[2].set_title("T2: discriminator accuracy"); ax[3].set_title("T3: classification loss")
ax[4].set_title("T3: DAN-DG pairwise MMD$^2$")
lo, hi = ax[4].get_ylim(); ax[4].set_ylim(lo, hi + 0.35 * (hi - lo))
for i, a in enumerate(ax):
    a.set_xlabel("epoch"); a.grid(alpha=0.25, lw=0.5)
    a.xaxis.set_major_locator(MaxNLocator(integer=True))
    a.legend(frameon=False, loc="upper center" if i == 4 else "best", ncol=3 if i == 4 else 1)
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)