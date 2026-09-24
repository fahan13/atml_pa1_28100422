"""
One compact training-curve figure for Tasks 2 and 3 (replaces the seven
separate task2_*_curves.png / task3_*_curves.png files in the report).

Reads ONLY the committed result JSONs, so every plotted value traces to a
saved file. Run from the repo root:

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
        d = json.load(f)
    return d["history"], d["best_epoch"]


T2 = {  # label: (json, colour)
    "Source-only": ("task2/results/source_only.json", "k"),
    "DAN": ("task2/results/dan.json", "tab:blue"),
    "DANN": ("task2/results/dann.json", "tab:orange"),
    "CDAN": ("task2/results/cdan.json", "tab:red"),
}
T3 = {
    "ERM": ("task2/results/source_only.json", "k"),   # reused unchanged
    "DAN-DG": ("task3/results/dan_dg.json", "tab:purple"),
    "SAM": ("task3/results/sam.json", "tab:green"),
}
LN7 = math.log(7)   # cross-entropy of a uniform guess over 7 classes

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})
fig, ax = plt.subplots(2, 4, figsize=(10, 4.6))


def ep(h):
    return [e["epoch"] for e in h]


def mark_best(a, h, best, key, c):
    y = [e[key] for e in h if e["epoch"] == best]
    if y:
        a.scatter([best], y, s=28, facecolors="none", edgecolors=c, linewidths=1.2, zorder=5)


# ---------------- row 1: Task 2 (UDA) ----------------
for lab, (p, c) in T2.items():
    h, best = hist(p)
    ax[0, 0].plot(ep(h), [e["loss_cls"] for e in h], "-o", ms=2.5, c=c, label=lab)
    ax[0, 3].plot(ep(h), [e["mean_macro_f1"] for e in h], "-o", ms=2.5, c=c, label=lab)
    mark_best(ax[0, 3], h, best, "mean_macro_f1", c)
    if lab == "DAN":
        ax[0, 1].plot(ep(h), [e["loss_mmd"] for e in h], "-o", ms=2.5, c=c, label="DAN: MMD$^2$")
    if lab in ("DANN", "CDAN"):
        ax[0, 1].plot(ep(h), [e["loss_dom"] for e in h], "-s", ms=2.5, c=c, label=f"{lab}: domain CE")
        ax[0, 2].plot(ep(h), [e["domain_acc"] for e in h], "-s", ms=2.5, c=c, label=lab)

ax[0, 0].axhline(LN7, ls=":", c="grey", lw=0.8)
ax[0, 0].set_yscale("log")
ax[0, 0].set_title("T2: source classification loss")
ax[0, 1].axhline(math.log(2), ls=":", c="grey", lw=0.8)
ax[0, 1].set_yscale("log")
ax[0, 1].set_title("T2: alignment / domain loss")
ax[0, 2].axhline(0.5, ls=":", c="grey", lw=0.8)
ax[0, 2].set_ylim(0.25, 0.85)
ax[0, 2].set_title("T2: online discriminator acc.")
ax[0, 3].set_title("T2: mean source-val macro-F1")

# ---------------- row 2: Task 3 (DG) ----------------
for lab, (p, c) in T3.items():
    h, best = hist(p)
    ax[1, 0].plot(ep(h), [e["loss_cls"] for e in h], "-o", ms=2.5, c=c, label=lab)
    ax[1, 3].plot(ep(h), [e["mean_macro_f1"] for e in h], "-o", ms=2.5, c=c, label=lab)
    mark_best(ax[1, 3], h, best, "mean_macro_f1", c)
    if lab == "DAN-DG":
        for k, name in (("mmd_pa", "P-A"), ("mmd_pc", "P-C"), ("mmd_ac", "A-C")):
            ax[1, 1].plot(ep(h), [e[k] for e in h], "-o", ms=2.5, label=name)
    if lab == "SAM":
        ax[1, 2].plot(ep(h), [e["loss_gap"] for e in h], "-o", ms=2.5, c=c,
                      label="SAM: $L(\\theta+\\epsilon)-L(\\theta)$")

ax[1, 0].axhline(LN7, ls=":", c="grey", lw=0.8)
ax[1, 0].set_yscale("log")
ax[1, 0].set_title("T3: classification loss")
ax[1, 1].set_title("T3: DAN-DG pairwise MMD$^2$ ($\\lambda_{DG}=1$)")
ax[1, 2].set_title("T3: SAM perturbation loss gap")
ax[1, 3].set_title("T3: mean source-val macro-F1")

for a in ax.flat:
    a.set_xlabel("epoch")
    a.xaxis.set_major_locator(MaxNLocator(integer=True))   # whole epochs only
    a.grid(alpha=0.25, lw=0.5)
    a.legend(frameon=False, loc="best")

# DAN-DG panel: headroom + one-row legend so it cannot sit on the lines
lo, hi = ax[1, 1].get_ylim()
ax[1, 1].set_ylim(lo, hi + 0.35 * (hi - lo))
ax[1, 1].legend(frameon=False, loc="upper center", ncol=3)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)