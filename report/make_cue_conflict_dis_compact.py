"""
Compact, print-legible version of the Task 1 cue-conflict disagreement figure.

The original report/figures/cue_conflict_disagreement.png (written by
task1/notebooks/03_shape_texture.ipynb) has 8-pt labels on a 12.6-inch-wide
canvas, so they become unreadable once the figure is shrunk to fit beside
Table 2. This script reuses the SAME three stylized images (cropped from that
PNG) and the SAME predictions (task1/results/cue_conflict_examples.json), and
redraws the labels at a size meant for ~3.3 inches of print width.

No model is run. From the repo root:
    python report/make_cue_conflict_panel.py
Writes report/figures/cue_conflict_disagreement_compact.png
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "report" / "figures" / "cue_conflict_disagreement.png"
EX = ROOT / "task1" / "results" / "cue_conflict_examples.json"
OUT = ROOT / "report" / "figures" / "cue_conflict_disagreement_compact.png"

img = np.asarray(Image.open(SRC).convert("RGB"))
nonwhite = img.min(axis=2) < 235
rows = np.flatnonzero(nonwhite.sum(1) > 0.5 * img.shape[1])      # rows crossing all 3 images
r0, r1 = rows.min(), rows.max() + 1
on = nonwhite[r0:r1].sum(0) > 0.5 * (r1 - r0)
edges = np.flatnonzero(np.diff(on.astype(int))) + 1                # [start, end, start, end, ...]
boxes = list(zip(edges[0::2], edges[1::2]))
assert len(boxes) == 3, f"expected 3 images, found {len(boxes)}"

examples = json.load(open(EX))["examples"]["disagreement"]
NAMES = {"resnet50": "ResNet", "vit_b16": "ViT", "clip_head": "CLIP head", "clip_zeroshot": "CLIP z.-s."}
TAG = {"shape": "S", "texture": "T", "other": "O"}

plt.rcParams.update({"font.size": 6.5, "font.family": "monospace"})
fig, axes = plt.subplots(1, 3, figsize=(3.4, 1.75))
for ax, (c0, c1), ex in zip(axes, boxes, examples):
    ax.imshow(img[r0:r1, c0:c1])
    ax.set_xticks([]); ax.set_yticks([])
    lines = [f"{NAMES[m]:<10}{p['predicted']:>9} {TAG[p['verdict']]}" for m, p in ex["predictions"].items()]
    ax.set_xlabel("\n".join(lines), fontsize=5.6, loc="left", labelpad=2)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.30, wspace=0.08)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=400, bbox_inches="tight", pad_inches=0.01)
print("saved", OUT)