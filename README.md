# Beyond the IID Assumption: Inductive Biases, Domain Shift, and Open-Set Recognition

Programming Assignment 1 — EE-5102 / CS-6304, Advanced Topics in Machine Learning
Fahan Hassan Khan (28100422), LUMS

This repository contains all code, configurations and per-step result files for the
four tasks of PA1:

| Task | Topic | Data |
|------|-------|------|
| 1 | Inductive biases of pretrained representations | STL-10 |
| 2 | Unsupervised domain adaptation | PACS (Photo, Art Painting, Cartoon → Sketch) |
| 3 | Domain generalization | PACS (Sketch unseen during training) |
| 4 | Open-set recognition | CIFAR-10 known, 16 fixed CIFAR-100 classes unknown |

**Seed 6304 everywhere. fp32 throughout.** Every number in the report is reproduced
from a JSON file under `taskN/results/` (see "Where the reported numbers live").

---

## 1. Repository layout

```
common/                     shared across tasks
  seed.py                   set_seed() — python, numpy, torch, cuda
  io_utils.py               save_results() / load_results() for the result JSONs
shared/                     shared by Tasks 2 and 3
  download_pacs.py          fetches PACS into datasets/
  pacs.py, pacs_protocol.py dataset wrappers and the fixed split protocol
  splits/                   frozen split index files (committed)
    pacs_sketch_seed6304.json
    cifar10_seed6304.json
taskN/
  img_prep/                 data preparation and interventions
  backbones/                model definitions
  methods/                  one file per training objective
  training/                 shared training loop
  evaluations/              metrics and analysis
  scores/                   Task 4 only: the unknownness scores
  notebooks/                one thin notebook per step (cells marked "# --- Cell N ---")
  results/                  one JSON per step (committed)
report/
  figures/                  every figure used in the PDF
  make_pacs_curves.py       regenerates the combined Task 2/3 curve figure
  make_cue_conflict_dis_compact.py
datasets/                   NOT committed (gitignored) — created by the prep notebooks
```

Modules hold the logic; notebooks only orchestrate, so every step can be re-run
by executing its notebook top to bottom. Checkpoints and cached tensors are
written to `datasets/cache/` and are gitignored (they run to several GB); the
result JSONs and figures they produce are committed.

---

## 2. Environment

Python 3.13, Windows 11, single NVIDIA RTX 4060 Laptop GPU (8 GB), CUDA 13.0.
All runs use fp32 (mixed precision overflowed in fp16 during Task 2) and
`num_workers=0`.

```bash
# PyTorch first, matched to your CUDA version:
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

`requirements.txt` lists only what the code imports. GPU is strongly recommended:
Task 4's three runs take about 3.4 hours in total on the hardware above.

---

## 3. Data

| Dataset | How it is obtained | Where it lands |
|---------|-------------------|----------------|
| STL-10 | downloaded automatically by `task1/notebooks/00_prep_data.ipynb` | `datasets/` |
| PACS | `python shared/download_pacs.py` | `datasets/PACS/` |
| CIFAR-10 / CIFAR-100 | downloaded automatically by `task4/notebooks/00_prep_data.ipynb` | `datasets/` |

`datasets/` is gitignored. The split **index files** are committed under
`shared/splits/`, so the exact same images are used on any machine:

- `pacs_sketch_seed6304.json` — 80/20 stratified split of each source domain
- `cifar10_seed6304.json` — 90/10 stratified split of the CIFAR-10 train partition
  (45,000 train / 5,000 validation, exactly 4,500/500 per class)

---

## 4. Reproducing each task

Run the notebooks of a task in numerical order, top to bottom. **Restart the
kernel between notebooks** — VS Code keeps one kernel alive per notebook file,
and a stale kernel holds GPU memory.

### Task 1 — Inductive biases (STL-10)

```
task1/notebooks/00_prep_data.ipynb        class-balanced 500-image subset, frozen ids
                01_clean_baseline.ipynb   linear heads on frozen backbones + CLIP zero-shot
                02_color_bias.ipynb       grayscale, hue rotation 180 degrees
                03_shape_texture.ipynb    AdaIN cue conflicts (alpha = 0.7), shape bias
                04_translation.ipynb      shifts of 8/16/32 px, four directions
                05_patch_structure.ipynb  4x4 patch shuffling
                06_representation_analysis.ipynb  cosine stability, t-SNE and UMAP
```

Backbones (ResNet-50, ViT-B/16 from torchvision; OpenCLIP ViT-B-32) stay frozen;
only the linear heads are trained, so runtimes are minutes per notebook.
Interventions are applied to a common 224x224 image *before* each model's own
normalization, so all models see the same pixels.

### Task 2 — Unsupervised domain adaptation (PACS)

```
task2/notebooks/00_prep_data.ipynb        splits, loaders, protocol checks
                01_source_only.ipynb      ERM baseline              (~6 min)
                02_dan.ipynb              MMD alignment             (~11 min)
                03_dann.ipynb             gradient reversal         (~11 min)
                04_cdan.ipynb             conditional adversarial   (~5 min)
                05_final_eval.ipynb       Sketch labels opened here, all checkpoints frozen
                06_lambda_study.ipynb     lambda_MMD in {0.1, 1, 10}
```

### Task 3 — Domain generalization (PACS, Sketch unseen)

Numbering starts at 02 because the ERM baseline **is** Task 2's source-only run
(`task2/notebooks/01_source_only.ipynb`); it is reused rather than retrained, so
Task 3's comparison shares its protocol exactly.

```
task3/notebooks/02_dan_dg.ipynb           MMD between source-domain pairs  (~3 min)
                03_sam.ipynb              sharpness-aware minimization     (~12 min)
                04_final_eval.ipynb       Sketch labels opened here
                05_lambda_study.ipynb     lambda_DG in {0.1, 1, 10}
```

### Task 4 — Open-set recognition (CIFAR-10 / CIFAR-100)

```
task4/notebooks/00_prep_data.ipynb        splits, unknown groups, model checks
                01_vanilla.ipynb          CIFAR ResNet-18, 100 epochs      (~73 min)
                02_gcsc.ipynb             same recipe + RandAugment        (~95 min)
                03_proser.ipynb           PROSER fine-tune, 50 epochs      (~33 min)
                04_scores.ipynb           four scores, thresholds frozen   (seconds)
                05_osr_eval.ipynb         CIFAR-100 loaded here, first time (~3 min)
```

Notebook 03 loads the Vanilla checkpoint, so Task 4's runs must be executed in
order. Notebooks 04 and 05 read cached logits and features and need no training.

---

## 5. Seeds and determinism

`common/seed.py` seeds Python, NumPy, `torch` and `torch.cuda`. Data loaders take
their own explicitly seeded `torch.Generator`, so shuffling order does not depend
on how many random numbers earlier cells consumed — each training cell re-seeds
and rebuilds its loaders, and is therefore reproducible independently of the rest
of the notebook.

`torch.backends.cudnn.deterministic` is **not** set. cuDNN may pick
non-deterministic convolution algorithms, so a re-run can differ in the last
digits and, over thousands of steps, can diverge slightly from the committed
numbers. The results JSONs record what the committed runs actually produced.

---

## 6. Design choices, deviations and protocol notes

Choices the assignment left open, and the two places where the specification was
deviated from. All are stated in the report; they are repeated here so the code
can be read without it.

**Tasks 2 and 3**

- *Deviation:* DANN's discriminator input is **L2-normalized**. Without it the run
  diverges after step 125 (feature norm peaks at 23,131, domain loss at 2,534);
  with it, norms stay in 20–76 and domain loss in 0.60–0.85. Both runs are recorded
  in `task2/results/dann_normalisation_evidence.json`. The decision used training
  losses only, no target labels. CDAN's discriminator is left unnormalized as
  specified, so the two methods differ in more than conditioning.
- BatchNorm statistics are frozen in the shared backbone (ImageNet-pretrained),
  including inside both of SAM's forward passes.
- DAN-DG applies the MMD penalty to each of the three source-domain pairs, with
  bandwidths taken from the median pairwise squared distance of that pair's batch.
- Checkpoints are selected by mean source-validation macro-F1 only.

**Task 4**

- Mahalanobis uses **one tied within-class diagonal covariance** — each feature
  minus its own class mean, pooled over classes — fitted on the 45,000
  **unaugmented training-split** features only (the validation split is reserved
  for thresholds), with 1e-6 added to every diagonal entry.
- MSP is computed as `s / (1 + s)` with `s` the sum of the non-top softmax terms,
  in float64. The textbook `1 - max p` in fp32 rounds confident images onto a
  coarse grid and leaves 27–56% of test images sharing a value.
- PROSER's placeholder score is stored as `log p_dummy` (monotone in `p_dummy`,
  which underflows for confident knowns); thresholds are reported in both forms.
- PROSER uses one lambda per batch drawn from Beta(2,2), partners from a random
  permutation of the second half-batch, with same-class pairs dropped.
- *Deviation:* mixed (blended) features do **not** update BatchNorm's running
  statistics. Blends have roughly half the variance of real features, and in the
  first run their inclusion pushed eval-time normalization off: validation
  accuracy fell from 94.6% to about 90% as placeholders were learned, and the
  best-validation rule then selected epoch 3, before PROSER had trained. With the
  exclusion, accuracy stays near 95% for all 50 epochs. The first run is kept as
  `task4/results/proser_run1_bn_mixed.json`; setting `mix_updates_bn_stats=True`
  reproduces it. The decision used CIFAR-10 validation accuracy only.
- Thresholds are the 95th percentile of unknownness on the CIFAR-10 validation
  split; an image is accepted when `u(x) <= tau`. FPR@95TPR is reported as the
  fraction of unknowns accepted at that validation-calibrated threshold, not at a
  threshold re-tuned on test data.
- Training uses `drop_last=True` (72 of 45,000 images dropped per epoch, a
  different 72 each epoch) so that steps per epoch are constant and every batch is
  even, which PROSER's half-batch split requires. Cosine decay steps once per
  epoch; the final epoch runs at lr 2.5e-5.
- GCSC (`task4/methods/gcsc.py`) deliberately inherits `compute_loss` from
  `Vanilla` unchanged: its objective *is* vanilla cross-entropy, and the only
  difference in the whole pipeline is RandAugment(2, 9) in the training transform.
  Vanilla and GCSC configs differ in exactly one key.

---

## 7. Evaluation integrity

- **Tasks 2 and 3.** No Sketch label is read during training, validation or
  checkpoint selection. Target labels are opened only in `05_final_eval.ipynb`
  (Task 2) and `04_final_eval.ipynb` (Task 3), after every checkpoint is frozen.
  Task 2 uses unlabelled Sketch images during training, as unsupervised adaptation
  requires; Task 3 never loads Sketch at all until evaluation.
- **Task 4.** No unknown image passes through any model before
  `05_osr_eval.ipynb`. `build_unknown_loaders()` is referenced in exactly two
  notebooks: `00_prep_data.ipynb`, which reads CIFAR-100 **labels** to verify that
  each group holds 800 images at 100 per class, and `05_osr_eval.ipynb`, the
  evaluation. All three trainings, all four score definitions and all thresholds
  (notebooks 01–04) never reference CIFAR-100. `05_osr_eval.ipynb` reads the
  thresholds from `task4/results/scores.json` and never recomputes them; its first
  cell verifies that the reloaded checkpoints reproduce the cached known-side
  logits and features exactly (max absolute difference 0.00e+00 for all three
  models).
- Unknown images go through the same evaluation transform and the same **CIFAR-10**
  normalization constants as the knowns; using CIFAR-100 statistics would push
  information about the unknown distribution into preprocessing.

---

## 8. Where the reported numbers live

| Report element | File |
|---|---|
| Task 1 main table | `task1/results/{clean_baseline,color_bias,patch_structure,translation}.json` |
| Cue-conflict table | `task1/results/shape_texture.json` (+ `cue_conflict_rejections.json`) |
| Cosine stability | `task1/results/representation_analysis.json` |
| PACS main table (Task 2 columns) | `task2/results/final_eval.json` |
| PACS main table (Task 3 columns) | `task3/results/final_eval.json` |
| Per-class Sketch table, prediction counts | `task2/results/final_eval.json` → `per_class`, `predicted_distribution`, `confusions` |
| Alignment-strength table | `task2/results/lambda_study.json`, `task3/results/lambda_study.json` |
| DANN normalization evidence | `task2/results/dann_normalisation_evidence.json` |
| Task 4 open-set table, per-class acceptance, absorption, failure cases | `task4/results/osr_eval.json` |
| Task 4 thresholds, score definitions, Mahalanobis fit | `task4/results/scores.json` |
| Task 4 CSA and training histories | `task4/results/{vanilla,gcsc,proser}.json` |
| PROSER BatchNorm counterfactual | `task4/results/proser_run1_bn_mixed.json` |

Every figure in the report is committed under `report/figures/` and is regenerated
by the notebook (or the `report/make_*.py` script) that produced it.

---

## 9. Attribution

Methods implemented from their papers, in this repository, from scratch:

- DAN — Long et al., *Learning Transferable Features with Deep Adaptation Networks*, ICML 2015
- DANN — Ganin et al., *Domain-Adversarial Training of Neural Networks*, JMLR 2016
- CDAN — Long et al., *Conditional Adversarial Domain Adaptation*, NeurIPS 2018
- SAM — Foret et al., *Sharpness-Aware Minimization*, ICLR 2021
- MSP — Hendrycks and Gimpel, ICLR 2017; MLS — Vaze et al., ICLR 2022; Energy — Liu et al., NeurIPS 2020; Mahalanobis — Lee et al., NeurIPS 2018
- PROSER — Zhou et al., *Learning Placeholders for Open-Set Recognition*, CVPR 2021.
  Where the public reference implementation differs from the paper's equations,
  the paper is followed, as the assignment specifies.

Third-party code and weights used as libraries, not copied:

- `torchvision` — ResNet-50 and ViT-B/16 architectures and ImageNet weights; the
  ResNet-18 used in Tasks 2–4 (Task 4 modifies its stem for 32x32 inputs);
  RandAugment; dataset loaders for STL-10, CIFAR-10 and CIFAR-100
- `open_clip` — OpenCLIP ViT-B-32 and its pretrained weights
- `scikit-learn` — logistic-regression probes, t-SNE, AUROC and F1; `umap-learn` — UMAP
- AdaIN (Huang and Belongie, ICCV 2017): the VGG-19 encoder and decoder definitions in
  `task1/img_prep/adain.py` follow the public pytorch-AdaIN implementation
  (https://github.com/naoto0804/pytorch-AdaIN), and we load its pretrained
  `vgg_normalised.pth` and `decoder.pth`. Download both from that repository and pass
  their paths to `AdaINStyleTransfer`. The cue-conflict protocol follows Geirhos et al., ICLR 2019.

Datasets: STL-10 (Coates et al., 2011), PACS (Li et al., 2017), CIFAR-10 and
CIFAR-100 (Krizhevsky, 2009).

---

## 10. Use of AI assistance

An AI assistant was used for writing and debugging code, for explanation of the
methods, and for review of the experimental design. All experiments were run by
the author, and the report was written entirely by the author.
