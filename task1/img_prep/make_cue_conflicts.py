import numpy as np
import torch
import torch.nn.functional as F

# 5 unordered pairs. Deliberately cross-category (vehicle vs animal) where
# possible so that a "shape" answer and a "texture" answer are clearly distinct
# and unlikely to be confused with each other by accident.
CLASS_PAIRS = [
    ("airplane", "cat"),
    ("car", "deer"),
    ("ship", "dog"),
    ("truck", "bird"),
    ("horse", "monkey"),
]


def sobel_edges(img):
    """Edge/gradient magnitude map — a cheap proxy for 'the shape in this image'."""
    gray = img.mean(dim=1, keepdim=True)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=img.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return (gx ** 2 + gy ** 2).sqrt()


def rejection_rule(stylized, content, edge_corr_min=0.20, min_std=0.05, min_change=0.05):
    """VISUAL REJECTION RULE — 

    An image is ACCEPTED only if all three hold:
      A) structure survived  : edge-map correlation with the content image >= edge_corr_min
      B) not degenerate      : mean per-channel std >= min_std (rejects washed-out/collapsed output)
      C) style was applied   : mean abs difference from content >= min_change (rejects near-copies)
    Returns a boolean array, one entry per image.
    """
    e_s = sobel_edges(stylized).flatten(1)
    e_c = sobel_edges(content).flatten(1)
    e_s = e_s - e_s.mean(dim=1, keepdim=True)
    e_c = e_c - e_c.mean(dim=1, keepdim=True)
    corr = (e_s * e_c).sum(1) / (e_s.norm(dim=1) * e_c.norm(dim=1) + 1e-8)

    std = stylized.std(dim=(2, 3)).mean(dim=1)
    change = (stylized - content).abs().mean(dim=(1, 2, 3))

    keep = (corr >= edge_corr_min) & (std >= min_std) & (change >= min_change)
    return keep.cpu().numpy(), corr.cpu().numpy(), std.cpu().numpy(), change.cpu().numpy()


def build_conflict_plan(test_labels, class_names, n_per_direction=20, seed=6304):
    """Decide which image indices pair with which, for every (pair, direction).
    Returns a list of dicts: content index, style index, shape label, texture label.
    Fully seeded so the same plan regenerates every run."""
    rng = np.random.RandomState(seed)
    name_to_id = {c: i for i, c in enumerate(class_names)}
    by_class = {i: np.where(np.array(test_labels) == i)[0] for i in range(len(class_names))}

    plan = []
    for a_name, b_name in CLASS_PAIRS:
        a, b = name_to_id[a_name], name_to_id[b_name]
        for content_cls, style_cls in [(a, b), (b, a)]:        # both directions
            content_pool = rng.permutation(by_class[content_cls])[:n_per_direction]
            style_pool = rng.permutation(by_class[style_cls])[:n_per_direction]
            for ci, si in zip(content_pool, style_pool):
                plan.append({
                    "content_pos": int(ci),      # position within the 500-image subset
                    "style_pos": int(si),
                    "shape_label": int(content_cls),    # content supplies the shape
                    "texture_label": int(style_cls),     # style supplies the texture
                    "pair": f"{a_name}-{b_name}",
                    "direction": f"content={class_names[content_cls]},style={class_names[style_cls]}",
                })
    return plan