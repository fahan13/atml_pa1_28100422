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