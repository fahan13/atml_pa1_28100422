import numpy as np

def cosine_stability(feats_clean, feats_transformed):
    """average cosine similarity between each image's
    clean feature vector and its transformed counterpart.

    1.0 = representation unchanged, 0.0 = perpendicular/unrelated.
    """
    a = np.asarray(feats_clean, dtype=np.float64)
    b = np.asarray(feats_transformed, dtype=np.float64)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    per_image = num / den
    return float(per_image.mean()), per_image