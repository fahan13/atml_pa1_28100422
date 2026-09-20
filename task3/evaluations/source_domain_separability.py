"""
How much SOURCE-domain information survives in a representation.

"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from task2.evaluations.metrics import evaluate_loader


def collect_source_features(model, source_val, device, seed=6304):
    """Balanced features from the three source validation sets: equal numbers
    per domain, so a probe cannot score well by favouring the largest."""
    rng = np.random.default_rng(seed)
    feats, doms = [], []
    per_domain = {}
    for di, (d, ld) in enumerate(source_val.items()):
        out = evaluate_loader(model, ld, device, collect_features=True)
        per_domain[d] = out['features']
    n = min(len(f) for f in per_domain.values())          # balance to the smallest
    for di, (d, f) in enumerate(per_domain.items()):
        idx = rng.permutation(len(f))[:n]
        feats.append(f[idx])
        doms.append(np.full(n, di))
    return np.concatenate(feats), np.concatenate(doms), n


def source_domain_separability(feats, doms, seed=6304, test_size=0.30, C=1.0):
    Xtr, Xte, ytr, yte = train_test_split(
        feats, doms, test_size=test_size, random_state=seed, stratify=doms)
    clf = LogisticRegression(C=C, max_iter=3000, random_state=seed)
    clf.fit(Xtr, ytr)
    return {'separability': float(clf.score(Xte, yte)),
            'n_per_domain': int(len(doms) // 3), 'chance': 1 / 3}