"""
Domain separability: how much source/target information survives in a frozen
representation. Independent of every training objective, which is why it can
disagree with MMD or with the DANN discriminator.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .metrics import evaluate_loader


@torch.no_grad()
def collect_features(model, loader, device):
    out = evaluate_loader(model, loader, device, collect_features=True)
    return out['features'], out['labels']


def domain_separability(src_feats, tgt_feats, seed=6304, test_size=0.30, C=1.0):
    
    
    rng = np.random.default_rng(seed)
    n = min(len(src_feats), len(tgt_feats))
    src = src_feats[rng.permutation(len(src_feats))[:n]]
    tgt = tgt_feats[rng.permutation(len(tgt_feats))[:n]]

    X = np.concatenate([src, tgt])
    y = np.concatenate([np.zeros(n), np.ones(n)])          # 0 = source, 1 = target

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    clf = LogisticRegression(C=C, class_weight='balanced',
                             max_iter=2000, random_state=seed)
    clf.fit(Xtr, ytr)
    return {'separability': float(clf.score(Xte, yte)),
            'n_per_domain': int(n), 'n_test': int(len(yte))}