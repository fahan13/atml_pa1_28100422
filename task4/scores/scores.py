import numpy as np
from scipy.special import logsumexp


def _f64(a):
    return np.asarray(a, dtype=np.float64)


def u_msp(logits):
  
    z = _f64(logits)
    top = z.argmax(1)
    e = np.exp(z - z.max(1, keepdims=True))
    e[np.arange(len(z)), top] = 0.0
    s = e.sum(1)
    return s / (1.0 + s)


def u_msp_naive_fp32(logits):
    z = np.asarray(logits, dtype=np.float32)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return (np.float32(1.0) - p.max(1)).astype(np.float64)


def u_mls(logits):
    return -_f64(logits).max(1)


def u_energy(logits):
    return -logsumexp(_f64(logits), axis=1)


class Mahalanobis:
    
    def __init__(self, eps=1e-6):
        self.eps = eps

    def fit(self, feats, labels, num_classes=10):
        F = _f64(feats)
        y = np.asarray(labels)
        self.means = np.stack([F[y == c].mean(0) for c in range(num_classes)])
        R = F - self.means[y]
        self.var = (R ** 2).mean(0) + self.eps
        return self

    def distances(self, feats):
        F = _f64(feats)
        return np.stack([(((F - m) ** 2) / self.var).sum(1) for m in self.means], axis=1)

    def score(self, feats):
        return self.distances(feats).min(1)


def u_placeholder_logp(logits, dummy_logits):
    
    z = _f64(logits)
    d = _f64(dummy_logits).max(1, keepdims=True)
    ext = np.concatenate([z, d], axis=1)
    return d[:, 0] - logsumexp(ext, axis=1)