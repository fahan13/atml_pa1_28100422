import numpy as np
import torch
from sklearn.metrics import f1_score


@torch.no_grad()
def extract_outputs(model, loader, device, keep_features=True):
  
    model.eval()
    L, Fe, Y = [], [], []
    for x, y in loader:
        logits, f = model(x.to(device, non_blocking=True))
        L.append(logits.float().cpu().numpy())
        if keep_features:
            Fe.append(f.float().cpu().numpy())
        Y.append(y.numpy())
    out = {'logits': np.concatenate(L), 'labels': np.concatenate(Y)}
    if keep_features:
        out['features'] = np.concatenate(Fe)
    return out


def accuracy(logits, labels):
    return float((logits.argmax(1) == labels).mean())


def macro_f1(logits, labels):
    return float(f1_score(labels, logits.argmax(1), average='macro', zero_division=0))


def per_class_accuracy(logits, labels, num_classes=10):
    p = logits.argmax(1)
    return np.array([float((p[labels == c] == c).mean()) if (labels == c).any() else np.nan
                     for c in range(num_classes)])