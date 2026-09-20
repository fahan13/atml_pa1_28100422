
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score


@torch.no_grad()
def evaluate_loader(model, loader, device, collect_features=False):
    """
    accuracy = fraction correct, dominated by frequent classes.
    macro-F1 = per-class F1 averaged with EQUAL weight, so a rare class counts
               as much as a common one. 
    """
    model.eval()                       # full eval mode here (BN already frozen anyway)
    preds, labels, feats = [], [], []
    for x, y in loader:
        logits, f = model(x.to(device, non_blocking=True))
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.numpy())
        if collect_features:
            feats.append(f.cpu().numpy())
    p, t = np.concatenate(preds), np.concatenate(labels)
    out = {'accuracy': float((p == t).mean()),
           'macro_f1': float(f1_score(t, p, average='macro', zero_division=0)),
           'n': int(len(t)), 'preds': p, 'labels': t}
    if collect_features:
        out['features'] = np.concatenate(feats)
    return out


def evaluate_source_val(model, source_val, device):
    """Per-domain validation metrics plus the mean macro-F1 used for checkpointing."""
    per_domain = {d: evaluate_loader(model, ld, device) for d, ld in source_val.items()}
    return {
        'per_domain': {d: {'accuracy': m['accuracy'], 'macro_f1': m['macro_f1'], 'n': m['n']}
                       for d, m in per_domain.items()},
        'mean_macro_f1': float(np.mean([m['macro_f1'] for m in per_domain.values()])),
        'mean_accuracy': float(np.mean([m['accuracy'] for m in per_domain.values()])),
    }


def per_class_accuracy(preds, labels, num_classes=7):
    return np.array([float((preds[labels == c] == c).mean()) if (labels == c).any()
                     else np.nan for c in range(num_classes)])


def confusion(preds, labels, num_classes=7):
    return confusion_matrix(labels, preds, labels=list(range(num_classes)))