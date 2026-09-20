"""Per-class target analysis: which classes adaptation helped, which it broke."""
import numpy as np

from ..evaluations.metrics import confusion, per_class_accuracy


def per_class_table(preds, labels, classes, num_classes=7):
    acc = per_class_accuracy(preds, labels, num_classes)
    support = np.bincount(labels, minlength=num_classes)
    return {c: {'accuracy': float(acc[i]), 'support': int(support[i])}
            for i, c in enumerate(classes)}


def class_deltas(base_table, method_table, classes):
    """Change relative to Source-only, with support carried along: ."""
    return {c: {'delta': method_table[c]['accuracy'] - base_table[c]['accuracy'],
                'base': base_table[c]['accuracy'],
                'method': method_table[c]['accuracy'],
                'support': base_table[c]['support']} for c in classes}


def dominant_confusions(preds, labels, classes, top_k=3, num_classes=7):
    """For each true class, the wrong labels it most often receives."""
    cm = confusion(preds, labels, num_classes)
    out = {}
    for i, c in enumerate(classes):
        row = cm[i].copy()
        row[i] = 0                                   # ignore correct predictions
        order = np.argsort(row)[::-1][:top_k]
        out[c] = [{'predicted': classes[j], 'count': int(row[j])}
                  for j in order if row[j] > 0]
    return out


def predicted_distribution(preds, classes, num_classes=7):
    """How often each label is predicted overall. """
    counts = np.bincount(preds, minlength=num_classes)
    return {c: int(counts[i]) for i, c in enumerate(classes)}