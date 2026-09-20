"""
Method 1 of 4: Source-only ERM.
Cross-entropy on the three labelled source domains. No target image is ever
loaded (needs_target = False). This is the measuring stick for every later
method, and is reused unchanged as the Task 3 ERM baseline.
"""
import torch
import torch.nn.functional as F


class SourceOnly:
    name = 'source_only'
    needs_target = False

    def __init__(self, cfg=None):
        self.cfg = cfg or {}

    def modules(self):
        return []          # no extra trainable modules beyond backbone + head

    def parameters(self):
        return []

    def compute_loss(self, model, src_x, src_y, tgt_x, p):
        logits, _feat = model(src_x)
        loss = F.cross_entropy(logits, src_y)
        with torch.no_grad():
            acc = (logits.argmax(1) == src_y).float().mean().item()
        return loss, {'loss_total': loss.item(), 'loss_cls': loss.item(), 'train_acc': acc}