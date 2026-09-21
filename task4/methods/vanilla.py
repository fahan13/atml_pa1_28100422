import torch.nn as nn
import torch.nn.functional as F


class Vanilla(nn.Module):
    
    def __init__(self, cfg=None):
        super().__init__()

    def compute_loss(self, model, x, y):
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        return loss, {'loss_cls': loss.item(),
                      'train_acc': (logits.argmax(1) == y).float().mean().item()}