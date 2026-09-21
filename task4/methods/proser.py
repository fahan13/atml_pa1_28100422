from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


@contextmanager
def bn_stats_frozen(*blocks):
    """
    Inside this context, BatchNorm layers in `blocks` still normalise with the
    current batch's statistics (train-mode behaviour, gradients unchanged) but
    do not update their running averages: PyTorch's update is
    running = (1 - momentum) * running + momentum * batch, so momentum = 0
    makes it a no-op. Original momentum is restored on exit.
    """
    bns = [m for b in blocks for m in b.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, mo in zip(bns, saved):
            m.momentum = mo


def mix_different_class_pairs(h, y, lam):
    """Blend each example with a random partner; keep only different-class pairs."""
    perm = torch.randperm(h.size(0), device=h.device)
    keep = y != y[perm]
    h_mix = lam * h[keep] + (1.0 - lam) * h[perm][keep]
    return h_mix, keep, perm


class PROSER(nn.Module):
    def __init__(self, cfg, feature_dim=512):
        super().__init__()
        self.K = cfg['num_classes']                                   # 10
        self.dummy = nn.Linear(feature_dim, cfg['num_dummy'])         # 5 dummy classifiers
        self.beta = cfg['beta']
        self.gamma = cfg['gamma']
        self.mix_updates_bn = cfg.get('mix_updates_bn_stats', True)   # True reproduces run 1
        a = float(cfg['mix_alpha'])
        self.mix_dist = torch.distributions.Beta(torch.tensor(a), torch.tensor(a))

    def extended_logits(self, model, f):
        """[10 known logits | max of 5 dummy logits]  ->  [N, 11]"""
        z = model.head(f)
        d = self.dummy(f).max(dim=1, keepdim=True).values
        return torch.cat([z, d], dim=1)

    def compute_loss(self, model, x, y):
        K = self.K
        half = x.size(0) // 2
        xa, ya = x[:half], y[:half]          # classifier placeholders
        xb, yb = x[half:], y[half:]          # data placeholders

        # --- classifier placeholders -----------------------------------
        _, fa = model(xa)
        ext_a = self.extended_logits(model, fa)
        loss_ce = F.cross_entropy(ext_a, ya)                          # true class beats dummy
        true_mask = F.one_hot(ya, K + 1).bool()
        masked = ext_a.masked_fill(true_mask, -1e9)                   # delete the true slot
        loss_cp = F.cross_entropy(masked, torch.full_like(ya, K))     # dummy beats the rest

        # --- data placeholders (manifold mixup after layer2) ------------
        lam = float(self.mix_dist.sample())
        hb = model.forward_pre(xb)
        h_mix, keep, _ = mix_different_class_pairs(hb, yb, lam)
        n_mix = int(keep.sum())
        if n_mix > 0:
            if self.mix_updates_bn:
                f_mix = model.forward_post(h_mix)
            else:
                with bn_stats_frozen(model.net.layer3, model.net.layer4):
                    f_mix = model.forward_post(h_mix)
            ext_m = self.extended_logits(model, f_mix)
            loss_dp = F.cross_entropy(ext_m, torch.full((n_mix,), K, device=x.device,
                                                        dtype=torch.long))
            mix_to_dummy = (ext_m.argmax(1) == K).float().mean().item()
        else:
            loss_dp = x.new_zeros(())
            mix_to_dummy = 0.0

        loss = loss_ce + self.beta * loss_cp + self.gamma * loss_dp
        with torch.no_grad():
            logs = {
                'loss': loss.item(),
                'loss_ce': loss_ce.item(),
                'loss_cp': loss_cp.item(),
                'loss_dp': loss_dp.item(),
                'train_acc': (ext_a[:, :K].argmax(1) == ya).float().mean().item(),
                'dummy_runner_up': (masked.argmax(1) == K).float().mean().item(),
                'dummy_wins_known': (ext_a.argmax(1) == K).float().mean().item(),
                'mix_to_dummy': mix_to_dummy,
                'n_mix': float(n_mix),
                'lam': lam,
            }
        return loss, logs