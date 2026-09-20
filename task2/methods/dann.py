"""
Method 3 of 4: DANN — adversarial alignment.

L_DANN = L_cls(source only) + L_domain(source + target)

A discriminator (512 -> 256 -> ReLU -> dropout 0.5 -> 2) predicts which domain a
feature came from. A gradient-reversal layer sits between the backbone and the
discriminator: identity forwards, multiply by -alpha backwards. So ONE
backward pass trains the discriminator to detect the domain and the backbone to
hide it.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversal(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha):
    return GradientReversal.apply(x, alpha)


def dann_alpha(p):
    """The manual's schedule: 0 at p=0, ~0.76 at p=0.5, ->1 at p=1."""
    return 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))).item() - 1.0


class DomainDiscriminator(nn.Module):
    """512 -> 256 -> ReLU -> dropout(0.5) -> 2. Same architecture is reused
    by CDAN in Step 4, with a wider input."""

    def __init__(self, in_dim=512, hidden=256, n_domains=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, x):
        return self.net(x)


class DANN:
    name = 'dann'
    needs_target = True

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.disc = DomainDiscriminator(in_dim=512, hidden=cfg.get('disc_hidden', 256))
        self.alpha_max = cfg.get('alpha_max', 1.0)   # 1.0 for the main run
        self.lam_domain = cfg.get('lambda_domain', 1.0)
        self.normalize_input = cfg.get('normalize_disc_input', False)

    def modules(self):
        return [self.disc]

    def parameters(self):
        return list(self.disc.parameters())

    def to(self, device):
        self.disc.to(device)
        return self

    def compute_loss(self, model, src_x, src_y, tgt_x, p):
        n_s = src_x.size(0)
        logits, feat = model(torch.cat([src_x, tgt_x], dim=0))

        loss_cls = F.cross_entropy(logits[:n_s], src_y)   # source labels only

        alpha = self.alpha_max * dann_alpha(p)
        # L2-normalise the discriminator's input: the feature extractor maximises
        # the domain loss, which is unbounded above, and with BatchNorm running
        # statistics frozen nothing bounds activation magnitude. Feeding direction
        # only means feature norm can no longer manufacture confident (wrong)
        # domain predictions. The classifier head still receives the raw feature.
        f_rev = grad_reverse(feat, alpha)
        if self.normalize_input:
            f_rev = F.normalize(f_rev, dim=1)
        d_logits = self.disc(f_rev)
        d_labels = torch.cat([
            torch.zeros(n_s, dtype=torch.long, device=feat.device),          # source = 0
            torch.full((feat.size(0) - n_s,), 1, dtype=torch.long, device=feat.device),
        ])
        loss_dom = F.cross_entropy(d_logits, d_labels)

        loss = loss_cls + self.lam_domain * loss_dom

        with torch.no_grad():
            acc = (logits[:n_s].argmax(1) == src_y).float().mean().item()
            # domain accuracy is the key DIAGNOSTIC: 0.5 means the discriminator
            # cannot tell the domains apart. Read it alongside loss_cls, since
            # chance can also mean an undertrained disc or collapsed features.
            d_acc = (d_logits.argmax(1) == d_labels).float().mean().item()
        return loss, {'loss_total': loss.item(), 'loss_cls': loss_cls.item(),
                      'loss_dom': loss_dom.item(), 'domain_acc': d_acc,
                      'alpha': alpha, 'train_acc': acc}