"""
Method 4 of 4: CDAN — class-conditional adversarial alignment.

Identical to DANN except for WHAT the discriminator sees:
    DANN:  g(x) = f                      (512-d feature)
    CDAN:  g(x) = vec(f (x) p)           (512 x 7 outer product, flattened = 3584)

where p = softmax(C(f)) is the classifier's probability vector. The outer
product gives the discriminator a separate 512-weight subspace per class, so it
can learn a different domain rule for each class -- which is what stops the
backbone from satisfying alignment by mixing classes across domains.

"""
import torch
import torch.nn.functional as F

from .dann import DomainDiscriminator, dann_alpha, grad_reverse


class CDAN:
    name = 'cdan'
    needs_target = True

    def __init__(self, cfg=None):
        cfg = cfg or {}
        feat_dim = cfg.get('feat_dim', 512)
        n_cls = cfg.get('num_classes', 7)
    
        self.disc = DomainDiscriminator(in_dim=feat_dim * n_cls,
                                        hidden=cfg.get('disc_hidden', 256))
        self.alpha_max = cfg.get('alpha_max', 1.0)
        self.lam_domain = cfg.get('lambda_domain', 1.0)
        
        self.normalize_input = cfg.get('normalize_disc_input', False)

    def modules(self):
        return [self.disc]

    def parameters(self):
        return list(self.disc.parameters())

    def to(self, device):
        self.disc.to(device)
        return self

    def compute_loss(self, model, src_x, src_y, tgt_x, p_progress):
        n_s = src_x.size(0)
        logits, feat = model(torch.cat([src_x, tgt_x], dim=0))

        loss_cls = F.cross_entropy(logits[:n_s], src_y)      # source labels only

        probs = F.softmax(logits, dim=1)                     # NOT detached (manual)
        # outer product: (B, 512, 1) * (B, 1, 7) -> (B, 512, 7) -> flatten to (B, 3584).
        # Block k of the result is the feature scaled by P(class k), so a confident
        # "dog" puts the feature in the dog block and near-zero elsewhere.
        g = torch.bmm(feat.unsqueeze(2), probs.unsqueeze(1)).flatten(1)

        alpha = self.alpha_max * dann_alpha(p_progress)
        # Same L2-normalisation as DANN, for the same reason: the feature extractor
        # maximises an unbounded domain loss, and with BatchNorm statistics frozen
        # nothing else bounds activation magnitude. Note the outer product SQUARES
        # the scale problem (entries are products of two quantities), so this
        # matters more here, not less.
        g_rev = grad_reverse(g, alpha)
        if self.normalize_input:
            
            g_rev = F.normalize(g_rev, dim=1)
        d_logits = self.disc(g_rev)
        d_labels = torch.cat([
            torch.zeros(n_s, dtype=torch.long, device=feat.device),
            torch.full((feat.size(0) - n_s,), 1, dtype=torch.long, device=feat.device),
        ])
        loss_dom = F.cross_entropy(d_logits, d_labels)

        loss = loss_cls + self.lam_domain * loss_dom

        with torch.no_grad():
            acc = (logits[:n_s].argmax(1) == src_y).float().mean().item()
            d_acc = (d_logits.argmax(1) == d_labels).float().mean().item()
            # how confident the classifier is on TARGET images -- diagnostic only,
            # uses no labels. Low values mean the conditioning signal is weak.
            tgt_conf = probs[n_s:].max(1).values.mean().item()
        return loss, {'loss_total': loss.item(), 'loss_cls': loss_cls.item(),
                      'loss_dom': loss_dom.item(), 'domain_acc': d_acc,
                      'alpha': alpha, 'tgt_conf': tgt_conf, 'train_acc': acc}