"""
Method 2 of 4: DAN — MMD alignment.

L_DAN = L_cls(source) + lambda * MMD^2(source features, target features)

MMD is applied to the 512-d feature immediately before the classifier head,
with a sum of three RBF kernels whose bandwidths are 0.5, 1 and 2 times the
median pairwise squared distance in the current combined batch.

"""
import torch
import torch.nn.functional as F


def mmd_rbf(src_f, tgt_f, mul_factors=(0.5, 1.0, 2.0), eps=1e-8):
    """
    Biased multi-kernel RBF MMD^2 between two sets of feature vectors.

    MMD^2 = mean(K_ss) + mean(K_tt) - 2 * mean(K_st)
          = within-source similarity + within-target similarity
            - 2 * cross-domain similarity
    Zero when the two clouds are indistinguishable at these length scales.
    """
    # float32 regardless of AMP: exp() of fp16 distances loses precision badly,
    # and this is a loss term, so accuracy here is cheap and worth it.
    z = torch.cat([src_f, tgt_f], dim=0).float()
    n_s = src_f.size(0)

    d2 = torch.cdist(z, z, p=2).pow(2)          # pairwise SQUARED distances

    with torch.no_grad():                        # the bandwidth is a constant
        n = z.size(0)
        off_diag = d2[~torch.eye(n, dtype=torch.bool, device=z.device)]
        median = off_diag.median().clamp(min=eps)   # typical pairwise distance

    # sum of three kernels: k(a,b) = exp(-||a-b||^2 / sigma), sigma = m * median
    K = sum(torch.exp(-d2 / (m * median)) for m in mul_factors)

    K_ss = K[:n_s, :n_s]
    K_tt = K[n_s:, n_s:]
    K_st = K[:n_s, n_s:]
    return K_ss.mean() + K_tt.mean() - 2.0 * K_st.mean()


class DAN:
    name = 'dan'
    needs_target = True

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.lam = cfg.get('lambda_mmd', 1.0)

    def modules(self):
        return []          # DAN adds no parameters, only a loss term

    def parameters(self):
        return []

    def compute_loss(self, model, src_x, src_y, tgt_x, p):
        n_s = src_x.size(0)
        # ONE forward pass over source+target concatenated. This is only safe
        # because BatchNorm running statistics are frozen: with BN in eval mode
        # the output for each image is independent of what else is in the batch,
        # so concatenating is numerically identical to two separate passes.
        logits, feat = model(torch.cat([src_x, tgt_x], dim=0))

        loss_cls = F.cross_entropy(logits[:n_s], src_y)     # source labels only
        loss_mmd = mmd_rbf(feat[:n_s], feat[n_s:])
        loss = loss_cls + self.lam * loss_mmd

        with torch.no_grad():
            acc = (logits[:n_s].argmax(1) == src_y).float().mean().item()
        return loss, {'loss_total': loss.item(), 'loss_cls': loss_cls.item(),
                      'loss_mmd': loss_mmd.item(), 'train_acc': acc}