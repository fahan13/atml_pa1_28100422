"""
Task 3, Method 2: DAN-DG — pairwise alignment of the OBSERVED source domains

Same MMD implementation and kernel construction as Task 2 (imported, not
copied), so the only thing that differs between DAN and DAN-DG is WHICH
distributions are aligned: Task 2 aligns source against unlabelled Sketch,
this aligns Photo/Art/Cartoon against each other and never sees Sketch.
"""
import torch
import torch.nn.functional as F

from task2.methods.dan import mmd_rbf          # identical discrepancy measure


class DAN_DG:
    name = 'dan_dg'
    needs_target = False                       # never requests a target batch

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.lam = cfg.get('lambda_dg', 1.0)
        self.n_per = cfg.get('batch_per_source', 8)

    def modules(self):
        return []

    def parameters(self):
        return []

    def compute_loss(self, model, src_x, src_y, tgt_x, p):
        logits, feat = model(src_x)
        loss_cls = F.cross_entropy(logits, src_y)

        # next_source_batch concatenates 8 images per domain in SOURCE_DOMAINS
        # order, so the batch splits cleanly into per-domain blocks.
        assert feat.size(0) == 3 * self.n_per, 'batch is not 3 x batch_per_source'
        e_photo, e_art, e_cartoon = feat.split(self.n_per)

        pairs = [(e_photo, e_art), (e_photo, e_cartoon), (e_art, e_cartoon)]
        mmds = [mmd_rbf(a, b) for a, b in pairs]      # bandwidth re-estimated per pair
        loss_mmd = sum(mmds) / len(pairs)             # the manual's lambda/3 * sum

        loss = loss_cls + self.lam * loss_mmd

        with torch.no_grad():
            acc = (logits.argmax(1) == src_y).float().mean().item()
        return loss, {'loss_total': loss.item(), 'loss_cls': loss_cls.item(),
                      'loss_mmd': loss_mmd.item(),
                      'mmd_pa': mmds[0].item(), 'mmd_pc': mmds[1].item(),
                      'mmd_ac': mmds[2].item(), 'train_acc': acc}