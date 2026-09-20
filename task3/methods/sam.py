"""
Task 3, Method 3: SAM — Sharpness-Aware Minimization.

    min_theta  max_{||eps|| <= rho}  L_ERM(theta + eps)

Standard non-adaptive SAM with rho = 0.05, over AdamW with the same lr and
weight decay as ERM. Two forward/backward passes per update: the first finds a
normalised ascent perturbation, the second computes the gradient at the
perturbed point, which is then applied to the ORIGINAL parameters.

"""
import torch
import torch.nn.functional as F


class SAM:
    name = 'sam'
    needs_target = False                 # never requests a target batch

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.rho = cfg.get('rho', 0.05)

    def modules(self):
        return []

    def parameters(self):
        return []

    def training_step(self, model, optimizer, src_x, src_y, tgt_x, p):
        params = [q for q in model.parameters() if q.requires_grad]

        # ---- pass 1: gradient at the current weights -------------------------
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(src_x)
        loss_clean = F.cross_entropy(logits, src_y)
        loss_clean.backward()

        # ---- ascent step: eps = rho * grad / ||grad||  (non-adaptive) --------
        with torch.no_grad():
            grad_norm = torch.sqrt(sum((q.grad.detach() ** 2).sum()
                                       for q in params if q.grad is not None))
            scale = self.rho / (grad_norm + 1e-12)
            eps = []
            for q in params:
                if q.grad is None:
                    eps.append(None)
                    continue
                e = q.grad.detach() * scale      # normalised, so ||eps|| == rho
                q.add_(e)                        # move UPHILL, to the worst
                eps.append(e)                    # nearby point in the ball

        # ---- pass 2: gradient at the perturbed weights -----------------------
        optimizer.zero_grad(set_to_none=True)
        logits_p, _ = model(src_x)
        loss_pert = F.cross_entropy(logits_p, src_y)
        loss_pert.backward()

        # ---- restore theta, then step with the PERTURBED gradients -----------
        with torch.no_grad():
            for q, e in zip(params, eps):
                if e is not None:
                    q.sub_(e)
        optimizer.step()

        with torch.no_grad():
            acc = (logits.argmax(1) == src_y).float().mean().item()
        
        return {'loss_total': loss_pert.item(), 'loss_cls': loss_clean.item(),
                'loss_pert': loss_pert.item(),
                'loss_gap': loss_pert.item() - loss_clean.item(),
                'train_acc': acc}