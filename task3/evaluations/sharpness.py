

import numpy as np
import torch
import torch.nn.functional as F


def fixed_val_batch(source_val, device, per_domain=32, seed=6304):

    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for d, ld in source_val.items():
        ds = ld.dataset
        idx = rng.permutation(len(ds))[:per_domain]
        for i in idx:
            x, y = ds[int(i)]
            xs.append(x)
            ys.append(y)
    return (torch.stack(xs).to(device),
            torch.tensor(ys, dtype=torch.long, device=device))


def sharpness_proxy(model, x, y, rho=0.05):
   
    model.eval()
    params = [q for q in model.parameters() if q.requires_grad]

    model.zero_grad(set_to_none=True)
    loss_clean = F.cross_entropy(model(x)[0], y)
    loss_clean.backward()

    with torch.no_grad():
        gnorm = torch.sqrt(sum((q.grad ** 2).sum() for q in params if q.grad is not None))
        scale = rho / (gnorm + 1e-12)
        eps = []
        for q in params:
            e = q.grad * scale if q.grad is not None else None
            if e is not None:
                q.add_(e)
            eps.append(e)

        loss_pert = F.cross_entropy(model(x)[0], y)

        for q, e in zip(params, eps):        # restore, always
            if e is not None:
                q.sub_(e)

    model.zero_grad(set_to_none=True)
    return {'loss_clean': float(loss_clean.item()),
            'loss_perturbed': float(loss_pert.item()),
            'delta_sharp': float(loss_pert.item() - loss_clean.item()),
            'grad_norm': float(gnorm.item()), 'rho': rho}