"""
One training loop for every Task 4 model.

"""
import copy
import time
from pathlib import Path

import torch

from task4.evaluations.metrics import extract_outputs, accuracy, macro_f1


def train_model(model, method, loaders, cfg, device, ckpt_path=None,
                verbose_every=5):
    params = list(model.parameters()) + list(method.parameters())
    opt = torch.optim.SGD(params, lr=cfg['lr'], momentum=cfg['momentum'],
                          weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'])

    history, best, best_state, best_method_state = [], {'val_acc': -1.0, 'epoch': -1}, None, None
    t0 = time.time()

    for epoch in range(1, cfg['epochs'] + 1):
        model.train()          # BatchNorm ACTIVE here (random init, unlike Task 2)
        method.train()
        logs, n_steps = {}, 0

        for x, y in loaders['train']:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss, step_logs = method.compute_loss(model, x, y)
            loss.backward()
            opt.step()
            for k, v in step_logs.items():
                logs[k] = logs.get(k, 0.0) + v
            n_steps += 1

        lr_now = opt.param_groups[0]['lr']
        sched.step()                               # cosine steps ONCE PER EPOCH
        logs = {k: v / n_steps for k, v in logs.items()}

        val = extract_outputs(model, loaders['val'], device, keep_features=False)
        val_acc = accuracy(val['logits'], val['labels'])
        history.append({'epoch': epoch, 'lr': lr_now, **logs, 'val_acc': val_acc})

        if val_acc > best['val_acc'] + 1e-9:
            # deepcopy is REQUIRED: state_dict() hands back live tensors, so
            # without it the "best" snapshot keeps mutating and the final epoch
            # gets saved instead (the Task 1 bug).
            best = {'val_acc': val_acc, 'epoch': epoch}
            best_state = copy.deepcopy(model.state_dict())
            best_method_state = copy.deepcopy(method.state_dict())

        if verbose_every and (epoch <= 3 or epoch % verbose_every == 0
                              or epoch == cfg['epochs']):
            body = ' '.join(f'{k}={v:.4f}' for k, v in logs.items())
            print(f"ep{epoch:03d} lr={lr_now:.4f} {body} | val_acc={val_acc:.4f} "
                  f"| best={best['val_acc']:.4f}@{best['epoch']} "
                  f"| {(time.time()-t0)/60:.1f}m")

    model.load_state_dict(best_state)              # leave the BEST model in memory
    method.load_state_dict(best_method_state)
    if ckpt_path:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model': best_state, 'method': best_method_state,
                    'best_epoch': best['epoch'], 'best_val_acc': best['val_acc'],
                    'config': cfg}, ckpt_path)
    return {'history': history, 'best_epoch': best['epoch'],
            'best_val_acc': best['val_acc'], 'minutes': (time.time() - t0) / 60}