"""
One training loop, used by all four methods.

"""
import copy
import time
from pathlib import Path

import torch

from shared.pacs_protocol import next_source_batch
from task2.backbones.backbone import set_train_mode
from task2.evaluations.metrics import evaluate_source_val


def train_model(model, method, loaders, cfg, device, ckpt_path=None, verbose=True):
    steps_per_epoch = cfg.get('steps_per_epoch') or loaders['steps_per_epoch']
    total_steps = steps_per_epoch * cfg['max_epochs']     # the PLANNED budget:
    # progress p is a fraction of this, so the DANN/CDAN schedule does not depend
    # on where early stopping happens to land

    params = list(model.parameters()) + list(method.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    use_amp = bool(cfg.get('amp', True)) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    history, best, best_state, bad, step = [], {'mean_macro_f1': -1.0, 'epoch': -1}, None, 0, 0
    t0 = time.time()

    for epoch in range(1, cfg['max_epochs'] + 1):
        set_train_mode(model)                 # train(), then BatchNorm back to eval
        for m in method.modules():
            m.train()
        epoch_logs = {}

        for _ in range(steps_per_epoch):
            src_x, src_y = next_source_batch(loaders['source_train'], device)
            tgt_x = None
            if method.needs_target:
                tx, _ = loaders['target_train'].next()     # labels exist on disk,
                tgt_x = tx.to(device, non_blocking=True)   # and are discarded right here
            p = min(1.0, step / max(1, total_steps - 1))

            if hasattr(method, 'training_step'):
                # SAM needs two forward/backward passes per update, so it owns
                # the step. Sampling, budget, schedule and checkpointing are
                # still the shared loop's, so the comparison stays controlled.
                logs = method.training_step(model, optimizer, src_x, src_y, tgt_x, p)
            else:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    loss, logs = method.compute_loss(model, src_x, src_y, tgt_x, p)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            for k, v in logs.items():
                epoch_logs[k] = epoch_logs.get(k, 0.0) + v
            step += 1

        epoch_logs = {k: v / steps_per_epoch for k, v in epoch_logs.items()}
        val = evaluate_source_val(model, loaders['source_val'], device)
        history.append({'epoch': epoch, 'progress': p, **epoch_logs, **val})

        if verbose:
            doms = ' '.join(f"{d}={m['macro_f1']:.3f}" for d, m in val['per_domain'].items())
            print(f"ep{epoch:02d} " + ' '.join(f'{k}={v:.4f}' for k, v in epoch_logs.items())
                  + f" | val {doms} mean_f1={val['mean_macro_f1']:.4f} | {time.time()-t0:.0f}s")

        if val['mean_macro_f1'] > best['mean_macro_f1'] + 1e-6:
            # deepcopy is REQUIRED: state_dict() returns live tensor references,
            # so without it the "best" snapshot keeps mutating and we would end
            # up saving the final epoch's weights instead (the Task 1 bug).
            best = {'mean_macro_f1': val['mean_macro_f1'], 'epoch': epoch, 'val': val}
            best_state = copy.deepcopy(model.state_dict())
            best_method_state = [copy.deepcopy(m.state_dict()) for m in method.modules()]
            bad = 0
        else:
            bad += 1
            if bad >= cfg['patience']:
                if verbose:
                    print(f"early stop at epoch {epoch} "
                          f"(no improvement for {cfg['patience']} epochs)")
                break

    model.load_state_dict(best_state)          # leave the BEST model in memory, not the last
    if ckpt_path:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model': best_state, 'method_modules': best_method_state,
                    'best_epoch': best['epoch'], 'source_val': best['val'],
                    'config': cfg}, ckpt_path)
    return {'history': history, 'best_epoch': best['epoch'],
            'best_mean_macro_f1': best['mean_macro_f1'], 'source_val_at_best': best['val'],
            'minutes': (time.time() - t0) / 60}