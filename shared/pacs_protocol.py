"""
The fixed experimental protocol for Tasks 2 and 3: loaders, batch composition,
and what counts as an epoch. Everything here stays IDENTICAL across
Source-only / DAN / DANN / CDAN — a method may only add a loss.
"""
import math
import torch
from torch.utils.data import DataLoader

from .pacs import PACSDataset, SOURCE_DOMAINS, TARGET_DOMAIN, build_transforms


class InfiniteLoader:
   
    def __init__(self, loader):
        self.loader = loader
        self._it = iter(loader)
        self.passes = 0          # how many full passes over this domain so far

    def next(self):
        try:
            return next(self._it)
        except StopIteration:
            self.passes += 1
            self._it = iter(self.loader)
            return next(self._it)


def _gen(seed):
    """A seeded generator so each loader's shuffling is reproducible."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def build_loaders(data_root, splits, batch_per_source=8, target_batch=24,
                  eval_batch=64, num_workers=2, seed=6304):
    """
    source_train : one InfiniteLoader per source domain, batch 8 (domain-balanced)
    target_train : one InfiniteLoader over ALL unlabelled sketches, batch 24
    source_val   : plain eval loader per source domain (checkpoint selection)
    target_eval  : plain eval loader over sketch (FINAL evaluation only, Step 5)
    """
    train_tf, eval_tf = build_transforms()

    source_train, source_val = {}, {}
    for i, d in enumerate(SOURCE_DOMAINS):
        tr = PACSDataset(data_root, splits['domains'][d]['train'], train_tf)
        va = PACSDataset(data_root, splits['domains'][d]['val'], eval_tf)
        source_train[d] = InfiniteLoader(DataLoader(
            tr, batch_size=batch_per_source, shuffle=True, drop_last=True,
            num_workers=num_workers, pin_memory=True, generator=_gen(seed + i)))
        source_val[d] = DataLoader(
            va, batch_size=eval_batch, shuffle=False, num_workers=num_workers)

    tgt = splits['domains'][TARGET_DOMAIN]['all']
    # adaptation sees target images with the TRAIN augmentation; the final
    # evaluation sees the very same images with the centre-crop pipeline
    target_train = InfiniteLoader(DataLoader(
        PACSDataset(data_root, tgt, train_tf), batch_size=target_batch,
        shuffle=True, drop_last=True, num_workers=num_workers,
        pin_memory=True, generator=_gen(seed + 100)))
    target_eval = DataLoader(
        PACSDataset(data_root, tgt, eval_tf), batch_size=eval_batch,
        shuffle=False, num_workers=num_workers)

    n_source_train = sum(len(splits['domains'][d]['train']) for d in SOURCE_DOMAINS)
    steps_per_epoch = math.ceil(n_source_train / (batch_per_source * len(SOURCE_DOMAINS)))

    return {'source_train': source_train, 'source_val': source_val,
            'target_train': target_train, 'target_eval': target_eval,
            'n_source_train': n_source_train, 'n_target': len(tgt),
            'steps_per_epoch': steps_per_epoch}


def next_source_batch(source_train, device):
    """Pull 8 images from each source domain, concatenate into one batch of 24."""
    xs, ys = [], []
    for loader in source_train.values():
        x, y = loader.next()
        xs.append(x)
        ys.append(y)
    return (torch.cat(xs).to(device, non_blocking=True),
            torch.cat(ys).to(device, non_blocking=True))