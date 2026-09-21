"""
Data for Task 4 (Open-Set Recognition).

Knowns  : all ten CIFAR-10 classes. Official TRAIN partition -> stratified 90/10
          split (seed 6304); official TEST partition used whole for final CSA.
Unknowns: sixteen fixed CIFAR-100 TEST classes, eight near + eight far,
          100 images each = 800 per group. Evaluation only.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

NEAR_UNKNOWN = ['bus', 'pickup_truck', 'motorcycle', 'tractor',
                'wolf', 'fox', 'leopard', 'camel']
FAR_UNKNOWN  = ['bottle', 'bowl', 'chair', 'clock',
                'keyboard', 'mushroom', 'sunflower', 'wardrobe']


def build_transforms(randaugment=False):
    
    train_tf = [transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip()]
    if randaugment:
        train_tf.append(transforms.RandAugment(num_ops=2, magnitude=9))
    train_tf += [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    eval_tf = [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return transforms.Compose(train_tf), transforms.Compose(eval_tf)


def make_splits(root, seed=6304, val_fraction=0.1, download=True):
   
    base = CIFAR10(root, train=True, download=download)
    targets = np.array(base.targets)
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for c in range(10):
        idx = np.where(targets == c)[0]
        idx = idx[rng.permutation(len(idx))]
        n_val = int(round(val_fraction * len(idx)))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    train_idx = np.sort(np.concatenate(train_idx))
    val_idx = np.sort(np.concatenate(val_idx))
    return {'seed': seed, 'val_fraction': val_fraction,
            'train_idx': train_idx.tolist(), 'val_idx': val_idx.tolist(),
            'classes': list(base.classes)}


def build_known_loaders(root, splits, batch_size=128, eval_batch=512,
                        num_workers=0, seed=6304, randaugment=False):
   
    train_tf, eval_tf = build_transforms(randaugment)
    aug_train = Subset(CIFAR10(root, train=True, transform=train_tf), splits['train_idx'])
    raw_train = Subset(CIFAR10(root, train=True, transform=eval_tf),  splits['train_idx'])
    raw_val   = Subset(CIFAR10(root, train=True, transform=eval_tf),  splits['val_idx'])
    test      = CIFAR10(root, train=False, transform=eval_tf)

    g = torch.Generator()
    g.manual_seed(seed)   # shuffling RNG is explicit, so a rebuilt loader
                          # always starts from the same data order (Task 2 lesson)
    return {
        'train':      DataLoader(aug_train, batch_size=batch_size, shuffle=True,
                                 drop_last=True, num_workers=num_workers, generator=g),
        'train_eval': DataLoader(raw_train, batch_size=eval_batch, shuffle=False,
                                 num_workers=num_workers),
        'val':        DataLoader(raw_val, batch_size=eval_batch, shuffle=False,
                                 num_workers=num_workers),
        'test':       DataLoader(test, batch_size=eval_batch, shuffle=False,
                                 num_workers=num_workers),
        'steps_per_epoch': len(aug_train) // batch_size,
        'n_train': len(aug_train), 'n_val': len(raw_val), 'n_test': len(test),
        'classes': splits['classes'],
    }


def build_unknown_loaders(root, eval_batch=512, num_workers=0, download=True):
    
    _, eval_tf = build_transforms(randaugment=False)
    c100 = CIFAR100(root, train=False, transform=eval_tf, download=download)
    idx_to_name = {v: k for k, v in c100.class_to_idx.items()}
    targets = np.array(c100.targets)

    out = {'idx_to_name': idx_to_name}
    for group, names in (('near', NEAR_UNKNOWN), ('far', FAR_UNKNOWN)):
        missing = [n for n in names if n not in c100.class_to_idx]
        if missing:
            raise ValueError(f'CIFAR-100 class names not found: {missing}')
        wanted = [c100.class_to_idx[n] for n in names]
        sel = np.sort(np.where(np.isin(targets, wanted))[0])
        out[group] = {
            'loader': DataLoader(Subset(c100, sel.tolist()), batch_size=eval_batch,
                                 shuffle=False, num_workers=num_workers),
            'index': sel,
            'fine_labels': targets[sel],
            'names': [idx_to_name[t] for t in targets[sel]],
            'n': int(len(sel)),
        }
    return out