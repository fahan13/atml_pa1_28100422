"""
PACS data layer. Shared by Task 2 (UDA) and Task 3 (DG).
Only indexing, splitting and transforms live here — no models, no losses,
so both tasks provably use the same images.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import train_test_split

DOMAINS = ['photo', 'art_painting', 'cartoon', 'sketch']
SOURCE_DOMAINS = ['photo', 'art_painting', 'cartoon']
TARGET_DOMAIN = 'sketch'

# Alphabetical and fixed forever: a class's integer label must never depend on
# the order the operating system happens to list folders in.
CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def _norm(name):
    """'Art Painting', 'art-painting' and 'art_painting' should all match."""
    return name.strip().lower().replace(' ', '_').replace('-', '_')


def _match_dir(parent, name):
    for p in sorted(Path(parent).iterdir()):
        if p.is_dir() and _norm(p.name) == _norm(name):
            return p
    return None


def find_domain_dir(root, domain):
    """Locate e.g. datasets/PACS/sketch, tolerating one wrapper level
    (many PACS downloads nest everything inside 'kfold/')."""
    root = Path(root)
    d = _match_dir(root, domain)
    if d is not None:
        return d
    for p in sorted(root.iterdir()):
        if p.is_dir():
            d = _match_dir(p, domain)
            if d is not None:
                return d
    raise FileNotFoundError(f"no '{domain}' folder found under {root}")


def scan_domain(root, domain):
    """
    Every image of one domain as records {'path', 'label', 'domain'}.
    Paths are stored RELATIVE to root and the list is sorted, so the same
    dataset on another machine produces the identical ordering.
    """
    root = Path(root)
    ddir = find_domain_dir(root, domain)
    records = []
    for cls in CLASSES:                      # iterate CLASSES, not the folder listing
        cdir = _match_dir(ddir, cls)
        if cdir is None:
            raise FileNotFoundError(f"class folder '{cls}' missing in {ddir}")
        files = sorted(p for p in cdir.rglob('*') if p.suffix.lower() in IMG_EXTS)
        for f in files:
            records.append({'path': f.relative_to(root).as_posix(),
                            'label': CLASS_TO_IDX[cls],
                            'domain': domain})
    return records


def make_splits(root, seed=6304, val_frac=0.2):
    """
    80/20 stratified train/val split INSIDE each source domain.
    The target domain is kept whole: all of it is the unlabelled adaptation
    set (Task 2) and, at the very end, the labelled evaluation set.
    """
    splits = {'seed': seed, 'val_frac': val_frac, 'classes': CLASSES,
              'source_domains': SOURCE_DOMAINS, 'target_domain': TARGET_DOMAIN,
              'domains': {}}

    for domain in SOURCE_DOMAINS:
        records = scan_domain(root, domain)
        labels = np.array([r['label'] for r in records])
        idx = np.arange(len(records))
        train_idx, val_idx = train_test_split(
            idx,
            test_size=val_frac,
            random_state=seed,   # locks the split: identical on every run and machine
            stratify=labels      # forces the same class proportions in both halves
        )
        splits['domains'][domain] = {
            'train': [records[i] for i in sorted(train_idx)],
            'val':   [records[i] for i in sorted(val_idx)],
        }

    splits['domains'][TARGET_DOMAIN] = {'all': scan_domain(root, TARGET_DOMAIN)}
    return splits


class PACSDataset(Dataset):
    """Reads the images listed in a record list produced by make_splits()."""
    def __init__(self, root, records, transform):
        self.root = Path(root)
        self.records = list(records)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        img = Image.open(self.root / rec['path']).convert('RGB')
        return self.transform(img), int(rec['label'])


def build_transforms():
    """
    Manual's protocol: resize to 256x256, then random 224 crop + horizontal
    flip for training, 224 centre crop for validation/evaluation.
    Normalization is the one that ships with the ImageNet ResNet-18 weights,
    because the pretrained filters expect inputs centred that way.
    """
    preset = ResNet18_Weights.IMAGENET1K_V1.transforms()
    mean = list(getattr(preset, 'mean', [0.485, 0.456, 0.406]))
    std  = list(getattr(preset, 'std',  [0.229, 0.224, 0.225]))

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf