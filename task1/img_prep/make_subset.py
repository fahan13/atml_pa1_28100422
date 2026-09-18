import numpy as np
from torchvision.datasets import STL10
from sklearn.model_selection import train_test_split

def load_stl10(root='../../datasets', split='train'):
    """split: 'train' (5000 labeled images) or 'test' (8000 labeled images)"""
    return STL10(root=root, split=split, download=True)

def stratified_train_val_split(dataset, seed=6304, val_frac=0.2):
    """
    Splits the train set into 80% train / 20% val, keeping each class's
    proportion the same in both pieces (stratify=labels does this).
    Returns two lists of indices (positions into the dataset), not the images themselves.
    """
    labels = dataset.labels  # array of class ids, one per image, e.g. [3, 7, 1, 3, ...]
    indices = np.arange(len(labels))  # [0, 1, 2, ..., 4999]

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_frac,
        random_state=seed,   # locks the "random" split so it's identical every run
        stratify=labels       # forces equal class proportion in both splits
    )
    return train_idx, val_idx

def balanced_test_subset(dataset, seed=6304, total=500):
    """
    Picks `total` images from the test set, split evenly across classes.
    Returns a shuffled list of indices (image identifiers).
    """
    labels = np.array(dataset.labels)
    classes = np.unique(labels)          # [0, 1, 2, ..., 9]
    per_class = total // len(classes)     # 500 // 10 = 50 images per class

    rng = np.random.RandomState(seed)     # a seeded random generator, reproducible
    selected = []
    for c in classes:
        class_idx = np.where(labels == c)[0]  # all positions belonging to class c
        chosen = rng.choice(class_idx, size=per_class, replace=False)  # pick 50, no repeats
        selected.extend(chosen.tolist())

    rng.shuffle(selected)  # mix classes together instead of leaving them grouped
    return selected