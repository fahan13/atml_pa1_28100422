"""
Download PACS once and write it out as plain image folders:
    datasets/PACS/<domain>/<class>/00000.jpg

Run from the repo root:
    python -m shared.download_pacs

"""
from pathlib import Path
from datasets import load_dataset, Image as HFImage

OUT = Path('datasets/PACS')


def main():
    ds = load_dataset('flwrlabs/pacs', split='train')
    class_names = ds.features['label'].names
    print('classes from the dataset card:', class_names)

    ds = ds.cast_column('image', HFImage(decode=False))

    counts = {}
    for rec in ds:
        domain, cls = rec['domain'], class_names[rec['label']]
        folder = OUT / domain / cls
        folder.mkdir(parents=True, exist_ok=True)
        n = counts.get((domain, cls), 0)

        blob = rec['image']
        if blob.get('bytes'):
            (folder / f'{n:05d}.jpg').write_bytes(blob['bytes'])
        else:                                    # fallback if only a path is given
            from PIL import Image
            Image.open(blob['path']).convert('RGB').save(folder / f'{n:05d}.jpg')
        counts[(domain, cls)] = n + 1

    for domain in sorted({d for d, _ in counts}):
        total = sum(v for (d, _), v in counts.items() if d == domain)
        per_class = {c: v for (d, c), v in sorted(counts.items()) if d == domain}
        print(f'{domain:>13s}  n={total:5d}  {per_class}')
    print('grand total:', sum(counts.values()), '(expected 9991)')


if __name__ == '__main__':
    main()