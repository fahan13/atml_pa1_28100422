from sklearn.metrics import f1_score
import torch    



def evaluate_head(head, feats, labels, device=None):
    if device is None:
        device = next(head.parameters()).device

    head = head.to(device)
    head.eval()

    feats_t = torch.tensor(
        feats,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        logits = head(feats_t)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()
        max_conf = probs.max(dim=1).values.cpu().numpy()

    top1_acc = (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average="macro")
    mean_max_conf = max_conf.mean()

    return {
        "top1_acc": float(top1_acc),
        "macro_f1": float(macro_f1),
        "mean_max_conf": float(mean_max_conf)
    }

def evaluate_head_with_preds(head, feats, labels, device=None):
    if device is None:
        device = next(head.parameters()).device

    head = head.to(device)
    head.eval()

    feats_t = torch.tensor(
        feats,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        logits = head(feats_t)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()
        max_conf = probs.max(dim=1).values.cpu().numpy()

    metrics = {
        "top1_acc": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "mean_max_conf": float(max_conf.mean())
    }

    return metrics, preds

def prediction_consistency(preds_a, preds_b):
    """Fraction of images where two prediction sets agree"""
    return float((preds_a == preds_b).mean())

import open_clip

def clip_zero_shot_eval(clip_model, tokenizer, images_transformed, labels, class_names, device='cuda'):
    clip_model = clip_model.to(device)
    prompts = [f"a photo of a {c}." for c in class_names]  # the manual's fixed prompt template
    text_tokens = tokenizer(prompts).to(device)
    
    with torch.no_grad():
        text_feats = clip_model.encode_text(text_tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)  # normalize, same reason as image side
        
        img_feats = clip_model.encode_image(images_transformed.to(device))
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        
        # scaled similarity — CLIP learns a temperature that sharpens this comparison
        logit_scale = clip_model.logit_scale.exp()
        logits = logit_scale * img_feats @ text_feats.T   # @ = matrix multiplication (batch of dot products at once)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()
        max_conf = probs.max(dim=1).values.cpu().numpy()
    
    top1_acc = (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average='macro')
    metrics = {"top1_acc": float(top1_acc), "macro_f1": float(macro_f1), "mean_max_conf": float(max_conf.mean())}
    return metrics, preds

import numpy as np

def shape_texture_counts(preds, shape_labels, texture_labels):
    """Classify each prediction as the shape label, the texture label, or neither."""
    preds = np.asarray(preds)
    shape_labels = np.asarray(shape_labels)
    texture_labels = np.asarray(texture_labels)

    n_shape = int((preds == shape_labels).sum())
    n_texture = int((preds == texture_labels).sum())
    n_total = len(preds)
    n_other = n_total - n_shape - n_texture

    denom = n_shape + n_texture
    shape_bias = 100.0 * n_shape / denom if denom > 0 else float("nan")
    coverage = 100.0 * denom / n_total if n_total > 0 else float("nan")

    return {
        "n_shape": n_shape, "n_texture": n_texture, "n_other": n_other, "n_total": n_total,
        "shape_bias_pct": shape_bias, "coverage_pct": coverage,
    }


def extract_features_from_tensors(model, img_tensors, normalize_fn, model_type='resnet_or_vit',
                                   batch_size=64, device='cuda'):
    """Like extract_all_features, but for images we generated in memory
    (already 224x224, [0,1]) rather than loaded from a dataset."""
    model = model.to(device)
    feats_all = []
    with torch.no_grad():
        for i in range(0, len(img_tensors), batch_size):
            batch = img_tensors[i:i + batch_size]
            batch = torch.stack([normalize_fn(x) for x in batch]).to(device)
            if model_type == 'clip':
                f = model.encode_image(batch)
                f = f / f.norm(dim=-1, keepdim=True)
            else:
                f = model(batch)
            feats_all.append(f.cpu().numpy())
    return np.concatenate(feats_all)

def clip_zero_shot_batched(clip_model, tokenizer, img_tensors, labels, class_names,
                            batch_size=64, device='cuda'):
    """Same as clip_zero_shot_eval but processes images in chunks so peak memory
    stays flat regardless of how many images we pass in."""
    clip_model = clip_model.to(device)
    prompts = [f"a photo of a {c}." for c in class_names]
    with torch.no_grad():
        text_feats = clip_model.encode_text(tokenizer(prompts).to(device))
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        logit_scale = clip_model.logit_scale.exp()

        preds_all, conf_all = [], []
        for i in range(0, len(img_tensors), batch_size):
            batch = img_tensors[i:i + batch_size].to(device)
            f = clip_model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
            probs = torch.softmax(logit_scale * f @ text_feats.T, dim=1)
            preds_all.append(probs.argmax(1).cpu().numpy())
            conf_all.append(probs.max(1).values.cpu().numpy())
            del batch, f, probs

    preds = np.concatenate(preds_all)
    max_conf = np.concatenate(conf_all)
    metrics = {"top1_acc": float((preds == labels).mean()),
               "macro_f1": float(f1_score(labels, preds, average='macro')),
               "mean_max_conf": float(max_conf.mean())}
    return metrics, preds