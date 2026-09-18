from sklearn.metrics import f1_score
import torch    



def evaluate_head(head, feats, labels, device='cuda'):
    head.eval()
    feats_t = torch.tensor(feats, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = head(feats_t)
        probs = torch.softmax(logits, dim=1)          # turn logits into a probability distribution
        preds = probs.argmax(dim=1).cpu().numpy()       # top-1 prediction per image
        max_conf = probs.max(dim=1).values.cpu().numpy()  # max_k p_k (the model's confidence in its own top guess), per image
    
    top1_acc = (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average='macro')  # average='macro' = per-class F1, then averaged equally
    mean_max_conf = max_conf.mean()
    
    return {"top1_acc": float(top1_acc), "macro_f1": float(macro_f1), "mean_max_conf": float(mean_max_conf)}

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
    return {"top1_acc": float(top1_acc), "macro_f1": float(macro_f1), "mean_max_conf": float(max_conf.mean())}