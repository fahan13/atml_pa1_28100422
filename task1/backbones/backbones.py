import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet50():
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    resnet = models.resnet50(weights=weights)
    
    # removing classification head
    resnet.fc = nn.Identity()
    
    # parameter freezing
    for param in resnet.parameters():
        param.requires_grad = False
    
    resnet.eval()  
    return resnet, weights.transforms()

def get_vit():
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    vit = models.vit_b_16(weights=weights)
    
    vit.heads = nn.Identity()  # removing classification head
    
    for param in vit.parameters():
        param.requires_grad = False
    
    vit.eval()
    return vit, weights.transforms()

import open_clip

def get_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='openai'
    )
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.eval()
    return model, preprocess

def extract_features(model, image_tensor, model_type='resnet_or_vit'):
    with torch.no_grad():
        if model_type == 'clip':
            features = model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)  # normalize to length 1
        else:
            features = model(image_tensor)
    return features

# task1/evaluations/evaluate_bias.py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

def extract_all_features(model, dataset, indices, normalize_fn, model_type='resnet_or_vit', batch_size=64, device='cuda'):
    from img_prep.transforms import to_common_224
    
    model = model.to(device)
    subset = Subset(dataset, indices)

    def collate(batch):
        imgs = torch.stack([normalize_fn(to_common_224(img)) for img, _ in batch])  # resize->tensor first, THEN normalize
        labels = torch.tensor([lbl for _, lbl in batch])
        return imgs, labels

    loader = DataLoader(subset, batch_size=batch_size, collate_fn=collate)

    all_feats, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            if model_type == 'clip':
                feats = model.encode_image(imgs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            else:
                feats = model(imgs)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_feats), np.concatenate(all_labels)