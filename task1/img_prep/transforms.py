#  task1/img_prep/transforms.py
from torchvision import transforms as T
import torchvision.transforms.functional as TF

def to_common_224(img):
    """One shared resize, identical for every backbone"""
    pipeline = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),  # PIL image -> tensor, pixel values scaled to [0,1]
    ])
    return pipeline(img.convert('RGB'))

def grayscale_transform(img_tensor):
    """Remove all color, keep 3 channels (still R=G=B) so normalization still works."""
    return TF.rgb_to_grayscale(img_tensor, num_output_channels=3)

def hue_rotate_transform(img_tensor, hue_factor=0.5):
    """Rotate every color around the color wheel by hue_factor*360 degrees.
    0.5 = maximum possible shift (180°) — chosen to make the color change as
    strong and unambiguous as possible, while shape/brightness stay untouched."""
    return TF.adjust_hue(img_tensor, hue_factor)

import torch.nn.functional as F

def translate_transform(img_tensor, shift, direction):
    """Shift the image by `shift` pixels in one cardinal direction.
    img_tensor: (3, H, W) in [0,1].
    direction: 'up', 'down', 'left', or 'right'.

    Uses reflection padding to avoid introducing black borders.
    """
    if shift == 0:
        return img_tensor

    C, H, W = img_tensor.shape
    p = shift
    # F.pad expects a batch dimension, so add one and remove it after
    padded = F.pad(img_tensor.unsqueeze(0), (p, p, p, p), mode='reflect').squeeze(0)

    # crop a HxW window, offset in the chosen direction
    if direction == 'right':    top, left = p, p + shift
    elif direction == 'left':   top, left = p, p - shift
    elif direction == 'down':   top, left = p + shift, p
    elif direction == 'up':     top, left = p - shift, p
    else: raise ValueError(f"unknown direction: {direction}")

    return padded[:, top:top + H, left:left + W]

# each model's own final normalization step, applied AFTER the shared resize
resnet_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # standard ImageNet stats
vit_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])      # same — both pretrained on ImageNet this way
clip_normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                              std=[0.26862954, 0.26130258, 0.27577711])                  # CLIP's own stats