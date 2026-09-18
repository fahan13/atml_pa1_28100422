import torch
import torch.nn as nn

# --- VGG-19 encoder, "normalised" variant used by the AdaIN authors.
# Note the leading 1x1 conv: this version expects plain [0,1] pixels
# (NO ImageNet mean/std), which is exactly what to_common_224 gives us.
vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 64, (3, 3)), nn.ReLU(),        # relu1_1
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),      # relu2_1
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, (3, 3)), nn.ReLU(),     # relu3_1
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, (3, 3)), nn.ReLU(),     # relu4_1 <- cutoff (index 30)
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
)

# --- Decoder: mirrors the encoder, turns modified features back into an image
decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3)),
)


def calc_mean_std(feat, eps=1e-5):
    """Per-image, per-channel mean and standard deviation of a feature map.
    feat shape: (N images, C channels, H, W).
    eps stops division by zero when a channel is perfectly flat."""
    N, C = feat.shape[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """AdaIN: strip the content's own per-channel statistics, then paste
    the style's statistics on instead. Spatial layout (= shape) survives;
    channel statistics (= texture/style) get swapped."""
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized = (content_feat - content_mean) / content_std       # centre + rescale
    return normalized * style_std + style_mean                      # re-scale + re-centre with style's stats


class AdaINStyleTransfer(nn.Module):
    def __init__(self, vgg_path, decoder_path, device='cuda'):
        super().__init__()
        vgg.load_state_dict(torch.load(vgg_path, map_location='cpu'))
        # keep only up to relu4_1 — that's where AdaIN operates
        self.encoder = nn.Sequential(*list(vgg.children())[:31]).to(device).eval()
        decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))
        self.decoder = decoder.to(device).eval()
        self.device = device
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, content, style, alpha=1.0):
        """content, style: (N,3,224,224) tensors in [0,1].
        alpha blends between pure content (0.0) and full style (1.0)."""
        content, style = content.to(self.device), style.to(self.device)
        content_f = self.encoder(content)
        style_f = self.encoder(style)
        t = adaptive_instance_normalization(content_f, style_f)
        t = alpha * t + (1 - alpha) * content_f      # style strength knob
        return self.decoder(t).clamp(0, 1)            # clamp back to valid pixel range