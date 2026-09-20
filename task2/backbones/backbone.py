
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

_BN = (nn.BatchNorm1d, nn.BatchNorm2d)


class PACSModel(nn.Module):
    """
    x -> F(x) = 512-d feature -> C(f) = 7 logits.
    forward returns BOTH, because every alignment loss in Task 2 (MMD, DANN,
    CDAN) is computed on the 512-d feature immediately before the head.
    """
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.feature_dim = net.fc.in_features     # 512
        net.fc = nn.Identity()                    # drop the 1000-way ImageNet head
        self.backbone = net
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        return self.head(f), f


def freeze_bn_stats(model):
    """Set all BatchNorm layers to eval mode, so they don't update their running stats."""
    for m in model.modules():
        if isinstance(m, _BN):
            m.eval()


def set_train_mode(model):
    """The exact sequence the manual asks for: train() first, then BN back to eval."""
    model.train()
    freeze_bn_stats(model)