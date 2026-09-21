import torch
import torch.nn as nn
from torchvision.models import resnet18


class CifarResNet18(nn.Module):
  
    def __init__(self, num_classes=10):
        super().__init__()
        net = resnet18(weights=None, num_classes=num_classes)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        self.feature_dim = net.fc.in_features      # 512
        net.fc = nn.Identity()                     # head lives outside the backbone
        self.net = net
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward_pre(self, x):
       
        n = self.net
        x = n.relu(n.bn1(n.conv1(x)))
        x = n.maxpool(x)                 # Identity
        x = n.layer1(x)
        x = n.layer2(x)
        return x

    def forward_post(self, h):

        n = self.net
        h = n.layer3(h)
        h = n.layer4(h)
        h = n.avgpool(h)
        return torch.flatten(h, 1)

    def forward(self, x):
        f = self.forward_post(self.forward_pre(x))
        return self.head(f), f