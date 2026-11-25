"""AlexNet and ResNet variants for 32x32 CIFAR-10 inputs (Lab 2, Part A9)."""

import torch
import torch.nn as nn


class CIFAR10AlexNet(nn.Module):
    """AlexNet adapted for 32x32 CIFAR-10 images."""

    def __init__(self, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 192, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        # 256 x 4 x 4 = 4096 flattened features -> 1024 -> 512 -> 10.
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 4 * 4, 1024), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(1024, 512), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    """Basic residual block: two 3x3 convolutions plus an identity or 1x1 projection shortcut."""

    def __init__(self, cin, cout, stride=1, dropout=0.0):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.d = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.s = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
        )

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.d(y)
        y = self.b2(self.c2(y))
        return torch.relu(y + self.s(x))


class CIFAR10ResNet(nn.Module):
    """ResNet-18 layout with a 3x3 stem and no initial max-pooling, as used for CIFAR-10."""

    def __init__(self, base_width=64, dropout=0.0):
        super().__init__()
        self.in_c = base_width
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_width, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_width), nn.ReLU(inplace=True),
        )
        self.s1 = self.stage(base_width, 2, 1, dropout)
        self.s2 = self.stage(base_width * 2, 2, 2, dropout)
        self.s3 = self.stage(base_width * 4, 2, 2, dropout)
        self.s4 = self.stage(base_width * 8, 2, 2, dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.d = nn.Dropout(dropout)
        self.fc = nn.Linear(base_width * 8, 10)

    def stage(self, out, n, stride, dropout):
        blocks = [BasicBlock(self.in_c, out, stride, dropout)]
        self.in_c = out
        blocks += [BasicBlock(out, out, 1, dropout) for _ in range(n - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        return self.fc(self.d(self.pool(x).flatten(1)))


def parameter_count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
