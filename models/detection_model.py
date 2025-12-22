import torch.nn as nn
import torchvision.models as models

class VehicleDetector(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        base = models.mobilenet_v3_large(pretrained=True)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls = nn.Linear(960, num_classes)
        self.box = nn.Linear(960, 4)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.cls(x), self.box(x)
