import torch.nn as nn
import torchvision.models as models

class VehicleRecognition(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_small(pretrained=True)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.type_head = nn.Linear(576, 4)
        self.color_head = nn.Linear(576, 6)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.type_head(x), self.color_head(x)
