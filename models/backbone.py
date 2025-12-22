import torch.nn as nn
import torchvision.models as models

class MobileNetV3Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.mobilenet_v3_large(pretrained=True)
        self.features = m.features
        self.out_channels = [40, 112, 960]

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [6, 12, 16]:
                outputs.append(x)
        return outputs
