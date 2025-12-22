import torch.nn as nn

class SSDLiteHead(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.cls = nn.ModuleList()
        self.reg = nn.ModuleList()

        for c in channels:
            self.cls.append(nn.Conv2d(c, num_classes * 6, 3, padding=1))
            self.reg.append(nn.Conv2d(c, 4 * 6, 3, padding=1))

    def forward(self, features):
        cls_out, reg_out = [], []
        for f, c, r in zip(features, self.cls, self.reg):
            cls_out.append(c(f))
            reg_out.append(r(f))
        return cls_out, reg_out
