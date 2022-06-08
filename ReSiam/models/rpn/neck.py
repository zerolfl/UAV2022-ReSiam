import torch.nn as nn

class AdjustLayerAlex(nn.Module):
    def __init__(self, center_size=7):
        super(AdjustLayerAlex, self).__init__()
        self.center_size = center_size

    def forward(self, x):
        l = (x.size(3) - self.center_size) // 2
        r = l + self.center_size
        x = x[:, :, l:r, l:r]
        
        return x

class AdjustLayerConv(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super(AdjustLayerConv, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, stride=2),
            nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x = self.downsample(x)
        return x
