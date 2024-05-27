import torch
import torch.nn as nn
import torch.nn.functional as F


class Res34Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1):
        super().__init__()
        # in_channels 和 out_channels 不匹配时需要 scale
        if in_channels != out_channels:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.scale = True
        else:
            self.scale = False

        # 卷积层
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.scale:
            x = self.linear(x)
        return F.relu(x + self.convs(x))


class ResNet34(nn.Module):
    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, nblocks: int, stride=2):
        layer = nn.Sequential()
        for i in range(nblocks):
            if i == 0 and in_channels != out_channels:
                layer.append(Res34Block(in_channels, out_channels, stride))
            else:
                layer.append(Res34Block(out_channels, out_channels))
        return layer

    def __init__(self, num_classes):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.res_convs = nn.Sequential(
            self._make_layer(64, 64, 3),
            self._make_layer(64, 128, 4),
            self._make_layer(128, 256, 6),
            self._make_layer(256, 512, 3)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pre(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
