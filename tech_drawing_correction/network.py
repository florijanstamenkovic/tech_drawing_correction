import torch
import torch.nn as nn
import torch.nn.functional as F

from tech_drawing_correction.data import SIDE_LENGTH


class ConvBNRelu(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size=(3, 3), padding=1):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                # biases are handled by the batchnorm layer that follows
                padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Bottleneck(nn.Module):
    CONTRACTION = 2

    def __init__(self, channels):
        super().__init__()

        self.block1 = ConvBNRelu(channels, channels // Bottleneck.CONTRACTION)
        self.block2 = ConvBNRelu(channels // Bottleneck.CONTRACTION, channels)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = out + x
        return out


class Combine(nn.Module):

    def __init__(self, use_conv, in_channels, out_channels, scale=2):
        super().__init__()
        self._use_conv = use_conv
        self._scale = scale
        if use_conv:
            self._conv = ConvBNRelu(in_channels, out_channels, (1, 1), 0)

    def forward(self, deep, shallow):
        deep = F.interpolate(deep, scale_factor=self._scale, mode='bilinear')
        if self._use_conv:
            out = torch.cat([deep, shallow], dim=1)
            return self._conv(out)
        else:
            return deep + shallow


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = ConvBNRelu(1, 64)
        self.pool1 = nn.MaxPool2d(2)  # 1/2
        self.combine1 = Combine(True, 128, 64)

        self.conv2 = Bottleneck(64)
        self.pool2 = nn.MaxPool2d(2)  # 1/4
        self.combine2 = Combine(True, 128, 64)

        self.conv3 = Bottleneck(64)
        self.pool3 = nn.MaxPool2d(2)  # 1/8
        self.combine3 = Combine(True, 128, 64)

        self.conv4 = Bottleneck(64)
        self.pool4 = nn.MaxPool2d(2)  # 1/16

        self.final = ConvBNRelu(64, 1, (1, 1), 0)

    def forward(self, x):
        out1 = self.pool1(self.conv1(x))  # 1/2
        out2 = self.pool2(self.conv2(out1))  # 1/4
        out3 = self.pool3(self.conv3(out2))  # 1/8
        out4 = self.pool4(self.conv4(out3))  # 1/16

        out = self.combine3(out4, out3) # 1/8
        out = self.combine2(out, out2) # 1/4
        out = self.combine1(out, out1) # 1/2

        out = F.interpolate(out, scale_factor=2, mode='bilinear')
        out = self.final(out)
        out = x + out
        return out


if __name__ == "__main__":
    from torchsummary import summary
    model = Network()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=(1, SIDE_LENGTH, SIDE_LENGTH))
