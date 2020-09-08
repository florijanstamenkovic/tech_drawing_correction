import numpy as np
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


class ConvTransposeBNRelu(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=(3, 3), stride=2, padding=1):

        super().__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Bottleneck(nn.Module):
    CONTRACTION = 2

    def __init__(self, channels, residual):
        super().__init__()

        self.residual = residual
        self.block1 = ConvBNRelu(channels, channels // Bottleneck.CONTRACTION)
        self.block2 = ConvBNRelu(channels // Bottleneck.CONTRACTION, channels)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        if self.residual:
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

        # with torch.no_grad():
        #     for cbn in (self.conv1, self.conv2, self.conv3):
        #         c = cbn.block[0]
        #         central_value = 1.0 / c.weight.shape[0]
        #         weights = (np.random.random(c.weight.shape) - 0.5) * \
        #             (central_value / 10)
        #         weights[:, :, 1, 1] = central_value
        #         c.weight = nn.Parameter(torch.FloatTensor(weights))

        # self.deconv4 = ConvTransposeBNRelu(256, 128, 3, 2)
        self.deconv3 = ConvBNRelu(128, 64)
        self.deconv2 = ConvBNRelu(64, 32)
        self.deconv1 = ConvBNRelu(32, 32)
        self.conv_out = nn.Conv2d(32, 1, kernel_size=(3, 3), bias=True,
                                  padding=1)

    def forward(self, x):
        out1 = self.pool1(self.conv1(x))  # 1/2
        out2 = self.pool2(self.conv2(out1))  # 1/4
        out3 = self.pool3(self.conv3(out2))  # 1/8
        # out4 = self.pool4(self.conv4(out3))  # 1/16

        # out3 = self.deconv4(out4)
        out2 = nn.functional.upsample_bilinear(self.deconv3(out3), scale_factor=2)
        out1 = nn.functional.upsample_bilinear(self.deconv2(out2), scale_factor=2)
        out0 = nn.functional.upsample_bilinear(self.deconv1(out1), scale_factor=2)
        out = out0[:, :, :800, :800]
        out = torch.relu(self.conv_out(out))

        # out = x
        # out = self.conv1(out)
        # out = self.conv2(out)
        # out = self.conv3(out)
        # out = self.conv4(out)
        # out = torch.relu(out)
        return out


if __name__ == "__main__":
    from torchsummary import summary
    model = Network()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, input_size=(1, SIDE_LENGTH, SIDE_LENGTH))
