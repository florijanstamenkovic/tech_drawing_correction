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


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = ConvBNRelu(1, 32)
        self.pool1 = nn.MaxPool2d(2)  # 1/2
        self.combine1 = ConvBNRelu(64, 32, (1, 1), 0)

        self.conv2 = ConvBNRelu(32, 64)
        self.pool2 = nn.MaxPool2d(2)  # 1/4
        self.combine2 = ConvBNRelu(128, 32, (1, 1), 0)

        self.conv3 = ConvBNRelu(64, 128)
        self.pool3 = nn.MaxPool2d(2)  # 1/8
        self.combine3 = ConvBNRelu(256, 64, (1, 1), 0)

        self.conv4 = ConvBNRelu(128, 128)
        self.pool4 = nn.MaxPool2d(2)  # 1/16

        self.final = ConvBNRelu(32, 1, (1, 1), 0)

    def forward(self, x):
        out1 = self.pool1(self.conv1(x))  # 1/2
        out2 = self.pool2(self.conv2(out1))  # 1/4
        out3 = self.pool3(self.conv3(out2))  # 1/8
        out4 = self.pool4(self.conv4(out3))  # 1/16

        def decode(out_higher, out_lower, combine):
            up = F.interpolate(out_lower, scale_factor=2, mode='bilinear')
            return combine(torch.cat([out_higher, up], dim=1))

        out = decode(out3, out4, self.combine3) # 1/8
        out = decode(out2, out, self.combine2)  # 1/4
        out = decode(out1, out, self.combine1)  # 1/2

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
