import torch
import torch.nn as nn


# "the convolutional layers mostly have 3×3 filters and follow two simple design rules: ..."
# He et al., ‘Deep Residual Learning for Image Recognition’
RESNET_KERNEL_SIZE = 3


# used to match dimensions of input to output, done by a 1x1 convolution
# He et al., ‘Deep Residual Learning for Image Recognition’ page 4
def projection_shortcut(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            # "when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2"
            # He et al., ‘Deep Residual Learning for Image Recognition’.
            stride=2,
            kernel_size=1,
        ),
        nn.BatchNorm2d(out_channels),
    )


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=1, shortcut=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        out = self.conv0(x)
        out = self.conv1(out)
        if self.shortcut:
            out += self.shortcut(residual)
        else:
            out += residual
        out = self.relu(out)
        return out


# MAI in ResNet with 34 layers
# He et al., ‘Deep Residual Learning for Image Recognition’.
class MaiRes(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # first 7x7 conv layer
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            stride=2,
            padding=3,
            kernel_size=RESNET_KERNEL_SIZE,
        )
        self.maxpool = nn.MaxPool2d(kernel_size=RESNET_KERNEL_SIZE, stride=2)

        # layers are named after the colors used for each group
        # in the diagram presented in the ResNet paper

        # 3 residual blocks for a total of 6 layers
        self.layer_purple = nn.Sequential(
            ResidualBlock(
                in_channels=64,
                out_channels=64,
                stride=1,
            ),
            ResidualBlock(
                in_channels=64,
                out_channels=64,
                stride=1,
            ),
            ResidualBlock(
                in_channels=64,
                out_channels=64,
                stride=1,
            ),
        )

        # 4 residual blocks for a total of 8 layers
        self.layer_green = nn.Sequential(
            ResidualBlock(
                in_channels=64,
                out_channels=128,
                stride=2,
                shortcut=projection_shortcut(in_channels=64, out_channels=128),
            ),
            ResidualBlock(
                in_channels=128,
                out_channels=128,
                stride=1,
            ),
            ResidualBlock(
                in_channels=128,
                out_channels=128,
                stride=1,
            ),
            ResidualBlock(
                in_channels=128,
                out_channels=128,
                stride=1,
            ),
        )

        # 6 residual blocks for a total of 12 layers
        self.layer_red = nn.Sequential(
            ResidualBlock(
                in_channels=128,
                out_channels=256,
                stride=2,
                shortcut=projection_shortcut(in_channels=128, out_channels=256),
            ),
            ResidualBlock(
                in_channels=256,
                out_channels=256,
                stride=1,
            ),
            ResidualBlock(
                in_channels=256,
                out_channels=256,
                stride=1,
            ),
            ResidualBlock(
                in_channels=256,
                out_channels=256,
                stride=1,
            ),
            ResidualBlock(
                in_channels=256,
                out_channels=256,
                stride=1,
            ),
            ResidualBlock(
                in_channels=256,
                out_channels=256,
                stride=1,
            ),
        )

        # 3 residual blocks for a total of 6 layers
        self.layer_blue = nn.Sequential(
            ResidualBlock(
                in_channels=256,
                out_channels=512,
                stride=2,
                shortcut=projection_shortcut(in_channels=256, out_channels=512),
            ),
            ResidualBlock(
                in_channels=512,
                out_channels=512,
                stride=1,
            ),
            ResidualBlock(
                in_channels=512,
                out_channels=512,
                stride=1,
            ),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=RESNET_KERNEL_SIZE)
        self.fc = nn.Linear(in_features=2048, out_features=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)

        x = self.layer_purple(x)
        x = self.layer_green(x)
        x = self.layer_red(x)
        x = self.layer_blue(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
