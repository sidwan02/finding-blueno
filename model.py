import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np


class UnetDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDoubleConv, self).__init__()
        # https://github.com/zhixuhao/unet/issues/98
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        feature_channel_powers = np.array([0, 6, 7, 8, 9, 10])
        base = 2
        feature_channels = np.power(base, feature_channel_powers)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_layer_1 = UnetDoubleConv(
            in_channels, feature_channels[1])

        self.down_layer_2 = UnetDoubleConv(
            feature_channels[1], feature_channels[2])

        self.down_layer_3 = UnetDoubleConv(
            feature_channels[2], feature_channels[3])

        self.down_layer_4 = UnetDoubleConv(
            feature_channels[3], feature_channels[4])

        self.up_layer_1 = UnetDoubleConv(
            feature_channels[5], feature_channels[4])

        self.up_conv_1 = nn.ConvTranspose2d(
            feature_channels[5], feature_channels[4],                                   kernel_size=2, stride=2)

        self.up_layer_2 = UnetDoubleConv(
            feature_channels[4], feature_channels[3])

        self.up_conv_2 = nn.ConvTranspose2d(feature_channels[4],
                                            feature_channels[3],
                                            kernel_size=2, stride=2)

        self.up_layer_3 = UnetDoubleConv(
            feature_channels[3], feature_channels[2])

        self.up_conv_3 = nn.ConvTranspose2d(feature_channels[3],
                                            feature_channels[2],
                                            kernel_size=2, stride=2)

        self.up_layer_4 = UnetDoubleConv(
            feature_channels[2], feature_channels[1])

        self.up_conv_4 = nn.ConvTranspose2d(feature_channels[2],
                                            feature_channels[1],
                                            kernel_size=2, stride=2)

        self.bottom_conv = UnetDoubleConv(
            feature_channels[4], feature_channels[5])
        self.final_conv = nn.Conv2d(
            feature_channels[1], feature_channels[0], kernel_size=1)

    def forward(self, x):
        upLayerResults = []

        # down
        x = self.down_layer_1(x)
        upLayerResults.append(x)
        x = self.pool(x)

        x = self.down_layer_2(x)
        upLayerResults.append(x)
        x = self.pool(x)

        x = self.down_layer_3(x)
        upLayerResults.append(x)
        x = self.pool(x)

        x = self.down_layer_4(x)
        upLayerResults.append(x)
        x = self.pool(x)

        # bottom
        x = self.bottom_conv(x)

        upLayerResults = upLayerResults[::-1]

        # up
        x = self.up_conv_1(x)
        skip_connection = upLayerResults[0]
        x = TF.resize(x, size=skip_connection.shape[2:])
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.up_layer_1(concat_skip)

        x = self.up_conv_2(x)
        skip_connection = upLayerResults[1]
        x = TF.resize(x, size=skip_connection.shape[2:])
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.up_layer_2(concat_skip)

        x = self.up_conv_3(x)
        skip_connection = upLayerResults[2]
        x = TF.resize(x, size=skip_connection.shape[2:])
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.up_layer_3(concat_skip)

        x = self.up_conv_4(x)
        skip_connection = upLayerResults[3]
        x = TF.resize(x, size=skip_connection.shape[2:])
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.up_layer_4(concat_skip)

        # final
        x = self.final_conv(x)

        return x
