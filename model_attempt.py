import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np


class UnetDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDoubleConv, self).__init__()
        print('in: ', in_channels)
        print('out: ',  out_channels)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True)
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


# class UnetDoubleUpConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UnetDoubleUpConv, self).__init__()
#         self.layer = nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
#                                stride=1, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3,
#                                stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))

#     def forward(self, x):
#         x = self.layer(x)
#         return x


class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        feature_channel_powers = np.array([0, 6, 7, 8, 9, 10])
        base = 2
        feature_channels = np.power(base, feature_channel_powers)

        self.conv_layers = nn.ModuleList()
        self.up_conv_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_channels = 3
        out_channels = 1

        feature_index = 0
        while feature_index < feature_channels.size - 2:
            self.conv_layers.append(UnetDoubleConv(
                feature_channels[feature_index], feature_channels[feature_index + 1]))

            feature_index += 1

        reversed_feature_index = 0
        reversed_feature_channels = np.flip(feature_channels)
        print(reversed_feature_channels)
        while reversed_feature_index < reversed_feature_channels.size - 2:
            self.up_conv_layers.append(
                nn.ConvTranspose2d(reversed_feature_channels[reversed_feature_index],
                                   reversed_feature_channels[reversed_feature_index + 1],
                                   kernel_size=2, stride=2))

            self.up_conv_layers.append(UnetDoubleConv(
                reversed_feature_channels[reversed_feature_index],
                reversed_feature_channels[reversed_feature_index + 1]))

            reversed_feature_index += 1

        self.bottom_conv = UnetDoubleConv(
            feature_channels[-2], feature_channels[-1])
        self.final_conv = nn.Conv2d(
            feature_channels[1], feature_channels[0], kernel_size=1)

    def forward(self, x):
        skip_connections = []

        print(len(self.conv_layers))

        for down in self.conv_layers:
            print('down')
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottom_conv(x)
        print('bottom conv')

        skip_connections = skip_connections[::-1]

        print(len(self.up_conv_layers))

        for idx in range(0, len(self.up_conv_layers), 2):
            x = self.up_conv_layers[idx](x)
            skip_connection = skip_connections[idx//2]

            print('x shape: ', x.shape)
            print('skip_connection shape: ', skip_connection.shape)

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.up_conv_layers[idx+1](concat_skip)

        return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 161, 161))
    # model = UNET(in_channels=1, out_channels=1)
    model = UNET()
    preds = model(x)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
