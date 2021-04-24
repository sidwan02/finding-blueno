import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np


class UnetDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDoubleConv, self).__init__()
        # print('in: ', in_channels)
        # https://github.com/zhixuhao/unet/issues/98
        # print('out: ',  out_channels)
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

        # print(feature_channels)

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
            feature_channels[5],                                feature_channels[4],                                   kernel_size=2, stride=2)

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

        # feature_index = 0
        # while feature_index < feature_channels.size - 2:
        #     if feature_index == 0:
        #         self.down_conv_layers.append(UnetDoubleConv(
        #             in_channels, feature_channels[feature_index + 1]))
        #     else:
        #         self.down_conv_layers.append(UnetDoubleConv(
        #             feature_channels[feature_index], feature_channels[feature_index + 1]))

        #     feature_index += 1

        # reversed_feature_index = 0
        # reversed_feature_channels = np.flip(feature_channels)
        # # print(reversed_feature_channels)
        # while reversed_feature_index < reversed_feature_channels.size - 2:
        #     self.up_conv_layers.append(
        #         nn.ConvTranspose2d(reversed_feature_channels[reversed_feature_index],
        #                            reversed_feature_channels[reversed_feature_index + 1],
        #                            kernel_size=2, stride=2))

        #     self.up_conv_layers.append(UnetDoubleConv(
        #         reversed_feature_channels[reversed_feature_index],
        #         reversed_feature_channels[reversed_feature_index + 1]))

        #     reversed_feature_index += 1

        self.bottom_conv = UnetDoubleConv(
            feature_channels[4], feature_channels[5])
        self.final_conv = nn.Conv2d(
            feature_channels[1], feature_channels[0], kernel_size=1)

    # def forward(self, x):
    #     upLayerResults = []

    #     for down in self.down_conv_layers:
    #         x = down(x)
    #         upLayerResults.append(x)
    #         x = self.pool(x)

    #     x = self.bottom_conv(x)
    #     upLayerResults = upLayerResults[::-1]

    #     for idx in range(0, len(self.up_conv_layers), 2):
    #         x = self.up_conv_layers[idx](x)
    #         skip_connection = upLayerResults[idx//2]

    #         if x.shape != skip_connection.shape:
    #             x = TF.resize(x, size=skip_connection.shape[2:])

    #         concat_skip = torch.cat((skip_connection, x), dim=1)
    #         x = self.up_conv_layers[idx+1](concat_skip)

    #     return self.final_conv(x)
    def forward(self, x):
        upLayerResults = []

        # down
        x = self.down_layer_1(x)
        upLayerResults.append(x)
        x = self.pool(x)
        # print(x.shape)
        x = self.down_layer_2(x)
        upLayerResults.append(x)
        x = self.pool(x)
        # print(x.shape)
        x = self.down_layer_3(x)
        upLayerResults.append(x)
        x = self.pool(x)
        # print(x.shape)
        x = self.down_layer_4(x)
        upLayerResults.append(x)
        x = self.pool(x)
        # print(x.shape)

        # bottom
        x = self.bottom_conv(x)
        # print(x.shape)

        upLayerResults = upLayerResults[::-1]
        # print(len(upLayerResults))

        # up
        x = self.up_conv_1(x)
        # print("up 1", x.shape)
        skip_connection = upLayerResults[0]
        # # print("skip", skip_connection.shape)
        x = TF.resize(x, size=skip_connection.shape[2:])
        # # print("new_size", x.shape)
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.up_layer_1(concat_skip)
        # # print("up conv", x.shape)
        # print(x.shape)

        x = self.up_conv_2(x)
        # # print("up 2", x.shape)
        skip_connection = upLayerResults[1]
        # # print("skip 2", skip_connection.shape)
        x = TF.resize(x, size=skip_connection.shape[2:])
        # # print("resize 2", x.shape)
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.up_layer_2(concat_skip)
        # print(x.shape)

        # # print("wow")
        x = self.up_conv_3(x)
        skip_connection = upLayerResults[2]
        x = TF.resize(x, size=skip_connection.shape[2:])
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.up_layer_3(concat_skip)
        # print(x.shape)

        x = self.up_conv_4(x)
        skip_connection = upLayerResults[3]
        x = TF.resize(x, size=skip_connection.shape[2:])
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.up_layer_4(concat_skip)
        # print(x.shape)

        # print("hihahahah")

        # final
        x = self.final_conv(x)

        return x
        # upLayerResults = upLayerResults[::-1]

        # for idx in range(0, len(self.up_conv_layers), 2):
        #     x = self.up_conv_layers[idx](x)
        #     skip_connection = upLayerResults[idx//2]

        #     if x.shape != skip_connection.shape:
        #         x = TF.resize(x, size=skip_connection.shape[2:])

        #     concat_skip = torch.cat((skip_connection, x), dim=1)
        #     x = self.up_conv_layers[idx+1](concat_skip)

        # return self.final_conv(x)


def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    # print(preds.shape)
    # print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
