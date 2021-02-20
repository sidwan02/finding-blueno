import numpy as np
feature_channel_powers = np.array([0, 6, 7, 8, 9, 10])
base = 2
feature_channels = np.power(base, feature_channel_powers)
print(feature_channels)


feature_index = 0
while feature_index < feature_channels.size - 2:
    print(feature_channels[feature_index])
    print(feature_channels[feature_index + 1])

    feature_index += 1

reversed_feature_index = 0
reversed_feature_channels = np.flip(feature_channels)[:-1]
while reversed_feature_index < reversed_feature_channels.size - 2:
    print(reversed_feature_channels[reversed_feature_index])
    print(reversed_feature_channels[reversed_feature_index + 1])

    reversed_feature_index += 1
