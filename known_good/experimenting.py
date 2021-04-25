import torch
import torchvision
from dataset import getBluenoImages
from torch.utils.data import DataLoader
from PIL import Image
import imageio
import albumentations as A
from albumentations.pytorch import ToTensorV2


import numpy as np

import os

my_path = os.path.dirname(__file__)

src_path = my_path + "\\data\\train_images\\"

for path in os.listdir(my_path + "\\data\\train_images\\"):
    inputPath = os.path.join(my_path + "\\data\\train_images\\", path)

    img = Image.open(inputPath)

    im = imageio.imread(inputPath)

    image = np.array(Image.open(inputPath).convert("RGB"))
    # mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
    comparison = im == image
    equal_arrays = comparison.all()
    print(equal_arrays)
    train_transform = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    image = np.array(Image.open(inputPath).convert("RGB"))

    # if self.transform is not None:
    augmentations = train_transform(image=image)
    print(augmentations["image"])
    break
