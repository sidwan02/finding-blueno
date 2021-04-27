import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os

my_path = os.path.dirname(__file__)

transform = A.Compose(
    [
        A.Resize(height=160, width=240),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)


def get_loaders(
    batch_size,
    num_workers=1,
    pin_memory=True,
):

    train_x = []
    train_y = []

    # convert each train image and associated mask to their pixel representation; then convert the pixel arrays into tensors
    for (path_img, path_mask) in zip(os.listdir(my_path + "\\data\\train_images\\"), os.listdir(my_path + "\\data\\train_masks\\")):
        full_path_img = os.path.join(
            my_path + "\\data\\train_images\\", path_img)
        full_path_mask = os.path.join(
            my_path + "\\data\\train_masks\\", path_mask)

        image = np.array(Image.open(full_path_img).convert("RGB"))
        mask = np.array(Image.open(full_path_mask).convert("L"),
                        dtype=np.float32)

        augmentations = transform(image=image, mask=mask)

        train_x.append(augmentations["image"])
        train_y.append(augmentations["mask"])

    # https://discuss.pytorch.org/t/how-to-turn-a-list-of-tensor-to-tensor/8868/3

    train_x = torch.stack(train_x)
    train_y = torch.stack(train_y)

    train_data = TensorDataset(
        train_x, train_y)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    test_x = []
    test_y = []

    # convert each test image and associated mask to their pixel representation; then convert the pixel arrays into tensors
    for (path_img, path_mask) in zip(os.listdir(my_path + "\\data\\test_images\\"), os.listdir(my_path + "\\data\\test_masks\\")):
        full_path_img = os.path.join(
            my_path + "\\data\\test_images\\", path_img)
        full_path_mask = os.path.join(
            my_path + "\\data\\test_masks\\", path_mask)

        from skimage.io import imread

        image = np.array(Image.open(full_path_img).convert("RGB"))
        mask = np.array(Image.open(full_path_mask).convert("L"),
                        dtype=np.float32)

        augmentations = transform(image=image, mask=mask)

        test_x.append(augmentations["image"])
        test_y.append(augmentations["mask"])

    test_x = torch.stack(test_x)
    test_y = torch.stack(test_y)

    test_data = TensorDataset(
        test_x, test_y)

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, test_loader
