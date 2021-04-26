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

    # train images
    train_x = []
    train_y = []

    for (path_img, path_mask) in zip(os.listdir(my_path + "\\data\\train_images\\"), os.listdir(my_path + "\\data\\train_masks\\")):
        full_path_img = os.path.join(
            my_path + "\\data\\train_images\\", path_img)
        full_path_mask = os.path.join(
            my_path + "\\data\\train_masks\\", path_mask)

        from skimage.io import imread

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

    # test data
    test_x = []
    test_y = []

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


# def check_accuracy(x, y, model, device):
#     num_correct = 0
#     num_pixels = 0

#     x = x.to(device)
#     y = y.to(device)
#     y = torch.reshape(y, [1, 1 * 2 * 160 * 240]).data[0]

#     preds = torch.sigmoid(model(x))
#     preds = (preds > 0.5).float()
#     preds = torch.reshape(preds, [1, 1 * 2 * 160 * 240]).data[0]

#     # print(y)

#     # print(preds.size())
#     # print(y.size())

#     # for (pred_point, target_point) in zip(y, preds):
#     #     # print(pred_point)
#     #     if (pred_point == target_point):
#     #         num_correct += 1

#     #     num_pixels += 1

#     num_correct = (preds == y).sum()
#     num_pixels = torch.numel(preds)

#     return (num_correct, num_pixels)

def check_accuracy(preds, target, device):
    num_correct = 0
    num_pixels = 0
    # dice_score = 0

    # because this was unsqueezed in train.py
    # target = target.to(device).squeeze(1)
    preds = preds.to(device)
    target = target.to(device)
    # print(target)
    preds = torch.sigmoid(preds).squeeze(1)
    # print(preds)
    # print(target.size())
    # print(preds.size())
    preds = (preds > 0.5).float()
    num_correct += (preds == target).sum()
    num_pixels += torch.numel(preds)
    # dice_score += (2 * (preds * target).sum()) / (
    #     (preds + target).sum() + 1e-8
    # )

    # print(
    #     f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    # )
    # print(f"Dice score: {dice_score/len(loader)}")
    # model.train()

    return (num_correct, num_pixels)


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    count = 0
    for idx, (x, y) in enumerate(loader):
        # print(y.size())
        if count > 10:
            break
        else:
            x = x.to(device=device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                # print(preds.size())
                preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
        count = count + 1

    model.train()
