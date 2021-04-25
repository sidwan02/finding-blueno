import torch
import torchvision
from dataset import getBluenoImages
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import imageio

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np

import os

my_path = os.path.dirname(__file__)


def get_loaders(
    batch_size,
    # train_transform,
    # test_transform,
    num_workers=4,
    pin_memory=True,
):
    # train_ds = getBluenoImages(
    #     image_dir=my_path + "\\data\\train_images\\",
    #     mask_dir=my_path + "\\data\\train_masks\\",
    #     # transform=train_transform,
    # )

    # train images
    train_x = []
    count = 0
    for path in os.listdir(my_path + "\\data\\train_images\\"):
        inputPath = os.path.join(my_path + "\\data\\train_images\\", path)

        img = Image.open(inputPath)

        from skimage.io import imread

        # im = imageio.imread(inputPath)
        # im = np.array(im)

        image = np.array(Image.open(inputPath).convert("RGB"))

        train_transform = A.Compose(
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

        augmentations = train_transform(image=image)
        # print(np.shape(augmentations["image"]))

        train_x.append(augmentations["image"])
        # count += 1
        # print("count: ", count)

    # train_x = np.array(train_x)
    # print(np.shape(train_x))

    # https://discuss.pytorch.org/t/how-to-turn-a-list-of-tensor-to-tensor/8868/3
    # print(train_x)

    # train_x = augmentations["train_x"]
    # print(np.shape(train_data))

    train_y = []
    count = 0
    for path in os.listdir(my_path + "\\data\\train_masks\\"):
        inputPath = os.path.join(my_path + "\\data\\train_masks\\", path)

        img = Image.open(inputPath)

        from skimage.io import imread

        # im = imageio.imread(inputPath)
        # # print(np.shape(im))
        # train_y.append(im)
        # mask = np.array(Image.open(inputPath).convert("L"), dtype=np.float32)

        image = np.array(Image.open(inputPath).convert("RGB"))
        # image = np.array(Image.open(inputPath).convert("L"), dtype=np.float32)

        # print(image)

        train_transform = A.Compose(
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

        augmentations = train_transform(image=image)

        train_y.append(augmentations["image"])
        # count += 1
    # print("count: ", count)

    # train_y = np.array(train_y)
    # print(train_y)
    # print(type(train_x))
    # print(type(train_y))

    train_x = torch.stack(train_x)
    train_y = torch.stack(train_y)

    train_data = TensorDataset(
        train_x, train_y)

    train_loader = DataLoader(
        # train_ds,
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    # test_ds = getBluenoImages(
    #     image_dir=my_path + "\\data\\test_images\\",
    #     mask_dir=my_path + "\\data\\test_masks\\",
    #     # transform=test_transform,
    # )

    test_x = []
    count = 0
    for path in os.listdir(my_path + "\\data\\test_images\\"):
        inputPath = os.path.join(my_path + "\\data\\test_images\\", path)

        img = Image.open(inputPath)

        from skimage.io import imread

        # im = imageio.imread(inputPath)
        # # print(np.shape(im))
        # test_x.append(im)
        test_transform = A.Compose(
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

        image = np.array(Image.open(inputPath).convert("RGB"))

        augmentations = test_transform(image=image)

        test_x.append(augmentations["image"])
        # count += 1
    # print("count: ", count)

    # test_x = np.array(test_x)
    # print(np.shape(test_data))

    test_y = []
    count = 0
    for path in os.listdir(my_path + "\\data\\test_masks\\"):
        inputPath = os.path.join(my_path + "\\data\\test_masks\\", path)

        img = Image.open(inputPath)

        from skimage.io import imread

        # im = imageio.imread(inputPath)
        # # print(np.shape(im))
        # test_y.append(im)

        test_transform = A.Compose(
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

        image = np.array(Image.open(inputPath).convert("RGB"))
        # image = np.array(Image.open(inputPath).convert("L"), dtype=np.float32)

        augmentations = test_transform(image=image)

        test_y.append(augmentations["image"])
        # count += 1
    # print("count: ", count)

    # test_y = np.array(test_y)

    # test_data = TensorDataset(torch.from_numpy(
    #     test_x), torch.from_numpy(test_y))

    test_x = torch.stack(test_x)
    test_y = torch.stack(test_y)

    test_data = TensorDataset(
        test_x, test_y)

    test_loader = DataLoader(
        # test_ds,
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, test_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    count = 0
    for idx, (x, y) in enumerate(loader):
        if count > 10:
            break
        else:
            x = x.to(device=device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
        count = count + 1

    model.train()
