import torch
import torchvision
from dataset import getBluenoImages
from torch.utils.data import DataLoader
from PIL import Image
import imageio

import numpy as np

import os

my_path = os.path.dirname(__file__)


def get_loaders(
    batch_size,
    train_transform,
    test_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = getBluenoImages(
        image_dir=my_path + "\\data\\train_images\\",
        mask_dir=my_path + "\\data\\train_masks\\",
        transform=train_transform,
    )

    # train images
    train_data = []
    count = 0
    for path in os.listdir(my_path + "\\data\\train_images\\"):
        inputPath = os.path.join(my_path + "\\data\\train_images\\", path)

        img = Image.open(inputPath)

        from skimage.io import imread

        im = imageio.imread(inputPath)
        # print(np.shape(im))
        train_data.append(im)
        count += 1
    print("count: ", count)

    train_data = np.array(train_data)
    print(np.shape(train_data))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    test_ds = getBluenoImages(
        image_dir=my_path + "\\data\\test_images\\",
        mask_dir=my_path + "\\data\\test_masks\\",
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_ds,
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
