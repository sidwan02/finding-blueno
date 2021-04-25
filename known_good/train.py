import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
# from model import UNET
from model_known_good import UNET

from loaders import (
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

import os
my_path = os.path.dirname(__file__)


# Hyperparameters etc.
lr = 1e-6
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 2
epochs = 5
img_height = 160
img_width = 240


def main():

    model = UNET(in_channels=3, out_channels=1).to(device)
    loss_fun = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader, test_loader = get_loaders(
        batch_size
    )

    # check_accuracy(test_loader, model, device=device)

    for epoch in range(epochs):
        train_loading_bar = tqdm(train_loader)
        model.train()

        train_correct_pixels = 0
        train_total_pixels = 0

        # save_predictions_as_imgs(
        #     test_loader, model, folder=my_path + "//saved_images//", device=device
        # )

        for _, (pixel_data, target_masks) in enumerate(train_loading_bar):
            pixel_data = pixel_data.to(device=device)
            # print(pixel_data.size())
            target_masks_unsqueezed = target_masks.float().unsqueeze(1).to(device=device)
            # target_masks_unsqueezed = target_masks_unsqueezed.to(device=device)
            # print(target_masks_unsqueezed.size())

            # print(torch.transpose(pixel_data, 0, 1).size())
            model.zero_grad()

            predictions = model(pixel_data)
            # print(predictions.size())
            loss = loss_fun(predictions, target_masks_unsqueezed)
            loss.backward()

            (total_pixels, correct_pixels) = check_accuracy(
                predictions, target_masks, device)

            train_correct_pixels = train_correct_pixels + correct_pixels
            train_total_pixels = train_total_pixels + total_pixels

            optimizer.step()

            # update tqdm train_loading_bar
            train_loading_bar.set_postfix(loss=loss.item())

            # print(
            #     f"{correct_pixels/total_pixels*100:.2f}"
            # )

        model.eval()

        # check_accuracy
        print(
            f"{train_correct_pixels/train_total_pixels*100:.2f}"
        )

        # print some examples to a folder
        save_predictions_as_imgs(
            test_loader, model, folder=my_path + "//saved_images//", device=device
        )

        test_loading_bar = tqdm(test_loader)

        test_correct_pixels = 0
        test_total_pixels = 0

        for _, (pixel_data, target_masks) in enumerate(test_loading_bar):
            pixel_data = pixel_data.to(device=device)
            # print(pixel_data.size())
            target_masks = target_masks.float().unsqueeze(1).to(device=device)
            # target_masks = target_masks.to(device=device)
            # print(target_masks.size())

            # print(torch.transpose(pixel_data, 0, 1).size())
            model.zero_grad()

            predictions = model(pixel_data)
            # print(predictions.size())
            loss = loss_fun(predictions, target_masks)
            loss.backward()

            (total_pixels, correct_pixels) = check_accuracy(
                predictions, target_masks, device)

            optimizer.step()

            # update tqdm test_loading_bar
            test_loading_bar.set_postfix(loss=loss.item())

            # check accuracy
            # check_accuracy(test_loader, model, device=device)
            # (correct_pixels, total_pixels) = check_accuracy(predictions, target_masks)

            test_correct_pixels += correct_pixels
            test_total_pixels += total_pixels

        # print some examples to a folder
        save_predictions_as_imgs(
            test_loader, model, folder=my_path + "//saved_images//", device=device
        )


if __name__ == "__main__":
    main()
