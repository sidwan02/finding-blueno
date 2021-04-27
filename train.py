import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET

from loaders import *

import os
my_path = os.path.dirname(__file__)


# Hyperparameters
lr = 1e-6
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
epochs = 3
img_height = 160
img_width = 240


def main():
    model = UNET(in_channels=3, out_channels=1).to(device)
    loss_fun = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader, test_loader = get_loaders(
        batch_size
    )

    print("Training Model")
    print("==============")

    epoch_count = 1
    for epoch in range(epochs):
        print("Epoch ", epoch_count)
        print("---------")
        train_loading_bar = tqdm(train_loader, position=0, leave=True)
        model.train()

        train_correct_pixels = 0
        train_total_pixels = 0

        count = 0
        # iterate over the train data loader
        for _, (pixel_data, target_masks) in enumerate(train_loading_bar):
            count += 1
            pixel_data = pixel_data.to(device=device)
            target_masks_unsqueezed = target_masks.float().unsqueeze(1).to(device=device)

            model.zero_grad()

            predictions = model(pixel_data)

            loss = loss_fun(predictions, target_masks_unsqueezed)
            loss.backward()

            # get and accumualate the train accuracy
            (correct_pixels, total_pixels) = get_accuracy(
                predictions, target_masks, device)

            train_correct_pixels = train_correct_pixels + correct_pixels
            train_total_pixels = train_total_pixels + total_pixels

            optimizer.step()

            train_loading_bar.set_postfix(loss=loss.item())

        print(
            f"\nTrain Accuracy: {train_correct_pixels/train_total_pixels*100:.2f}%"
        )

        model.eval()

        epoch_count += 1

    # save model upon training
    print("Training Complete!")
    torch.save(model.state_dict(), r"model" + r"\blueno_detection.pth")

    test_loading_bar = tqdm(test_loader)

    test_correct_pixels = 0
    test_total_pixels = 0

    print("Testing Model")
    print("=============")
    count = 0
    # iterate over the test data loader
    for _, (pixel_data, target_masks) in enumerate(test_loading_bar):
        count += 1
        pixel_data = pixel_data.to(device=device)
        target_masks_unsqueezed = target_masks.float().unsqueeze(1).to(device=device)

        predictions = model(pixel_data)

        # get and accumualate the test accuracy
        (correct_pixels, total_pixels) = get_accuracy(
            predictions, target_masks, device)

        test_correct_pixels = test_correct_pixels + correct_pixels
        test_total_pixels = test_total_pixels + total_pixels

        test_loading_bar.set_postfix(loss=loss.item())

    print(
        f"\nTest Accuracy: {test_correct_pixels/test_total_pixels*100:.2f}%"
    )

# get the accuracy of the model prediction against the target mask


def get_accuracy(preds, target, device):
    num_correct = 0
    num_pixels = 0

    preds = preds.to(device)
    target = target.to(device)

    preds = torch.sigmoid(preds).squeeze(1)

    preds = (preds > 0.5).float()
    num_correct += (preds == target).sum()
    num_pixels += torch.numel(preds)

    return (num_correct, num_pixels)


if __name__ == "__main__":
    main()
