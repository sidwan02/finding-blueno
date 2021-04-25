import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
# from model import UNET
from model_known_good import UNET

from utils import (
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
lr = 1e-6
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 1
epochs = 1
img_height = 160
img_width = 240


def main():
    # train_transform = A.Compose(
    #     [
    #         A.Resize(height=img_height, width=img_width),
    #         A.Normalize(
    #             mean=[0.0, 0.0, 0.0],
    #             std=[1.0, 1.0, 1.0],
    #             max_pixel_value=255.0,
    #         ),
    #         ToTensorV2(),
    #     ],
    # )

    # test_transform = A.Compose(
    #     [
    #         A.Resize(height=img_height, width=img_width),
    #         A.Normalize(
    #             mean=[0.0, 0.0, 0.0],
    #             std=[1.0, 1.0, 1.0],
    #             max_pixel_value=255.0,
    #         ),
    #         ToTensorV2(),
    #     ],
    # )

    model = UNET(in_channels=3, out_channels=1).to(device)
    loss_fun = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader, test_loader = get_loaders(
        batch_size,
        # train_transform,
        # test_transform,
    )

    check_accuracy(test_loader, model, device=device)

    for epoch in range(epochs):
        for (pixel_data, target_masks) in train_loader:
            pixel_data = pixel_data.to(device=device)
            target_masks = target_masks.float().unsqueeze(1).to(device=device)

            # forward
            predictions = model(pixel_data)
            loss = loss_fun(predictions, target_masks)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # check accuracy
        check_accuracy(test_loader, model, device=device)

        # print some examples to a folder
        save_predictions_as_imgs(
            test_loader, model, folder=my_path + "//saved_images//", device=device
        )


if __name__ == "__main__":
    main()
