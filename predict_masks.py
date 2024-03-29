import torch
import os
from model import UNET
import numpy as np
from PIL import Image
import albumentations as A
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torchvision
from pathlib import Path
from albumentations.pytorch import ToTensorV2

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


def generate_masks():
    # https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch

    model = UNET(in_channels=3, out_channels=1)

    model.load_state_dict(torch.load(
        r"model" + r"\blueno_detection.pth"))

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    test_x = []

    image_names = []

    # get the image names to label the prediction masks appropriately
    for path_img in os.listdir(my_path + "/test_model/test_images"):
        image_names.append(path_img)
        full_path_img = os.path.join(
            my_path + "/test_model/test_images", path_img)

        image = np.array(Image.open(full_path_img).convert("RGB"))

        augmentations = transform(image=image)

        test_x.append(augmentations["image"])

    test_x = torch.stack(test_x)

    # the second test_x is a dummy to make the tensor of appropriate dimension so that it may be passed to the model
    test_data = TensorDataset(
        test_x, test_x)

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
    )

    Path(my_path + "/test_model/generated_masks/").mkdir(parents=True, exist_ok=True)

    for file in os.listdir(my_path + "/test_model/generated_masks/"):
        os.remove(my_path + "/test_model/generated_masks/" + file)

    save_prediction_masks(
        test_loader, model, image_names, "test_model/generated_masks/", device)

# convert preiction tensor into grayscale images to be saved


def save_prediction_masks(
    test_loader, model, image_names, target_folder, device,
):
    test_loading_bar = tqdm(test_loader, position=0, leave=True)

    count = 0
    for _, (pixel_data, _) in enumerate(test_loading_bar):
        pixel_data = pixel_data.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(pixel_data))

            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{target_folder}/prediction_{image_names[count]}"
        )

        count += 1


if __name__ == '__main__':
    generate_masks()
