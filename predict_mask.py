import torch
import os
from model import UNET

my_path = os.path.dirname(__file__)


def generate_masks():
    # https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch

    model = UNET(in_channels=3, out_channels=1)

    model.load_state_dict(torch.load(
        r"model" + r"\blueno_detection.pth"))

    is_cuda = torch.cuda.is_available()

   # check if cuda is available
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    model.to(device)

    def process_images(text):
        test_x = []

        for path_img in os.listdir(my_path + "/test_model/test_images"):
            full_path_img = os.path.join(
                my_path + "/test_model/test_images", path_img)

            image = np.array(Image.open(full_path_img).convert("RGB"))

            augmentations = transform(image=image)

            test_x.append(augmentations["image"])

        test_x = torch.stack(test_x)
        save_predictions_as_imgs(
            test_x, model, "test_model/generated_masks/", device)

    def save_predictions_as_imgs(
        test_x, model, target_folder, device
    ):
        # model.eval()
        for x in test_x:
            x = x.to(device=device)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                # print(preds.size())
                preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{target_folder}/pred_{idx}.png"
            )
            torchvision.utils.save_image(
                y.unsqueeze(1), f"{target_folder}{idx}.png")


generate_masks()
