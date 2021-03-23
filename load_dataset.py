# import os
# from PIL import Image
# from torch.utils.data import Dataset
# import numpy as np


# class LoadDataset(Dataset):
#     def __init__(self, image_path, mask_path, transform=None):
#         self.image_path = image_path
#         self.mask_path = mask_path
#         self.transform = transform
#         self.images = os.listdir(image_path)

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, index):
#         img_path = os.path.join(self.image_path, self.images[index])
#         mask_path = os.path.join(
#             self.mask_path, self.images[index])
#         image = np.array(Image.open(img_path).convert("RGB"))
#         mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
#         mask[mask == 255.0] = 1.0

#         if self.transform is not None:
#             augmentations = self.transform(image=image, mask=mask)
#             image = augmentations["image"]
#             mask = augmentations["mask"]

#         return image, mask
