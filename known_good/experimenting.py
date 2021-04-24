import torch
import torchvision
from dataset import getBluenoImages
from torch.utils.data import DataLoader
from PIL import Image
import imageio

import numpy as np

import os

my_path = os.path.dirname(__file__)

src_path = my_path + "\\data\\train_images\\"

for path in os.listdir(my_path + "\\data\\train_images\\"):
    inputPath = os.path.join(my_path + "\\data\\train_images\\", path)

    img = Image.open(inputPath)

    im = imageio.imread(inputPath)
    print(np.shape(im))
    break
