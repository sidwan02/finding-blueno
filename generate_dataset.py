from PIL import Image
import os
from PIL import ImageFilter
import random
from pathlib import Path


my_path = os.path.dirname(__file__)


class ProcessDataset():
    def __init__(self, src_path, train_image_path, train_mask_path, test_image_path, test_mask_path, restricted):
        self.src_path = src_path
        self.train_image_path = train_image_path
        self.train_mask_path = train_mask_path
        self.test_image_path = test_image_path
        self.test_mask_path = test_mask_path
        self.restricted = restricted

        # create the directories if not already existing
        Path(my_path + "/data").mkdir(parents=True, exist_ok=True)
        Path(my_path + "/data/test_images").mkdir(parents=True, exist_ok=True)
        Path(my_path + "/data/train_images").mkdir(parents=True, exist_ok=True)
        Path(my_path + "/data/test_masks").mkdir(parents=True, exist_ok=True)
        Path(my_path + "/data/train_masks").mkdir(parents=True, exist_ok=True)

        # delete the contents of the directories if containing previous iterations of processed images
        for file in os.listdir(train_image_path):
            os.remove(train_image_path + "\\" + file)
        for file in os.listdir(train_mask_path):
            os.remove(train_mask_path + "\\" + file)
        for file in os.listdir(test_image_path):
            os.remove(test_image_path + "\\" + file)
        for file in os.listdir(test_mask_path):
            os.remove(test_mask_path + "\\" + file)

        count = 1
        for path in os.listdir(self.src_path):
            input_path = os.path.join(self.src_path, path)

            img = Image.open(input_path)

            # fixed dimension
            img = img.resize((240, 160))

            # 3:1 train:test ratio
            if count % 3 == 0:
                processed_image_path = os.path.join(
                    self.test_image_path, 'processed_' + path)
                mask_path = os.path.join(self.test_mask_path, 'mask_' + path)
            else:
                processed_image_path = os.path.join(
                    self.train_image_path, 'processed_' + path)
                mask_path = os.path.join(self.train_mask_path, 'mask_' + path)

            (scale_factor, horizontal_pos,
             vertical_pos) = self.get_img_superposition_state()
            processed_img = self.superimpose_target(
                background=img, scale_factor=scale_factor, horizontal_pos=horizontal_pos,
                vertical_pos=vertical_pos)

            processed_img.save(processed_image_path)

            mask = self.generate_mask(thresh=255, scale_factor=scale_factor,
                                      horizontal_pos=horizontal_pos, vertical_pos=vertical_pos)

            mask.save(mask_path)

            count += 1

    # randomly generate scaling and translation coefficients
    def get_img_superposition_state(self):
        if self.restricted:  # centered
            scale_factor = 100
        else:
            scale_factor = random.uniform(40, 160)

        if self.restricted:  # monosize
            horizontal_pos = 70
            vertical_pos = 40
        else:
            horizontal_pos = random.uniform(0, 240 - scale_factor)
            vertical_pos = random.uniform(0, 160 - scale_factor)

        return (scale_factor, horizontal_pos, vertical_pos)

    # superimpose the target (Blueno) onto the background
    def superimpose_target(self, background, scale_factor, horizontal_pos, vertical_pos):
        foreground = Image.open(
            my_path + "/target_original/blueno_cropped.png")

        scale_position = (scale_factor, scale_factor)

        foreground.thumbnail(scale_position)

        align_position = (int(horizontal_pos), int(vertical_pos))

        background.paste(foreground, align_position, foreground)
        return background

    # generate a mask of the processed image
    def generate_mask(self, thresh, scale_factor, horizontal_pos,
                      vertical_pos):
        background = Image.new('RGB', (240, 160), (255, 255, 255))
        img = self.superimpose_target(
            background=background, scale_factor=scale_factor, horizontal_pos=horizontal_pos,
            vertical_pos=vertical_pos)

        def fn(x): return 255 if x < thresh else 0

        mask = img.convert('L').point(fn, mode='1')

        return mask


ProcessDataset(src_path=my_path + "\\background_images\\images",
               train_image_path=my_path + "\\data\\train_images",
               train_mask_path=my_path + "\\data\\train_masks",
               test_image_path=my_path + "\\data\\test_images",
               test_mask_path=my_path + "\\data\\test_masks",
               restricted=False)
