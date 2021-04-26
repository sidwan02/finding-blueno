from PIL import Image
import os
from PIL import ImageFilter
import random

my_path = os.path.dirname(__file__)


class ProcessDataset():
    def __init__(self, src_path, target_image_path, target_mask_path, restricted):
        self.src_path = src_path
        self.target_image_path = target_image_path
        self.target_mask_path = target_mask_path
        self.restricted = restricted

        for path in os.listdir(self.src_path):
            inputPath = os.path.join(self.src_path, path)

            # inputPath contains the full directory name
            img = Image.open(inputPath)

            img = img.resize((240, 160))

            processed_image_path = os.path.join(
                self.target_image_path, 'processed_' + path)
            mask_path = os.path.join(self.target_mask_path, 'mask_' + path)

            (scale_factor, horizontal_pos,
             vertical_pos) = self.get_img_superposition_state()
            processed_img = self.superimpose_bleuno(
                background=img, scale_factor=scale_factor, horizontal_pos=horizontal_pos,
                vertical_pos=vertical_pos)

            processed_img.save(processed_image_path)

            mask = self.generate_mask(thresh=255, scale_factor=scale_factor, horizontal_pos=horizontal_pos,
                                      vertical_pos=vertical_pos)

            mask.save(mask_path)
            # print("here")
            # break

    def get_img_superposition_state(self):
        if self.restricted:  # centered
            scale_factor = 100
        else:
            scale_factor = random.uniform(10, 160)
        # print(scale_factor)

        if self.restricted:  # monosize
            horizontal_pos = 70
            vertical_pos = 40
            # align_position = (70, 40)
        else:
            horizontal_pos = random.uniform(0, 240 - scale_factor)
            vertical_pos = random.uniform(0, 160 - scale_factor)
            # align_position = (int(horizontal_pos), int(vertical_pos))

        return (scale_factor, horizontal_pos, vertical_pos)

    def superimpose_bleuno(self, background, scale_factor, horizontal_pos, vertical_pos):
        foreground = Image.open(
            my_path + "/blueno_original/blueno_cropped.png")

        scale_position = (scale_factor, scale_factor)

        foreground.thumbnail(scale_position)

        align_position = (int(horizontal_pos), int(vertical_pos))
        # print(align_position)

        background.paste(foreground, align_position, foreground)
        # print(background)
        return background
        # background.show()

    def generate_mask(self, thresh, scale_factor, horizontal_pos,
                      vertical_pos):
        background = Image.new('RGB', (240, 160), (255, 255, 255))
        img = self.superimpose_bleuno(
            background=background, scale_factor=scale_factor, horizontal_pos=horizontal_pos,
            vertical_pos=vertical_pos)

        def fn(x): return 255 if x < thresh else 0

        mask = img.convert('L').point(fn, mode='1')

        return mask


ProcessDataset(src_path=my_path + "\\background_images\\images",
               target_image_path=my_path + "\\data\\train_images",
               target_mask_path=my_path + "\\data\\train_masks",
               restricted=False)
