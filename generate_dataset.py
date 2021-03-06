from PIL import Image
import os
from PIL import ImageFilter

my_path = os.path.dirname(__file__)


class ProcessDataset():
    def __init__(self, src_path, iter):
        # path of the folder containing the raw images
        # src_path = "/background_images/images"

        # path of the folder that will contain the modified image
        target_image = my_path + "/processed_images/images"
        target_mask = my_path + "/processed_images/masks"

        counter = 0

        for path in os.listdir(src_path):
            if iter == counter:
                break
            # path contains name of the image
            inputPath = os.path.join(src_path, path)
            # print('input_path: ', inputPath)

            # inputPath contains the full directory name
            img = Image.open(inputPath)

            full_image_path = os.path.join(target_image, path)
            # fullOutPath contains the path of the output

            # image that needs to be generated
            img.rotate(90).save(full_image_path)
            # img

            # print(full_image_path)
            counter += 1

    def superimpose_bleuno():
        background = Image.open("test1.png")

        foreground = Image.open("test2.png")

        background.paste(foreground, (0, 0), foreground)
        background.show()

    def superimpose_bleuno():
        background = Image.open("test1.png")

        foreground = Image.open("test2.png")

        background.paste(foreground, (0, 0), foreground)
        background.show()

    def generate_image_masks():
        # get the directory path of the current python file
        my_path = os.path.dirname(__file__)

        img = Image.open(
            my_path + '/pngtree-random-energy-wave-background-image_307670.jpg', 'r')
        img_w, img_h = img.size
        background = Image.new('RGBA', (320, 240), (255, 255, 255, 255))
        bg_w, bg_h = background.size
        # generate at multiple offsets

        offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
        background.paste(img, offset)
        background.save('out.png')

        ProcessDataset(my_path + "/background_images/images", 1)
