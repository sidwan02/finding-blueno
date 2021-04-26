from PIL import Image
import os
from PIL import ImageFilter

my_path = os.path.dirname(__file__)

# step 1


class ProcessDataset():
    def __init__(self, src_path, target_image_path, target_mask_path, predict):
        self.src_path = src_path
        self.target_image_path = target_image_path
        self.target_mask_path = target_mask_path
        self.predict = predict
        # path of the folder containing the raw images
        # src_path = "/background_images/images"

        # path of the folder that will contain the modified image

        self.process_images_in_folder()

    def process_images_in_folder(self):
        for path in os.listdir(self.src_path):
            if predict == False:
                # if iter == counter:
                #     break
                # path contains name of the image
                inputPath = os.path.join(self.src_path, path)
                # print('input_path: ', inputPath)

                # inputPath contains the full directory name
                img = Image.open(inputPath)

                img = img.resize((160, 240))

                processed_image_path = os.path.join(
                    self.target_image_path, 'processed_' + path)
                mask_path = os.path.join(self.target_mask_path, 'mask_' + path)
                # fullOutPath contains the path of the output

                # image that needs to be generated
                # img.rotate(90).save(full_image_path)
                processed_img = self.superimpose_bleuno(img)
                processed_img.save(processed_image_path)

            mask = self.generate_mask(255)

            mask.save(mask_path)

            # img

            # print(full_image_path)

    def superimpose_bleuno(self, background):
        foreground = Image.open(
            my_path + "/blueno_original/blueno_cropped.png")

        foreground.thumbnail((100, 100))

        background.paste(foreground, (100, 80), foreground)
        return background
        # background.show()

    def generate_mask(self, thresh):
        # # processed_image = Image.open(
        # #     my_path + '/pngtree-random-energy-wave-background-image_307670.jpg', 'r')
        # img_w, img_h = processed_image.size
        # mask = Image.new('RGBA', (160, 240), (255, 255, 255, 255))
        # bg_w, bg_h = mask.size
        # # generate at multiple offsets

        background = Image.new('RGB', (160, 240), (255, 255, 255))
        img = self.superimpose_bleuno(background)

        # offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
        # mask.paste(processed_image, offset)
        def fn(x): return 255 if x < thresh else 0

        # img.convert('L').point(fn, mode='1').save('foo.png')
        mask = img.convert('L').point(fn, mode='1')

        return mask
        # mask.save('out.png')


ProcessDataset(my_path + "/background_images/images", target_image_path=my_path +
               "/processed_images/images", target_mask_path=my_path + "/processed_images/masks", False)
