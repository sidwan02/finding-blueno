from PIL import Image
import os

# get the directory path of the current python file
my_path = os.path.dirname(__file__)

img = Image.open(
    my_path + '/pngtree-random-energy-wave-background-image_307670.jpg', 'r')
img_w, img_h = img.size
background = Image.new('RGBA', (1440, 900), (255, 255, 255, 255))
bg_w, bg_h = background.size
# generate at multiple offsets

offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
background.paste(img, offset)
background.save('out.png')
