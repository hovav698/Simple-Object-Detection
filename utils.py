import numpy as np
import torch
from PIL import Image
from skimage.transform import resize
from glob import glob


# reformat the image for pytorch
def reformat_image(img):
    img = torch.tensor(img)
    img = img.permute(2, 0, 1)

    return torch.unsqueeze(img, 0)


# the function resize the image according to the scale
def resize_image(img, scale):
    img_dim = img.shape
    img_H, img_W = int(img_dim[0] * scale), int(img_dim[1] * scale)
    img = resize(img, (img_H, img_W), preserve_range=True).astype(np.uint8)

    return img


def load_data():
    # load the animal images
    pig = np.array(Image.open('data/animals/pig.png'))
    chimp = np.array(Image.open('data/animals/chimp.png'))
    chicken = np.array(Image.open('data/animals/chicken.png'))
    croc = np.array(Image.open('data/animals/croc.png'))
    tiger = np.array(Image.open('data/animals/tiger.png'))

    # resize the images to a similar scale
    chimp = resize_image(chimp, 0.2)
    tiger = resize_image(tiger, 1 / 13)
    chicken = resize_image(chicken, 1 / 15)
    croc = resize_image(croc, 1 / 10)
    pig = resize_image(pig, 1 / 15)

    # convenient to use later
    idx2animal = {0: "chimp", 1: "chicken", 2: "pig", 3: "tiger", 4: "croc"}
    animals = [chimp, chicken, pig, tiger, croc]

    # load and resize the background images

    backgrounds = []

    background_files = glob('data/backgrounds/*.jpg')
    resize_scale = [1 / 4, 1 / 2, 1 / 3, 1 / 4, 1 / 4]
    for i, f in enumerate(background_files):
        # Note: they may not all be the same size
        bg = np.array(Image.open(f))
        bg = resize_image(bg, resize_scale[i])
        bg = np.array(bg)
        backgrounds.append(bg)

    return animals, idx2animal, backgrounds
