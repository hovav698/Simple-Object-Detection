import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

from utils import resize_image
from config import num_classes, num_backgrounds, img_dim, batch_size


# the function will generate random image
# it will return the image, the object location parameters, the object class and
# if object exist in the image or not

def generate_img(animals, backgrounds):
    # an object will appear in probability of 5/6
    appear = (np.random.random() < num_classes / num_classes + 1)

    # choose random background
    bg_idx = np.random.randint(num_backgrounds)

    # choose random class
    animal_class = np.random.randint(num_classes)
    animal = animals[animal_class]

    loc_params = np.zeros(4)

    gen_img = backgrounds[bg_idx].copy()

    if appear:

        # add data augmentation to the image - randomly flip and resize the objects
        scale = np.random.random() * 0.2 + 0.8
        animal = resize_image(animal, scale)

        if np.random.random() < 0.5:
            animal = np.fliplr(animal)

        animal_H, animal_W = animal.shape[:-1]

        backgrounds_dims = backgrounds[bg_idx].shape[:-1]

        # randomly choose the object's coordinates
        left_coord = np.random.randint(0, backgrounds_dims[1] - animal_W)
        right_coord = left_coord + animal_W

        buttom_coord = np.random.randint(0, backgrounds_dims[0] - animal_H)
        upper_coord = buttom_coord + animal_H

        # Take a slice of the background for those coordinate.
        bg_slice = gen_img[buttom_coord:upper_coord, left_coord:right_coord]

        # Multiple the background slice with the mask.
        # It will change the pixel value to zero in the place where the animal is located,
        # and won't be affected in other parts because the mask value is 1 in those areas
        mask = (animal[:, :, 3] == 0)
        bg_slice = np.expand_dims(mask, -1) * bg_slice
        bg_slice += animal[:, :, :3]
        gen_img[buttom_coord:upper_coord, left_coord:right_coord] = bg_slice

        # normal the coordinates between 0 and 1
        loc_params[0] = left_coord / backgrounds_dims[1]
        loc_params[1] = buttom_coord / backgrounds_dims[0]
        loc_params[2] = (right_coord - left_coord) / backgrounds_dims[1]
        loc_params[3] = (upper_coord - buttom_coord) / backgrounds_dims[0]

    return gen_img / 255.0, loc_params, appear, animal_class


# The data generator will create batchs of data. It will be used in the model training
def data_generator(animals, backgrounds, batch_size=batch_size):
    for _ in range(40):

        img_batch = np.zeros((batch_size, img_dim, img_dim, 3))
        target_batch = np.zeros((batch_size, 6))

        for i in range(batch_size):
            gen_img, loc_params, appear, animal_class = generate_img(animals, backgrounds)

            gen_img = resize(gen_img, (img_dim, img_dim), preserve_range=True)
            img_batch[i] = gen_img

            target_arr = np.concatenate([loc_params, [animal_class], [appear]])

            target_batch[i] = target_arr

        yield img_batch, target_batch
