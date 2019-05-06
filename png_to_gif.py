import imageio
import numpy as np
import os
import skimage

if __name__ == '__main__':
    dir_name = "images"
    max_iter = 2000
    images = []
    for i in range(0, max_iter, 10):
        filename = os.path.join(dir_name, str(i) + ".png")
        image = imageio.imread(filename)

        # new_shape = (int(image.shape[0]*0.5), int(image.shape[1]*0.5), image.shape[2])
        # image = skimage.transform.resize(image, new_shape)

        images.append(image)

    # some post processing as matplotlib may save them in different (width) sizes depending on if there are negatives to be had or not
    max_size = max([image.shape for image in images], key=lambda x: x[1])

    for i in range(len(images)):
        image = images[i]
        if image.shape[1] != max_size[1]:
            ones = 255 * np.ones((image.shape[0], max_size[1] - image.shape[1], image.shape[2]), dtype=np.uint8)
            image = np.append(ones, image, axis=1)
            images[i] = image


    imageio.mimsave('readme_images/sin.gif', images)
