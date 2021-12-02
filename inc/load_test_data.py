import os

import numpy as np
import tensorflow.keras.preprocessing.image as image

from inc.transform_test_labels import transform_test_labels_to_indexes


def load_test_data(img_width, img_height, class_names):
    """
    Walk through test dir and get all the images to array. Returns np array and test labels coded
    to integers tuple.
    """

    folder_path = 'test'
    test_labels = []
    test_imgs = []
    for subdir in os.listdir(folder_path):
        path = os.path.join(folder_path, subdir)

        for img_title in os.listdir(path):
            test_labels.append(subdir)
            img_path = os.path.join(path, img_title)

            img = image.load_img(img_path, target_size=(img_width, img_height))
            img = image.img_to_array(img)
            test_imgs.append(img)

    x_test = np.array(test_imgs)
    test_labels = transform_test_labels_to_indexes(class_names, test_labels)

    return x_test, test_labels
