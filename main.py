""" EXTERNAL """
import numpy as np
import tensorflow as tf

""" MINE """
from inc.helpers import die
from inc.transform_test_labels import transform_test_labels_to_indexes
from inc.confusion_matrix import draw_confusion_matrix
from inc.load_dataset import load_dataset
from inc.load_test_data import load_test_data
from inc.train_model import train_model
"""
ZDROJE:

https://www.tensorflow.org/tutorials/images/classification
https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
"""

img_height = 50
img_width = 50

train_ds = load_dataset(img_height, img_width)
val_ds = load_dataset(img_height, img_width, 'validation')
class_names = train_ds.class_names


# CONFIGURE FOR PERFORMANCE
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# TRAIN MODEL
model = train_model(train_ds, val_ds, class_names, img_height, img_width)

# LOAD MODEL
# model = keras.models.load_model('neuronka')


# PREDICT
test_imgs, test_labels = load_test_data(img_width, img_height, class_names)

predict_x = model.predict(test_imgs)
classes_x = np.argmax(predict_x, axis=1)

draw_confusion_matrix(classes_x, test_labels, class_names)
