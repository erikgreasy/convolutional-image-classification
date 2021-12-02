import tensorflow as tf


def load_dataset(img_height, img_width, data_type='training'):
    """Loads training dataset from directory using keras function"""

    batch_size = 32

    return tf.keras.utils.image_dataset_from_directory(
        'train',
        validation_split=0.2,
        subset=data_type,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)