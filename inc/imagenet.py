import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.preprocessing import image


def imagenet():

    image_dir = Path('train')
    file_paths = list(image_dir.glob(r'**/*.jpg'))
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], file_paths))

    file_paths = pd.Series(file_paths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    image_df = pd.concat([file_paths, labels], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
    image_df.to_csv('train_fp.csv', index=False)


    rows = []
    model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

    for index, row in tqdm(image_df.iterrows()):
        full_path = row['Filepath']
        img = image.load_img(full_path, target_size=(96, 96))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = model.predict(x).ravel().tolist()

        features.insert(0, row['Filepath'])
        # print(type(features))
        # print(type(row))
        rows.append(features)

    col_names = [*range(0, len(features)-1)]
    col_names.insert(0, 'filename')

    train_df = pd.DataFrame(rows, columns=col_names)
    train_df.to_csv('train.csv', index=False)
    # for image in test_data:
    #
    #     features = model.predict().ravel().toList()
