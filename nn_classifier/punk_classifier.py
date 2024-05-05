import logging

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import layers

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

IMG_SHAPE = 224

def normalize_and_resize(image):
    image = tf.image.resize(image, (IMG_SHAPE, IMG_SHAPE))
    image /= 255
    return image

classes = ['Desyatki',
            'Dvadcatki',
            'fizfuck',
            'himfuck',
            'matmekh',
            'pmpu',
            'Shayba'
]
num_classes = len(classes)

checkpoint_path = "../trained_models/ckpt/checkpoint.model.keras"

model = tf.keras.models.load_model(checkpoint_path)


def make_prediction(image):
    img = np.reshape(normalize_and_resize(image), (1, IMG_SHAPE, IMG_SHAPE, 3))
    pred = model.predict(img)
    print(pred)
    return classes[np.argmax(pred)]

if __name__ == "__main__":
    image = cv2.imread('image.png')
    print(make_prediction(image))
