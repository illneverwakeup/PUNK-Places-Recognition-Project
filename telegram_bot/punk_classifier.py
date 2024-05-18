import logging

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

IMG_SHAPE = 224

def normalize(image):
    image /= 255
    return image

classes = ['Desyatki',
            'Dvadcatki',
            'Shayba',
            'fizfuck',
            'himfuck',
            'matmekh',
            'pmpu'
]
num_classes = len(classes)

checkpoint_path = "trained_models/ckpt/checkpoint.model.keras"

model = tf.keras.models.load_model(checkpoint_path)


def make_prediction(image_path):
    img = tf.keras.utils.load_img(
        image_path, target_size=(IMG_SHAPE, IMG_SHAPE)
    )
    img = normalize(tf.keras.utils.img_to_array(img))
    img = tf.expand_dims(img, 0)
    pred = model.predict(img)
    print(pred)
    print(np.argmax(pred[0]))
    return classes[np.argmax(pred[0])], max(pred[0])
