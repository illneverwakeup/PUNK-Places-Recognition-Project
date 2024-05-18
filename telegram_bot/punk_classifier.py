import logging

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

checkpoint_path = "trained_models/ckpt/checkpoint.model.keras"

model = tf.keras.models.Sequential([
    layers.InputLayer(shape=(IMG_SHAPE,IMG_SHAPE,3)),
    layers.Conv2D(64, (5, 5), padding="same", activation='relu'),
    layers.Conv2D(64, (5, 5), padding="same", activation='relu'),
    layers.MaxPool2D((3, 3), strides=2),
    layers.BatchNormalization(),
    layers.Conv2D(128, (5, 5), padding="same", activation='relu'),
    layers.Conv2D(128, (5, 5), padding="same", activation='relu'),
    layers.MaxPool2D((3, 3), strides=2),
    layers.BatchNormalization(),
    layers.Conv2D(64, (5, 5), padding="same", activation='relu'),
    layers.Conv2D(64, (5, 5), padding="same", activation='relu'),
    layers.MaxPool2D((3, 3), strides=2),
    layers.BatchNormalization(),
    layers.Conv2D(32, (5, 5), padding="same", activation='relu'),
    layers.Conv2D(32, (5, 5), padding="same", activation='relu'),
    layers.MaxPool2D((3, 3), strides=2),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

model.load_weights(checkpoint_path)


def make_prediction(image_path):
    img = tf.keras.utils.load_img(
        image_path, target_size=(IMG_SHAPE, IMG_SHAPE)
    )
    img = tf.keras.utils.img_to_array(img)
    img = tf.expand_dims(img, 0)
    pred = model.predict(img)
    print(pred)
    return classes[np.argmax(pred[0])], max(pred[0])

if __name__ == "__main__":
    image = 'image.png'
    print(make_prediction(image))
