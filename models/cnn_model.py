import tensorflow as tf
from tensorflow.keras import layers, models

"""
    ( Flow chart)

Input Image (28x28x1)
        ↓
Conv Layer (32 filters)
        ↓
    MaxPool
        ↓
Conv Layer (64 filters)
        ↓
    MaxPool
        ↓
    Flatten
        ↓
  Dense (128)
        ↓
Output (10 digits)

"""


def create_model():
    model = models.Sequential([

        tf.keras.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),

        layers.Dense(10, activation="softmax")

    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

