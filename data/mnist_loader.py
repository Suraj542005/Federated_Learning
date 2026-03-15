import tensorflow as tf


def load_mnist():

    # Load dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize images (converting 0 to 1) to make it stable
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Add channel dimension for CNN
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    return x_train, y_train, x_test, y_test
