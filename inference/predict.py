import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load trained global model
model = tf.keras.models.load_model("global_model.keras")

# Load MNIST test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
x_test = x_test / 255.0
x_test = x_test[..., None]

# Pick a sample image
image = x_test[2]
label = y_test[2]

# Predict
prediction = model.predict(image.reshape(1, 28, 28, 1))
digit = np.argmax(prediction)

# Show result
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Predicted: {digit} | Actual: {label}")
plt.axis("off")
plt.show()
