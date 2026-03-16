from flask import Flask, render_template, jsonify, request
import json
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

MODEL_PATH = "../global_model.keras"
METRICS_FILE = "metrics.json"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/metrics")
def metrics():
    with open(METRICS_FILE, "r") as f:
        data = json.load(f)

    return jsonify(data)


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    img = Image.open(file)
    img = img.convert("L")
    img = img.resize((28,28))

    img = np.array(img)
    img = 255 - img
    img = img / 255.0
    img = img.reshape(1,28,28,1)

    model = tf.keras.models.load_model(MODEL_PATH)

    pred = model.predict(img)

    digit = int(np.argmax(pred))

    return jsonify({
        "prediction": digit
    })


if __name__ == "__main__":
    app.run(debug=True)

