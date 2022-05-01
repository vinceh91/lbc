from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
from feature_extractor import fingerprint_features
import numpy as np

model = keras.models.load_model("./models/first_model_oversamplig.h5")

app = Flask(__name__)
output={}

def predict(smile):
    input = np.array(fingerprint_features(smile))[np.newaxis, :]
    pred = model.predict(input)
    return 1 if pred[0][0] > 0.5 else 0

@app.route('/predict', methods=["POST"])
def infer_smile():
    if request.method == "POST":
        smile = request.form["q"]
        p1 = predict(smile)
        output["P1"] = p1
        return jsonify(output)

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'

if __name__ == "__main__":
     app.run()
