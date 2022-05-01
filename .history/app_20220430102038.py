from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("./model/")

app = Flask(__name__)
output={}

def predict(smile):
    pred = model.predict(smile)
    return 1 if pred > 0.5 else 0

@app.route('/predict', methods=["POST"])
def infer_smile():
    if request.method == "POST":
        smile = request.form["q"]
        p1 = predict(smile)
        output["P1"] = p1
        return jsonify(output)

if __name__ == "__main__":
     app.run(debug=True, host="0.0.0.0")
