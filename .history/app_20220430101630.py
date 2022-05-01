from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("./model/")

app = Flask(__name__)

def predict(smile):
    
    pred = model.predict(smiles)
    return 1 if pred > 0.5 else 0

@app.route('/predict', methods=['POST'])
def infer_smile():
    pass

if __name__ == '__main__':
     app.run(debug=True, host='0.0.0.0')
