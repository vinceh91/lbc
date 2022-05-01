from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

def prediction(smile):
    model = keras.models.load_model("./model/")
    pred = model.predict(smiles)
    return 1 if pred > 0.5 else 0 
