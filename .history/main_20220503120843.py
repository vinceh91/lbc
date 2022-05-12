import pandas as pd
import numpy as np
import tensorflow as tf
from feature_extractor import fingerprint_features
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow import keras


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

BATCH_SIZE = 32
EPOCHS = 5

def oversampling(features: np.ndarray, labels: np.ndarray):
    bool_labels = labels != 0
    
    pos_features = features[bool_labels]
    neg_features = features[~bool_labels]
    pos_labels = labels[bool_labels]
    neg_labels = labels[~bool_labels]

    ids = np.arange(len(neg_features))
    choices = np.random.choice(ids, len(pos_features))
    res_neg_features = neg_features[choices]
    res_neg_labels = neg_labels[choices]

    resampled_features = np.concatenate([res_neg_features, pos_features], axis=0)
    resampled_labels = np.concatenate([res_neg_labels, pos_labels], axis=0)

    return resampled_features, resampled_labels

def get_features(arr):
    return np.array([fingerprint_features(s) for s in arr])

df_single_raw = pd.read_csv("dataset_single.csv")
train_df, test_df = train_test_split(df_single_raw, test_size=0.1)
train_df, val_df = train_test_split(train_df, test_size=0.2)

train_labels = np.array(train_df.pop('P1'))
val_labels = np.array(val_df.pop('P1'))
test_labels = np.array(test_df.pop('P1'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

resampled_train_features, resampled_train_labels = oversampling(train_features, train_labels)

extracted_train_features = get_features(resampled_train_features[:,1])
extracted_val_features = get_features(val_features[:,1])
extracted_test_features = get_features(test_features[:,1])

tf_train_dataset = tf.data.Dataset.from_tensor_slices((extracted_train_features, resampled_train_labels))
tf_val_dataset = tf.data.Dataset.from_tensor_slices((extracted_val_features, val_labels))
tf_test_dataset = tf.data.Dataset.from_tensor_slices((extracted_test_features, test_labels))

tf_train_dataset = tf_train_dataset.shuffle(5000).batch(BATCH_SIZE).prefetch(1)
tf_val_dataset = tf_val_dataset.batch(BATCH_SIZE).prefetch(1)
tf_test_dataset = tf_test_dataset.batch(BATCH_SIZE).prefetch(1)

def build_model_graph(metrics=METRICS, input_shape=None):
    model = keras.Sequential([
      keras.layers.Dense(
          2, activation='relu',
          input_shape=input_shape),
      keras.layers.Dense(1, activation='sigmoid')
  ])

    model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)
    return model

def train(train_dataset=tf_train_dataset, val_dataset=tf_val_dataset):

    # Define model
    model = build_model_graph(input_shape=(2048,))
    model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)
    model.save("./models/first_model_oversamplig.h5")

    # score = model.evaluate(x_test, y_test, verbose=0)

def evaluate(model_path="./models/first_model_oversamplig.h5", test_dataset=tf_test_dataset):
    model_loaded = keras.models.load_model(model_path)
    model_loaded.evaluate(test_dataset)



if __name__ == '__main__':
    main()