import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import tensorflow as tf
from tensorflow import keras

from heartdisease_model.config.core import config

def getModel(inputs, encoded_features):
    concat_layer = tf.keras.layers.concatenate(encoded_features)
    x = tf.keras.layers.Dense(128, activation="relu")(concat_layer)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)


    model = keras.Model(inputs, output)
    
    return model
