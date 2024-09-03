from typing import List
import sys
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.layers import IntegerLookup# Used in Feature processing
from keras.layers import Normalization     # Used in Feature processing
from keras.layers import StringLookup

from heartdisease_model.config.core import config

def encode_categorical_feature(feature, name, dataset, is_string):

    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature

def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature

def getInputsAndLookupLayers(dataset):
    all_inputs = []
    all_encoded_features = []
    # Integer categorical features
    for col in config.model_config.categorical_features_numerical:
        col_input = keras.Input(shape=(1,), name=col, dtype="int64")
        col_encoded = encode_categorical_feature(col_input, col, dataset, False)
        all_inputs.append(col_input)
        all_encoded_features.append(col_encoded)
    
    for col in config.model_config.numerical_features:
        col_input = keras.Input(shape=(1,), name=col)
        col_encoded = encode_numerical_feature(col_input, col, dataset)
        all_inputs.append(col_input)
        all_encoded_features.append(col_encoded)
    
    return all_inputs, all_encoded_features
