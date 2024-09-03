import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

import tensorflow as tf

from heartdisease_model import __version__ as _version
from heartdisease_model.config.core import config
from heartdisease_model.processing.data_manager import load_model
from heartdisease_model.processing.validation import validate_inputs


model_file_name = f"{config.app_config.saved_model_name}_{_version}.keras"
model = load_model()

def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    validated_data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    
    print(validated_data.shape)
    
    #validated_data = validated_data.reindex(columns = ['dteday', 'season', 'hr', 'holiday', 'weekday', 'workingday', 
    #                                                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'yr', 'mnth'])
    validated_data = validated_data.reindex(columns = config.model_config.features)
    
    print(validated_data.shape)
    print(validated_data.head())
    
    results = {"predictions": None, "version": _version, "errors": errors}
      
    if not errors:
        input_dict = {name: tf.convert_to_tensor([value]) for name, value in validated_data.items()}
        predictions = model.predict(input_dict)
        results = {"predictions": (100 * predictions[0][0]), "version": _version, "errors": errors}
        print(results)

    return results



if __name__ == "__main__":

    #data_in = {'dteday': ['2012-11-6'], 'season': ['winter'], 'hr': ['6pm'], 'holiday': ['No'], 'weekday': ['Tue'],
    #           'workingday': ['Yes'], 'weathersit': ['Clear'], 'temp': [16], 'atemp': [17.5], 'hum': [30], 'windspeed': [10]}
    
    sample = {
    "age": [60],
    "sex": [1],
    "cp": [1],
    "trestbps": [145],
    "chol": [233],
    "fbs": [1],
    "restecg": [2],
    "thalach": [150],
    "exang": [0],
    "oldpeak": [2.3],
    "slope": [3],
    "ca": [0],
    "thal": [2],}


    make_prediction(input_data = sample)