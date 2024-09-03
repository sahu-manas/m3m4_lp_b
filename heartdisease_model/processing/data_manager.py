import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
from pathlib import Path

import joblib
import pandas as pd

import tensorflow as tf

from heartdisease_model import __version__ as _version
from heartdisease_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe


def getModelCheckpoint() -> None:
    # Prepare versioned save file name
    save_file_name = f"{config.app_config.saved_model_name}_{_version}.keras"
    save_path = TRAINED_MODEL_DIR / save_file_name
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    
    return model_checkpoint_callback

def load_model():
    """Load a persisted pipeline."""
    save_file_name = f"{config.app_config.saved_model_name}_{_version}.keras"
    save_path = TRAINED_MODEL_DIR / save_file_name
    
    model = tf.keras.models.load_model(save_path)
    return model


def remove_old_model() -> None:
    do_not_delete = ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
