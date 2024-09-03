import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
import tensorflow as tf

from heartdisease_model.config.core import config
from heartdisease_model.processing.data_manager import load_dataset, getModelCheckpoint, remove_old_model
from heartdisease_model.processing.features import getInputsAndLookupLayers
from heartdisease_model.model import getModel

def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name = config.app_config.training_data_file)
    
    # divide train and test
    
    total_size = data.shape[0]
    print("Total Number of rows:",total_size)
    
    val_df = data.sample(frac=config.model_config.val_size, random_state=1337)
    train_df = data.drop(val_df.index)

    print(
      "Using %d samples for training and %d for validation"
      % (len(train_df), len(val_df))
    )

    
    #Converting into tensorflow dataset & creating batches
    train_ds = dataframe_to_dataset(train_df)
    val_ds = dataframe_to_dataset(val_df)
    
    for x, y in train_ds.take(1):
        print("Input:", x)
        print("Condition:", y)

    #Batch the datasets
    train_ds = train_ds.batch(config.model_config.batch_size)
    val_ds = val_ds.batch(config.model_config.batch_size)
    
    all_inputs, all_encoded_features = getInputsAndLookupLayers(train_ds)
    
    model = getModel(all_inputs, all_encoded_features)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = config.model_config.learning_rate)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryFocalCrossentropy(), metrics=["accuracy"])
    
    model_checkpoint_callback = getModelCheckpoint()
    
    remove_old_model()
    
    history = model.fit(train_ds, epochs=config.model_config.n_epochs, batch_size= config.model_config.batch_size, validation_data=val_ds, callbacks=[model_checkpoint_callback])

    train_loss, train_accuracy = model.evaluate(train_ds)
    val_loss, val_accuracy = model.evaluate(val_ds)

    print('Training Accuracy: ',train_accuracy)
    print('Validation Accuracy: ',val_accuracy)

def dataframe_to_dataset(dataframe):
    ## YOUR CODE HERE
    df = dataframe.copy()
    labels = df.pop(config.model_config.target)
    #features = {column: tf.convert_to_tensor(dataframe[column]) for column in dataframe}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.shuffle(buffer_size=len(df))
    return ds

if __name__ == "__main__":
    run_training()