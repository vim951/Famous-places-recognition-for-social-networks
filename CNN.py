##

import numpy as np

from sklearn.utils import class_weight

from database import load_db_csv , id_to_np, joined_shuffle

import keras
from keras.models import Sequential
import tensorflow as tf

import datetime
import os
import shutil

##

csv_db_path = 'train_clean.csv'
csv_labels_path = 'train_label_to_category.csv'
preprocessed_db_path = 'PDB'

epochs=100
train_size = 36966
size = 100
categories=50

## Init

def tensorboard_init():
    print("Initializing directory")
    try:
        shutil.rmtree("logs")
    except:
        pass

## Data preparation

def getTrainingData():
    print("Generating training data")
    C,L = load_db_csv(categories)
    X,Y,W=[],[],[]
    
    for i in range(categories):
        for x in C[i][1].split(' '):
            if not id_to_np(x) is None:
                X.append(id_to_np(x))
                Y.append([i])
                W.append(i)
            
    X,Y = joined_shuffle(X, Y)
    
    class_weights = class_weight.compute_class_weight('balanced',np.unique(W),W)
    class_weight_dict = dict(enumerate(class_weights))
    
    return X,Y,class_weight_dict

## CNN architecture

def getCNN():
    print("Generating model")
    model = Sequential([
        keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(100,100,1)),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(categories)
    ])
    model.compile(optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['sparse_categorical_accuracy'])
    return model

##VISUALIZATION

#tensorboard --logdir logs/fit

##TRAINING

def train(model, X, Y, W):
    print("Training model")
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(
        x=X.reshape(train_size,100,100,1),
        y=Y,
        batch_size=128,
        epochs=epochs,
        verbose=2,
        callbacks=[tensorboard_callback],
        shuffle=True,
        initial_epoch=0,
        validation_split=0.1,
        max_queue_size=100,
        workers=4,
        use_multiprocessing=True,
        class_weight=W
    )
    return history

##

if __name__ == "__main__":
    tensorboard_init()
    X,Y,W = getTrainingData()
    model = getCNN()
    train(model, X, Y, W)