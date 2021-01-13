##

import numpy as np

from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from database import load_db_csv , id_to_np, joined_shuffle

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras import regularizers

import tensorflow as tf

import datetime
import os
import shutil
import itertools
import io

##

csv_db_path = 'train_clean.csv'
csv_labels_path = 'train_label_to_category.csv'

CATEGORIES=50
CONFUSION_PERIOD=10
BATCH_SIZE = 128
EPOCHS=100
IMG_SIZE = 256

train_size = 0

## Init

def tensorboard_init():
    global file_writer_cm
    print("Initializing directory")
    try:
        shutil.rmtree("logs")
    except:
        pass
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
    return tensorboard_callback, cm_callback

def tensorboard_init_from_disk():
    global file_writer_cm
    print("Initializing directory")
    try:
        shutil.rmtree("logs")
    except:
        pass
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix_from_disk)
    return tensorboard_callback, cm_callback

## Data preparation

def getTrainingData():
    print("Generating training data")
    global X,Y,L
    
    C,L = load_db_csv(CATEGORIES)
    X,Y,W=[],[],[]
    
    for i in range(CATEGORIES):
        print(i)
        for x in C[i][1].split(' '):
            if not id_to_np(x) is None:
                X.append(id_to_np(x))
                Y.append([i])
                W.append(i)
            
    X,Y = joined_shuffle(X, Y)
    
    class_weights = class_weight.compute_class_weight('balanced',np.unique(W),W)
    class_weight_dict = dict(enumerate(class_weights))
    
    global train_size
    train_size = len(Y)
    
    return X,Y,class_weight_dict

def get_training_data_from_disk():
    
    global L
    C,L = load_db_csv(CATEGORIES)
    X,Y,W=[],[],[]
    
    for i in range(CATEGORIES):
        print('Getting', i)
        for x in C[i][1].split(' '):
            if not id_to_np(x) is None:
                X.append(x)
                Y.append([i])
                W.append(i)
                
    print(X,Y)
            
    X,Y = joined_shuffle(X, Y)
    
    class_weights = class_weight.compute_class_weight('balanced',np.unique(W),W)
    class_weight_dict = dict(enumerate(class_weights))
    
    global train_size
    train_size = len(Y)
    
    return X,Y,class_weight_dict
    
## CNN architecture

def getCNN():
    print("Generating model")
    model = Sequential([
    
        Conv2D(32, 4, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-3), input_shape=(IMG_SIZE,IMG_SIZE,4)),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.5),
        
        Conv2D(64, 4, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.5),
        
        Conv2D(128, 4, padding='same', activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.5),
        
        Flatten(),
        
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-3)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(CATEGORIES, activation='softmax', kernel_regularizer=regularizers.l2(1e-3))
    ])
    model.compile(optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['sparse_categorical_accuracy'])
    return model

## Visualization

def plot_confusion_matrix(cm, class_names):
    
    figure = plt.figure(figsize=(64, 64))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def log_confusion_matrix(epoch, logs):
    if (epoch%CONFUSION_PERIOD == 0):
        test_pred_raw = model.predict(X.reshape(train_size,IMG_SIZE,IMG_SIZE,4))
        test_pred = np.argmax(test_pred_raw, axis=1)
        cm = confusion_matrix(Y, test_pred)
        figure = plot_confusion_matrix(cm, class_names=L)
        cm_image = plot_to_image(figure)
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

def log_confusion_matrix_from_disk(epoch, logs):
    if (epoch%CONFUSION_PERIOD == 0):
        test_pred_raw = model.predict_generator(generator=test_generator, use_multiprocessing=True)
        test_pred = np.argmax(test_pred_raw, axis=1)
        cm = confusion_matrix(validation_generator.classes, y_pred)
        figure = plot_confusion_matrix(cm, class_names=L)
        cm_image = plot_to_image(figure)
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

## Visualization

#tensorboard --logdir logs/fit

## Training

def train(model, X, Y, W, tensorboard_callback, cm_callback):
    print("Training model")
    history = model.fit(
        x=X.reshape(train_size,IMG_SIZE,IMG_SIZE,4),
        y=Y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2,
        callbacks=[tensorboard_callback, cm_callback],
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

class DB_Generator(keras.utils.Sequence) :
  
  def __init__(self, X, Y, batch_size) :
    self.X = X
    self.Y = Y
    self.batch_size = batch_size
    
  def __len__(self) :
    return (np.floor(len(self.X) / float(self.batch_size))).astype(np.int)
  
  def __getitem__(self, idx) :
    batch_x = np.array([id_to_np(i) for i in self.X[idx * self.batch_size : (idx+1) * self.batch_size]]).reshape(self.batch_size,IMG_SIZE,IMG_SIZE,4)
    batch_y = self.Y[idx * self.batch_size : (idx+1) * self.batch_size]
    return batch_x, batch_y

def train_from_disk(model, X, Y, W, tensorboard_callback, cm_callback):
    print("Training model from disk")
    training_batch_generator = DB_Generator(X[:9*len(X)//10], Y[:9*len(Y)//10], BATCH_SIZE)
    testing_batch_generator = DB_Generator(X[9*len(X)//10:], Y[9*len(Y)//10:], BATCH_SIZE)
    history = model.fit_generator(
        generator=training_batch_generator,
        epochs=EPOCHS,
        verbose=2,
        callbacks=[tensorboard_callback, cm_callback],
        shuffle=True,
        validation_data = testing_batch_generator,
        workers=4,
        use_multiprocessing=True,
        class_weight=W
    )
    return history

## Main

if __name__ == "__main__":
    tensorboard_callback, cm_callback = tensorboard_init_from_disk()
    X,Y,W = get_training_data_from_disk()
    model = getCNN()
    train_from_disk(model, X, Y, W, tensorboard_callback, cm_callback)
