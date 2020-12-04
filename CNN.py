##

import numpy as np

from database import load_db_csv , id_to_np, joined_shuffle

import keras
from keras.models import Sequential

import tensorflow as tf
import datetime

csv_db_path = '/Users/hugodanet/Downloads/train_clean.csv'
csv_labels_path = '/Users/hugodanet/Downloads/train_label_to_category.csv'
preprocessed_db_path = '/Users/hugodanet/Downloads/DOWNLOAD DATASET/ENTRY_DATA'

train_size = 6133
size = 100

##

model = Sequential([
    keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(100,100,1)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(5)
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

model.summary()

##

C,L = load_db_csv(5)
X,Y=[],[]

for i in range(5):
    for x in C[i][1].split(' '):
        if not id_to_np(x) is None:
            X.append(id_to_np(x))
            Y.append([i])
        
X,Y = joined_shuffle(X, Y)
#X=[id_to_np(x) for x in X]
#Y=[np.array([1 if i==y else 0 for i in range(5)]) for y in Y]

Xarr = np.array(X)
Yarr = np.array(Y)

##VISUALIZATION

##execute each time rm -rf ./logs/ and %tensorboard --logdir logs/fit

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

##TRAINING

epochs=100

#the shuffle parameter only shuffles the training data
history = model.fit(
    x=Xarr.reshape(train_size,100,100,1),
    y=Yarr,
    batch_size=128,
    epochs=epochs,
    verbose=1,
    callbacks=[tensorboard_callback],
    shuffle=True,
    initial_epoch=0,
    validation_split=0.1,
    max_queue_size=10,
    workers=2,
    use_multiprocessing=True,
)
