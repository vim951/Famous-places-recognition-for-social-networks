##

import numpy as np
import pandas as pd
from os.path import join

import keras
from keras.models import Sequential

import tensorflow as tf
import datetime

csv_db_path = '/Users/hugodanet/Downloads/train_clean.csv'
csv_labels_path = '/Users/hugodanet/Downloads/train_label_to_category.csv'
preprocessed_db_path = '/Users/hugodanet/Downloads/DOWNLOAD DATASET/ENTRY_DATA'

train_size = 6133
image_dimension_x = 100
image_dimension_y = 100


def id_to_np(i):
    try:
        return np.load(join(preprocessed_db_path, i+'.npy')).reshape(1,image_dimension_x,image_dimension_y,1)
    except:
        pass

def load_db_csv(n):
    excluded = [138982, 126637, 177870]
    db_df = pd.read_csv(csv_db_path)
    labels_df = pd.read_csv(csv_labels_path)
    
    print(db_df)
    
    values = sorted(db_df.values, key=lambda x: -len(x[1]))
    i,R=0,[]
    while len(R)<n:
        if not values[i][0] in excluded:
            R.append(values[i])
        i+=1
    return R, [labels_df[labels_df['landmark_id']==r[0]].values[0][-1].split(':')[-1] for r in R]

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
              metrics=['accuracy'])

model.summary()

##

C,L = load_db_csv(5)
X,Y=[],[]

for i in range(5):
    for x in C[i][1].split(' '):
        if not id_to_np(x) is None:
            X.append(id_to_np(x))
            Y.append(i)
        
#X=[id_to_np(x) for x in X]
#Y=[np.array([1 if i==y else 0 for i in range(5)]) for y in Y]

Xarr = np.array(X)
Yarr = np.array(Y)


print(X)
print(Y)

##VISUALIZATION

##execute each time rm -rf ./logs/ and %tensorboard --logdir logs/fit

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

##TRAINING

epochs=5

history = model.fit(
    x=Xarr.reshape(train_size,image_dimension_x,image_dimension_y,1),
    y=Yarr,
    batch_size=50,
    epochs=epochs,
    verbose=1,
    callbacks=[tensorboard_callback],
    validation_split=0.0,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
)
