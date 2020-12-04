## Libraries

from os import listdir
from os.path import isfile, join
from pathlib import Path

from PIL import Image
import numpy as np
import pandas as pd

import random

## Constants

<<<<<<< HEAD
csv_db_path = '/Users/hugodanet/Downloads/train_clean.csv'
csv_labels_path = '/Users/hugodanet/Downloads/train_label_to_category.csv'
preprocessed_db_path = '/Users/hugodanet/Downloads/DOWNLOAD DATASET/ENTRY_DATA'
=======
csv_db_path = 'train_clean.csv'
csv_labels_path = 'train_label_to_category.csv'
preprocessed_db_path = 'data'
preprocessed_db_path = 'PDB'
>>>>>>> aed8b5003e5d3d84d846d57dfcbd0339ca1f6d1c

size=100

## DB reading

def id_to_np(i):
    try:
        return np.load(join(preprocessed_db_path, i+'.npy')).reshape(1,100,100,1)
    except:
        pass

def load_db_csv(n, excluded = [138982, 126637, 177870]):
    db_df = pd.read_csv(csv_db_path)
    labels_df = pd.read_csv(csv_labels_path)
    values = sorted(db_df.values, key=lambda x: -len(x[1]))
    i,R=0,[]
    while len(R)<n:
        if not values[i][0] in excluded:
            R.append(values[i])
        i+=1
    return R, [labels_df[labels_df['landmark_id']==r[0]].values[0][-1].split(':')[-1] for r in R]

def load_db(n):
    C,L = load_db_csv(n)
    R = [[id_to_np(i) for i in c[1].split(' ')] for c in C]
    return [[x for x in r if not x is None] for r in R], L
    

## Preprocessing

def preprocess_image(image_path, size, bin_path):
    img = Image.open(image_path)
    l,h = img.size
    c = min(l,h)
    dl, dh = (l-c)//2, (h-c)//2
    img_cropped = img.crop((dl, dh, c+dl, c+dh))
    img_resized = img_cropped.resize((size,size))
    np_img = np.array(img_resized)
    np_img_bw = np.mean(np_img, axis=2)
    np_img_normal = np.float16(np_img_bw/255)
    with open(bin_path, 'wb') as f:
        np.save(f, np_img_normal)
    return np_img_normal

def preprocess_database(from_path=None, to_path=None):
    
    from_path = db_path if from_path==None else from_path
    to_path = preprocessed_db_path #if to_path==None else to_path
    print("Preprocessing " + from_path)
    
    to_dir = Path(to_path)
    if not to_dir.exists():
        to_dir.mkdir()
    
    for f in [f for f in listdir(from_path) if (isfile(join(from_path, f)) and not ".DS_Store" in f)]:
        preprocess_image(join(from_path, f), size, join(to_path, f).replace(".jpg", ".npy"))
    
    for d in [f for f in listdir(from_path) if not isfile(join(from_path, f))]:
        preprocess_database(join(from_path, d), join(to_path, d))

## NN aux functions

def joined_suffle(X,Y):
    Z=list(zip(X,Y))
    random.shuffle(Z)
    return list(zip(*Z))