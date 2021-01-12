## Libraries

from os import listdir
from os.path import isfile, join
from pathlib import Path

from shutil import copyfile

from PIL import Image
import numpy as np
import pandas as pd

from multiprocessing import Process

import random

## Constants

PREPROCESS_MODE = "png"
csv_db_path = 'train_clean.csv'
csv_labels_path = 'train_label_to_category.csv'
db_path = '/media/victor/Seagate Wireless/Datasets/MALIS-DB/JPG files'
preprocessed_db_path = '/media/victor/Seagate Wireless/Datasets/MALIS-DB/NPY files'

## Memory optim

class DB_Generator(keras.utils.Sequence) :
  
  def __init__(self, X, Y, batch_size) :
    self.X = X
    self.Y = Y
    self.batch_size = batch_size
    
  def __len__(self) :
    return (np.ceil(len(self.X) / float(self.batch_size))).astype(np.int)
  
  def __getitem__(self, idx) :
    batch_x = [id_to_np(i) for i in self.X[idx * self.batch_size : (idx+1) * self.batch_size]]
    batch_y = self.Y[idx * self.batch_size : (idx+1) * self.batch_size]
    return batch_x, batch_y

## DB reading

def id_to_np(i):
    try:
        return np.load(join(preprocessed_db_path, i+'.npy')).reshape(1,256,256,4)
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

def copy_DB(n, from_path, to_path):
    C,L = load_db_csv(n)
    t=1
    for c in C:
        print("Copying category " + str(t) + "/" + str(len(C)))
        t+=1
        for x in c[1].split(' '):
            src = from_path + '/' + str(x) + '.npy'
            dst = to_path + '/' + str(x) + '.npy'
            copyfile(src, dst)
    

## Preprocessing

def preprocess_image(image_path, size, bin_path):
    if PREPROCESS_MODE == "bw":
        return preprocess_image_bw(image_path, size, bin_path)
    elif PREPROCESS_MODE == "png":
        return preprocess_image_png(image_path, size, bin_path)

def preprocess_image_bw(image_path, size, bin_path):
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

def preprocess_image_png(image_path, size, bin_path):
    old_im = Image.open(image_path)
    (l,h) = old_im.size
    c = max(l,h)
    new_size = (c, c)
    old_size = (l, h)
    new_im = Image.new("RGBA", new_size)
    new_im.paste(old_im, ((new_size[0]-old_size[0])//2, (new_size[1]-old_size[1])//2))
    new_im_sized = new_im.resize((size,size))
    np_img_normal = np.float16(np.array(new_im_sized)/255)
    with open(bin_path, 'wb') as f:
        np.save(f, np_img_normal)
    return np_img_normal

def preprocess_database(from_path=None, to_path=None):
    
    from_path = db_path if from_path==None else from_path
    to_path = preprocessed_db_path if to_path==None else to_path
    
    print("Preprocessing " + from_path)
    
    to_dir = Path(to_path)
    if not to_dir.exists():
        to_dir.mkdir()
    
    for f in [f for f in listdir(from_path) if (isfile(join(from_path, f)) and not ".DS_Store" in f)]:
        preprocess_image(join(from_path, f), size, join(to_path, f).replace(".jpg", ".npy"))
    
    for d in [f for f in listdir(from_path) if not isfile(join(from_path, f))]:
        preprocess_database(join(from_path, d), join(to_path, d))

def preprocess_database_partial(C, L, n=100, size=256, from_path=None, to_path=None):
    
    from_path = db_path if from_path==None else from_path
    to_path = preprocessed_db_path if to_path==None else to_path
    
    print("Preprocessing " + from_path)
    
    t=1
    for c in C:
        print("Preprocessing category " + str(t) + "/" + str(len(C)))
        t+=1
        for x in c[1].split(' '):
            img_path = from_path + '/' + '/'.join([y for y in x[:3]]) + '/' + str(x) + '.jpg'
            npy_path = to_path + '/' + str(x) + '.npy'
            preprocess_image_png(img_path, size, npy_path)

def preprocess_database_partial_aws(n=100, size=256, from_path=None, to_path=None):
    
    from_path = db_path if from_path==None else from_path
    to_path = preprocessed_db_path if to_path==None else to_path
    
    C,L = load_db_csv(n, excluded=[])
    
    for i in range(500):
        download_tar(i)
        preprocess_database_partial(C, L, n, size, from_path, to_path)
    
    t=1
    for c in C:
        print("Preprocessing category " + str(t) + "/" + str(len(C)))
        t+=1
        for x in c[1].split(' '):
            img_path = from_path + '/' + '/'.join([y for y in x[:3]]) + '/' + str(x) + '.jpg'
            npy_path = to_path + '/' + str(x) + '.npy'
            preprocess_image_png(img_path, size, npy_path)

## NN aux functions

def joined_shuffle(X,Y):
    Z=list(zip(X,Y))
    random.shuffle(Z)
    X,Y=list(zip(*Z))
    return np.array(X),np.array(Y)
