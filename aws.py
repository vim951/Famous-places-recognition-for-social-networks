## Libraries

import tarfile

import sys
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path
import shutil

from PIL import Image
import numpy as np
import pandas as pd

import upload_to_S3

## DEBUG

DEBUG=False
VERBOSE=False

## Constants

csv_db_path = 'train_clean.csv'
untar_dir='tmp1'
npy_dir='tmp2'

excluded = [138982, 126637, 177870]
size=100

global_c=0
global_max=10**4
global_id=0

## AWS specific

def send_database(path):
    if not upload_to_S3.uploadNPY(path):
        print("!"*20)
        sys.exit(-1)

def get_tar_name():
    global global_id
    s="0"*(3-len(str(global_id)))+str(global_id)
    global_id+=1
    return s+'.tar'

def dir_to_tar(path, tar_name):
    with tarfile.open(tar_name, "w:gz") as tar_handle:
        for root, dirs, files in os.walk(path):
            for file in files:
                tar_handle.add(os.path.join(root, file))

def compress_send_wipe():
    if VERBOSE:
        print("compress_send_wipe() was called.")
    global global_c
    global_c=0
    preprocess_database()
    path=get_tar_name()
    dir_to_tar(npy_dir, path)
    send_database(path)
    if DEBUG:
        sys.exit(2)
    else:
        os.remove(path)
        shutil.rmtree(npy_dir)
        shutil.rmtree(untar_dir)

def extract_image(tar, tarinfo):
    if VERBOSE:
        print("Extracting " + tarinfo.name)
    global global_c
    tar.extract(tarinfo, untar_dir)
    global_c += 1
    if global_c==global_max:
        compress_send_wipe()

def read_tar(path):
    if VERBOSE:
        print("Reading " + str(path))
    tar = tarfile.open(path)
    for tarinfo in tar:
        #print(tarinfo.name, "is", tarinfo.size, "bytes in size and is ", end="")
        if tarinfo.isreg():
            i = tarinfo.name.split('/')[-1].split('.')[0]
            if i in ids:
                extract_image(tar, tarinfo)
    tar.close()
    if DEBUG:
        sys.exit(1)
    else:
        os.remove(path)

## Not AWS specific

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
    
    from_path = untar_dir if from_path==None else from_path
    to_path = npy_dir #if to_path==None else to_path
    print("Preprocessing " + from_path)
    
    to_dir = Path(to_path)
    if not to_dir.exists():
        to_dir.mkdir()
    
    for f in [f for f in listdir(from_path) if (isfile(join(from_path, f)) and not ".npy" in f)]:
        preprocess_image(join(from_path, f), size, join(to_path, f).replace(".jpg", ".npy"))
    
    for d in [f for f in listdir(from_path) if not isfile(join(from_path, f))]:
        preprocess_database(join(from_path, d), join(to_path, d))

def load_db_csv(n):
    db_df = pd.read_csv(csv_db_path)
    values = sorted(db_df.values, key=lambda x: -len(x[1]))
    i,R=0,[]
    while len(R)<n:
        if not values[i][0] in excluded:
            R.append(values[i])
        i+=1
    return R

## Init

def init():
    global C,L,ids
    print("Initialising")
    C=load_db_csv(5)
    ids=[x for c in C for x in c[1].split(' ')]

def lets_go():
    init()
    for i in range(500):
        k = str(i)
        k = "0"*(3-len(k))+k
        print('Treating images_' + k + '.tar')
        upload_to_S3.downloadTar('images_' + k + '.tar')
        read_tar('tmp.tar')
    compress_send_wipe()

# if __name__ == "__main__":
#     lets_go()