## Libraries

import boto3
import botocore
import tarfile

import sys
import os
import shutil

import database

## Constants

VERBOSE,DEBUG=False,False

#Those five values can be changed
n=50
size=100
global_file_per_new_tar=10**4
untar_dir='tmp1'
npy_dir='tmp2'

#Those two values should only be changed in case of errors
global_counter=0
global_new_tar_counter=0

## AWS specific



def downloadTar(filename):
    try:
        s3.Bucket('google-landmark').download_file('train/'+filename, 'tmp.tar')
        print('tar downloaded successfully')
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

def uploadNPY(filename):
    print('\033[96m------------------ UPLOAD ' + filename + ' -------------------\033[0m')

    s3_client = boto3.client('s3',
    	aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
    	aws_secret_access_key=os.environ.get('AWS_SECRET_KEY')
    	)
    bucket_size=sum([object.size for object in s3.Bucket('cassettefbisurveillancevan').objects.all()])
    print(bucket_size)
    if(bucket_size>3.5e9):
        print('\033[96m------------------ BUCKET IS FULL -------------------\033[0m')
        return False

    print('')
    try:
        response = s3_client.upload_file(filename, 'cassettefbisurveillancevan', filename)
    except ClientError as e:
        print(e)
        logging.error(e)
        return False
    return True

## Non AWS specific

def send_database(path):
    if not uploadNPY(path):
        sys.exit(-1)

def get_tar_name():
    global global_new_tar_counter
    s="0"*(3-len(str(global_new_tar_counter)))+str(global_new_tar_counter)
    global_new_tar_counter+=1
    return s+'.tar'

def dir_to_tar(path, tar_name):
    with tarfile.open(tar_name, "w:gz") as tar_handle:
        for root, dirs, files in os.walk(path):
            for file in files:
                tar_handle.add(os.path.join(root, file))

def compress_send_wipe():
    if VERBOSE:
        print("compress_send_wipe() was called.")
    global global_counter
    global_counter=0
    database.preprocess_database()
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
    global global_counter
    tar.extract(tarinfo, untar_dir)
    global_counter += 1
    if global_counter==global_file_per_new_tar:
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

## Init

def init():
    global C,L,ids,s3
    s3 = boto3.resource('s3',
    	aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
    	aws_secret_access_key=os.environ.get('AWS_SECRET_KEY')
    )
    print("Initialising")
    C,_=database.load_db_csv(n)
    ids=[x for c in C for x in c[1].split(' ')]

def process_s3():
    init()
    for i in range(500):
        k = str(i)
        k = "0"*(3-len(k))+k
        print('Treating images_' + k + '.tar')
        downloadTar('images_' + k + '.tar')
        read_tar('tmp.tar')
    compress_send_wipe()

def process_local():
    init()
    for i in range(500):
        k = str(i)
        k = "0"*(3-len(k))+k
        print('Treating images_' + k + '.tar')
        read_tar('data/images_' + k + '.tar')
    compress_send_wipe()

if __name__ == "__main__":
    if sys.argv[1]=="local":
        process_local()
    elif sys.argv[1]=="s3":
        process_s3()
    else:
        print("Invalid syntax")
