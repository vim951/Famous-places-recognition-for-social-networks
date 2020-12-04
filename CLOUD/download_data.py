import os

import boto3
import botocore

s3_ressource = boto3.resource('s3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_KEY')
)
def downloadTar(filename):
    try:
        print('Start download ' + filename)
        s3_ressource.Bucket('google-landmark').download_file('train/'+filename, '../data/'+filename)
        print(filename + 'downloaded successfully')
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print(filename + " does not exist in the bucket google-landmark")
        else:
            raise
    
def init():
    if not 'AWS_ACCESS_KEY' in os.environ:
        print('AWS_ACCESS_KEY must be in your environement')
        return(0)
    if not 'AWS_SECRET_KEY' in os.environ:
        print('AWS_SECRET_KEY must be in your environement')
        return(0)
    return(1)
    
if(init()):
    for i in range(10):
        fileNumber=s="0"*(3-len(str(i)))+str(i)
        downloadTar('images_' + fileNumber + '.tar')
