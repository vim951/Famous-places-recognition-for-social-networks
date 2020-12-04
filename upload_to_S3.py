import boto3
import botocore
import tarfile

print('\033[96m------------------HELLO Victoria and Huguette-------------------\033[0m')

s3 = boto3.resource('s3')

#Download file from S3 bucket of dataset, where filename has the format images_XXX.tar
def downloadTar(filename):
    try:
        s3.Bucket('google-landmark').download_file('train/'+filename, 'tmp.tar')
        print('tar downloaded successfully')
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
        

#Upload .zip of around 250MO of .npy files to S3
def uploadNPY(filename):
    print('\033[96m------------------ UPLOAD ' + filename + ' -------------------\033[0m')

    s3_client = boto3.client('s3')
    bucket_size=sum([object.size for object in boto3.resource('s3').Bucket('cassettefbisurveillancevan').objects.all()])
    print(bucket_size)
    if(bucket_size>3.5e9):
    	#Thers is an error if we try to upload too many data to our S3 bucket, to prevent cost issues
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

