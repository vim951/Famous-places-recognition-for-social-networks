import wget

a,b = 72,166

print('Beginning file downloads with wget module')

for i in range(a,b+1):
    k = str(i)
    k = "0"*(3-len(k))+k
    print("\nDownloading tar n°" + k)
    wget.download('https://s3.amazonaws.com/google-landmark/train/images_' + k + '.tar', '/home/victor/images_' + k + '.tar')