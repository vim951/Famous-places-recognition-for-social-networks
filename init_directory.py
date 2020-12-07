import wget
import zipfile
import sys

def download_csv():
    try:
        print("Downloading train_clean.csv")
        wget.download('https://s3.amazonaws.com/google-landmark/metadata/train_clean.csv', 'train_clean.csv')
    except:
        print("train_clean.csv download failed")
    try:
        print("Downloading train_label_to_category.csv")
        wget.download('https://s3.amazonaws.com/google-landmark/metadata/train_label_to_category.csv', 'train_label_to_category.csv')
    except:
        print("train_label_to_category.csv download failed")
    print("Done downloading")

def download_pdb():
    try:
        print("Downloading PDB.zip")
        wget.download('https://cassettefbisurveillancevan.s3.eu-west-3.amazonaws.com/PDB.zip', 'PDB.zip')
    except:
        print("PDB.zip download failed")
    try:
        with zipfile.ZipFile('PDB.zip', 'r') as zip_ref:
            zip_ref.extractall()
    except:
        print("PDB.zip unzip failed")

if __name__ == "__main__":
    for i in range(1,len(sys.argv)):
        if sys.argv[i]=='csv':
            download_csv()
        elif sys.argv[i]=='pdb':
            download_pdb()
        else:
            print("Unknown parameter: " + str(sys.argv[i]))