import wget

def download_csv():
    print("Downloading train_clean.csv")
    wget.download('https://s3.amazonaws.com/google-landmark/metadata/train_clean.csv', 'train_clean.csv')
    print("Downloading train_label_to_category.csv")
    wget.download('https://s3.amazonaws.com/google-landmark/metadata/train_label_to_category.csv', 'train_label_to_category.csv')
    print("Done downloading")

def download_pdb():
    pass