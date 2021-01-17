import os
from tensorflow import keras


def load_db_csv(n, excluded = [138982, 126637, 177870]):
    db_df = pd.read_csv(csv_db_path)
    labels_df = pd.read_csv(csv_labels_path)
    values = sorted(db_df.values, key=lambda x: -len(x[1]))
    i,R=0,[]
    while len(R)<n:
        if not values[i][0] in excluded:
            R.append(values[i])
        i+=1
    return [labels_df[labels_df['landmark_id']==r[0]].values[0][-1].split(':')[-1] for r in R]

def preprocess_image_png(image_path, size=256):
    old_im = Image.open(image_path)
    (l,h) = old_im.size
    c = max(l,h)
    new_size = (c, c)
    old_size = (l, h)
    new_im = Image.new("RGBA", new_size)
    new_im.paste(old_im, ((new_size[0]-old_size[0])//2, (new_size[1]-old_size[1])//2))
    new_im_sized = new_im.resize((size,size))
    np_img_normal = np.float16(np.array(new_im_sized)/255)
    return np_img_normal

def init(model_path, csv_path, nb_cat):
    global model, labels
    model = keras.models.load_model(model_path)
    labels = load_db_csv(nb_cat)

def url_to_result(url):
    file_path = 'image.' + url.split('.')[-1]
    wget.download(url, file_path)
    X = np.array([preprocess_image_png(file_path)])
    Y = model.predict(X)
    os.remove(path)
    return Y