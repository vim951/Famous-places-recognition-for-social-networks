import os
from PIL import Image
import numpy as np
import wget
from tensorflow import keras

classes=['Noraduz Cemetery', 'Museum of Folk Architecture and Ethnography in Pyrohiv', 'Salève', 'Nieuwe Waterweg', 'Khotyn Fortress', 'Pešter', 'Niagara Falls', 'Haleakalā National Park', 'Feroz Shah Kotla', 'Catedral San Sebastián', 'Cochabamba', 'Grand Canyon', 'Golden Gate Bridge', 'Madrid Río', 'Kecharis', 'Mathura Museum', 'Hayravank monastery', 'Qutb Minar and its monuments', 'Delhi', 'Sofiyivsky Park', 'St. Lawrence', 'Toronto', 'Akkerman fortress', 'Edinburgh Castle', 'Eiffel Tower', 'Genoese fortress (Sudak)', 'Skopje Fortress', 'Masada', 'Matka Canyon', 'Pakistan Monument Islamabad', 'Purana Qila', 'Itmad-Ud-Daulahs Tomb', 'Faisal Mosque', 'Saint Holy Mother church of Yeghvard', 'Haghpat', 'Ogrodzieniec Castle', 'Grote of Onze-Lieve-Vrouwekerk (Breda)', 'York River State Park', 'Douthat State Park', 'St. Hripsime church in Vagharshapat', 'Lake Como', 'Ljubljana Castle', 'Victoria Memorial', 'Kolkata', 'Episcopal Diocese of Southwest Florida', 'Perperikon', 'Akbars Tomb', 'Harichavank', 'Jurassic Coast', 'Isa Khan Niyazis tomb', 'Nyhavn', 'Çufut Qale', 'Saiful Muluk Lake', 'SBB Historic']

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

def init(model_path):
    global model, labels
    model = keras.models.load_model(model_path)
    
def url_to_result(url):
    file_path = 'image.' + url.split('.')[-1]
    wget.download(url, file_path)
    X = np.array([preprocess_image_png(file_path)])
    Y = model.predict(X)
    os.remove(file_path)
    return Y

if __name__ == '__main__':
    path="https://upload.wikimedia.org/wikipedia/commons/c/c3/Saint_Holy_Mother_church_of_Yeghvard_01.jpg"
    init('./test_model')
    result = url_to_result(path)
    y=sorted(list(zip(classes, result[0])), key=lambda x: -x[1])
    print(dict((i+1, dict((['landmark', y[i][0]], ['proba', y[i][1]]))) for i in range(len(y))))
    
def trigger_lambda(event, context):
    return event['url'];
