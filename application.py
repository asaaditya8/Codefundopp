from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
from keras import backend as K
from keras.models import load_model,model_from_json
from keras.applications.resnet50 import preprocess_input
from shutil import copyfile
import json

app = Flask(__name__)
photos = UploadSet('photos',IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'data/'
configure_uploads(app, photos)
PATH = 'data/'

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        copyfile(PATH+filename,'static/'+filename)

        output = call_work(filename)
        ou1 = output
        if output.argmax(axis=-1) == 1:
        	output = "WildFire"
        else:
        	output = "Storm"
        return render_template('result.html',name = str('static/'+filename),ou = output,ou1=ou1)
    return render_template('./upload.html')


if __name__ == '__main__':
    app.run(debug=True)


def call_work(filename):
    arr = load_img(filename)
    K.clear_session()
    json_file = open('model/resnet_cnn_1_arch.json', 'r')
    loaded_model_json = json.load(json_file)
    json_file.close()
    loaded = model_from_json(loaded_model_json)
    # load weights into new model
    loaded.load_weights("model/resnet_cnn_1_weights.h5")
    # loaded = load_model('model/resnet2_cnn_1.h5')
    return loaded.predict(arr)

import numpy as np, pandas as pd
# import matplotlib.pyplot as plt
from PIL import Image


# np.set_printoptions(suppress=True, precision=1)

# ROOT = '/content/gdrive/My Drive/ml_summer18/Codefundopp/'

# PATH = '/content/gdrive/My Drive/ml_summer18/Codefundopp/data/'
# CSV_PATH = '/content/gdrive/My Drive/ml_summer18/Codefundopp/eo_nasa_urls.csv'
# FEATURES = '/content/gdrive/My Drive/ml_summer18/Codefundopp/nasnet_features.npy'
# BATCH_SIZE = 8

def open_pil_image(path):
    return Image.open(path)


def open_image(path):
    return np.array(Image.open(path), dtype='float32')


def resize_pil_image(Img, size):
    return Img.resize(size, Image.ANTIALIAS)


def resize_image(img, size):
    return np.array(resize_pil_image(Image.fromarray(img.astype('uint8')), size), dtype='float32')


def rescale_pil_image(Img, size):
    W, H = size
    w, h = Img.size
    factor = np.min([W/w, H/h])
    nw = int(w*factor)
    nh = int(h*factor)
    return resize_pil_image(Img, (nw,nh))


def rescale_image(img, size):
    return np.array(rescale_pil_image(Image.fromarray(img.astype('uint8')), size), dtype='float32')


def pad_image(img, size):
    wp, hp = size
    wp1 = int(np.ceil(wp/2)); wp2 = int(np.floor(wp/2))
    hp1 = int(np.ceil(hp/2)); hp2 = int(np.floor(hp/2))
    H, W = img.shape[:-1]
    w = W + wp
    h = H + hp
    new_img = np.zeros((h,w,3), dtype='float32')
    new_img[hp1:h-hp2, wp1:w-wp2, :] = img
    return new_img


def pad_pil_image(Img, size):
    return Image.fromarray(pad_image(np.array(Img, dtype='float32'), size).astype('uint8'))


def rescale_pad(img, size):
    img = rescale_image(img, size)
    W, H = size
    h, w = img.shape[:-1]
    return pad_image(img, (W-w, H-h))


def rescale_pad_pil(Img, size):
    return Image.fromarray(rescale_pad(np.array(Img, dtype='float32'), size).astype('uint8'))
  
  
def load_img(fname):
    img = open_image(PATH + fname)
    return rescale_pad(img, (224, 224)).reshape(1, 224, 224, 3)