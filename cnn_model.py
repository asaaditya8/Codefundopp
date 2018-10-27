import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras.models import *
from keras.optimizers import *
from keras.layers import *
from keras.activations import *
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sub_sampling import MultiLabels
from extraction import get_batch

np.set_printoptions(suppress=True, precision=1)

PATH = 'data/'
CSV_PATH = 'eo_nasa_urls.csv'
FEATURES = 'nasnet_features.npy'
BATCH_SIZE = 8


def get_y():
    df = pd.read_csv(CSV_PATH, index_col=0)
    df_arr = np.array(df)[:, 1:]

    fnames = [int(f.split('.')[0]) for f in sorted(os.listdir(PATH))]
    # print(len(fnames))

    y = df_arr[fnames].astype('int64')
    print(y.shape)
    return y


def get_X():
    features = np.load(FEATURES)
    X = features
    X = features.reshape(-1, 7, 7, 1056).mean(axis=1).mean(axis=1)
    X = X / np.linalg.norm(X, axis=-1, keepdims=True)
    print(X.shape)
    # X = PCA(n_components=1446).fit_transform(X)
    # print(X.shape)
    return X


def get_block(f, k, strides):
    block = [
        Conv2D(f, k, strides=1, padding='same', use_bias=False),
        BatchNormalization(),
        ELU(),
        # Conv2D(f//2, 1, strides=strides, padding='same', use_bias=False),
        # BatchNormalization(),
        # ELU()
    ]
    if strides == 2:
        block.append(MaxPooling2D())
    return block


def use_block(x, block):
    for b in block:
        x = b(x)
    return x


def get_model():

    setting = [
        (32, 3, 2, 1),
        (64, 3, 1, 1),
        (64, 3, 2, 1),
        (128, 3, 1, 2),
        (128, 3, 2, 1),
        (256, 3, 1, 2),
        (256, 3, 2, 1),
        (512, 3, 1, 3),
        (512, 3, 2, 1),
    ]

    block_list = []
    for item in setting:
        for n in range(item[-1]):
            block_list.append(get_block(*item[:3]))

    inp = Input((224, 224, 3))
    x = BatchNormalization()(inp)

    for block in block_list:
        # for i in range(item[-1]):
        x = use_block(x, block)

    x = Average()([GlobalMaxPooling2D()(x), GlobalAveragePooling2D()(x)])
    x = Dense(50, activation='relu', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Dense(2, activation='softmax', use_bias=False)(x)

    model = Model(inp, x)

    model.compile(Adam(), 'categorical_crossentropy', ['accuracy'])
    return model


def plot_hist(hist):
    pd.DataFrame(hist.history).plot()
    plt.show()


def data_generator(X_train, y_train, batch_size):
    while True:
        index = np.random.choice(len(X_train), batch_size)
        yield X_train[index], y_train[index]

def train_model(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1)

    gen = data_generator(X_train, y_train, batch_size=BATCH_SIZE)
    steps_per_epoch = len(X_train) // BATCH_SIZE

    val_gen = data_generator(X_val, y_val, batch_size=BATCH_SIZE)
    val_steps = len(X_val) // BATCH_SIZE

    es = EarlyStopping(patience=6)
    hist = model.fit_generator(gen, steps_per_epoch, epochs=100, callbacks=[es], validation_data=val_gen, validation_steps=val_steps)
    model.save('weights/custom_sep_cnn_2.h5')
    return hist


def test_model(model, X_test, y_test):
    print('\nTest result:', model.evaluate(X_test, y_test, verbose=0))
    pred = model.predict(X_test)
    y_pred = pred.argmax(axis=-1)
    y_true = y_test.argmax(axis=-1)
    print('\nTest accuracy:', accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='weighted'))


def get_img(mask):
    fnames = np.array(os.listdir(PATH))
    fnames = fnames[mask]
    n = len(fnames)

    X = get_batch(fnames)
    print(X.shape)
    return X


if __name__ == '__main__':

    y, mask = MultiLabels().get_y()

    X = get_img(mask)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    model = get_model()
    # model = load_model('weights/custom_sep_cnn_1.h5')
    hist = train_model(model, X_train, y_train)

    # model = load_model('weights/dense_500_50_13__1.h5')
    test_model(model, X_test, y_test)

    pred = model.predict(X_test[:15])
    for a,b in zip(pred.argmax(-1), y_test[:15].argmax(-1)):
        print(a, b)

    plot_hist(hist)