import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras.models import *
from keras.optimizers import *
from keras.layers import *
from keras.callbacks import EarlyStopping
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sub_sampling import MultiLabels

np.set_printoptions(suppress=True, precision=1)

PATH = 'data/'
CSV_PATH = 'eo_nasa_urls.csv'
FEATURES = 'nasnet_features.npy'


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


def get_model():
    model = Sequential(
        [
            Dense(1500, input_shape=(1056,), activation='relu', use_bias=False),
            BatchNormalization(),
            Dense(1000, activation='relu', use_bias=False),
            BatchNormalization(),
            Dense(50, activation='relu', use_bias=False),
            BatchNormalization(),
            Dense(11, activation='softmax', use_bias=False)
        ],
        name='MLP'
    )

    model.compile(Adam(), 'categorical_crossentropy', ['accuracy'])
    return model


def plot_hist(hist):
    pd.DataFrame(hist.history).plot()
    plt.show()


def train_model(model, X_train, y_train):
    es = EarlyStopping(patience=2)
    hist = model.fit(X_train, y_train, batch_size=16, epochs=100, callbacks=[es], validation_split=0.2)
    model.save('weights/dense_1000_500_100_12__nonorm_nobias_subsmpl.h5')
    return hist


def test_model(model, X_test, y_test):
    print('\nTest result:', model.evaluate(X_test, y_test, verbose=0))
    pred = model.predict(X_test)
    y_pred = pred.argmax(axis=-1)
    y_true = y_test.argmax(axis=-1)
    print('\nTest accuracy:', accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='weighted'))



if __name__ == '__main__':

    y, mask = MultiLabels().get_y()

    X = MultiLabels().get_X(mask)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    model = get_model()
    hist = train_model(model, X_train, y_train)

    # model = load_model('weights/dense_500_50_13__1.h5')
    test_model(model, X_test, y_test)

    pred = model.predict(X_test[:15])
    for a,b in zip(pred.argmax(-1), y_test[:15].argmax(-1)):
        print(a, b)

    plot_hist(hist)