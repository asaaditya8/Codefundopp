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
from sklearn.model_selection import train_test_split

np.set_printoptions(suppress=True, precision=1)

PATH = 'data/'
CSV_PATH = 'eo_nasa_urls.csv'
FEATURES = 'nasnet_features.npy'

if __name__ == '__main__':

    df = pd.read_csv(CSV_PATH, index_col=0)
    df_arr = np.array(df)[:, 1:]

    fnames = [int(f.split('.')[0]) for f in sorted(os.listdir(PATH))]
    # print(len(fnames))

    y = df_arr[fnames]
    print(y.shape)

    # print(df_arr[:5])

    features = np.load(FEATURES)
    X = features / np.linalg.norm(features, axis=-1, keepdims=True)
    print(X.shape)
    # X = PCA(n_components=1446).fit_transform(X)
    # print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    # svm = OneVsRestClassifier(LinearSVC())
    # svm.fit(X, y)
    model = Sequential(
        [
            Dense(50, input_shape=(51744,), activation='tanh'),
            Dense(150, activation='tanh'),
            Dense(13, activation='sigmoid')
        ],
        name='MLP'
    )

    model.compile(Adam(), 'categorical_crossentropy', ['accuracy'])

    es = EarlyStopping(patience=2)
    hist = model.fit(X_train, y_train, batch_size=128, epochs=100, callbacks=[es], validation_split=0.2)
    model.save('weights/dense_500_50_13__2.h5')

    # model = load_model('weights/dense_500_50_13__1.h5')
    print('Test result:', model.evaluate(X_test, y_test, verbose=0))

    pred = model.predict(X_test[:15])
    for a,b in zip(np.round(pred), y_test[:15]):
        print(a, b)

    pd.DataFrame(hist.history).plot()
    plt.show()