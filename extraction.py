from concurrent.futures import ThreadPoolExecutor

from keras.applications.nasnet import NASNetMobile
from keras.models import *
from tqdm import tqdm

from data_utils import open_image, rescale_pad

PATH = 'data/'
BATCH_SIZE = 8


def get_batch(fnames):

    def load(fname):
        img = open_image(PATH + fname)
        return rescale_pad(img, (224, 224))

    with ThreadPoolExecutor(max_workers=5) as execr:
        res = execr.map(load, fnames)
        arr = [r for r in res]

    return np.stack(arr, axis=0)


def extract(model, batch_size=8):
    fnames = os.listdir(PATH)
    n = len(fnames)

    predictions = []
    for i in tqdm(range(batch_size, n, batch_size)):
        batch = get_batch(fnames[i-batch_size: i])
        predictions.append(model.predict(batch))

    if n % batch_size:
        batch = get_batch(fnames[n - (n % batch_size):])
        predictions.append(model.predict(batch))

    return np.concatenate(predictions, axis=0).reshape(n, -1)


def save_predictions():
    output_file = 'nasnet_features'
    model = NASNetMobile(include_top=False)
    predictions = extract(model, BATCH_SIZE)
    # norm_predictions = predictions / np.linalg.norm(predictions, axis=-1, keepdims=True)
    print(predictions.shape)
    np.save(output_file, predictions)

if __name__ == '__main__':
    pass