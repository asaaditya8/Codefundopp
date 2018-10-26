import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=1)

PATH = 'data/'
CSV_PATH = 'eo_nasa_urls.csv'
FEATURES = 'nasnet_features.npy'

class SingleLabels():
    def __init__(self):
        pass

    def get_y(self):
        df = pd.read_csv(CSV_PATH, index_col=0)
        df_arr = np.array(df)[:, 1:]

        fnames = [int(f.split('.')[0]) for f in sorted(os.listdir(PATH))]
        # print(len(fnames))

        y = df_arr[fnames].astype('int64')
        print('All image labels:', y.shape)

        print(len([n for n in y.sum(axis=-1) if n == 1]), 'Images with single labels')

        f_idxs = [i for i, n in enumerate(y.sum(axis=-1)) if n == 1]

        return y[f_idxs], f_idxs

    def get_X(self, idxs):
        features = np.load(FEATURES)
        X = features[idxs]
        X = X.reshape(-1, 7, 7, 1056).mean(axis=1).mean(axis=1)
        X = X / np.linalg.norm(X, axis=-1, keepdims=True)
        print(X.shape)
        # X = PCA(n_components=1446).fit_transform(X)
        # print(X.shape)
        return X


class MultiLabels():
    def __init__(self):
        pass

    def get_y(self):
        np.random.seed(0)

        df = pd.read_csv(CSV_PATH, index_col=0)
        df_arr = np.array(df)[:, 1:]

        fnames = [int(f.split('.')[0]) for f in sorted(os.listdir(PATH))]
        # print(len(fnames))

        df_arr = df_arr[fnames].astype('int64')

        # print('All image labels:', df_arr.shape)

        # print('No of images in each class:', df_arr.sum(axis=0))

        # print('Mean:', np.mean(df_arr.sum(axis=0)), ' Median:', np.median(df_arr.sum(axis=0)))

        MEAN = int(np.mean(df_arr.sum(axis=0)))

        mask = [True]*len(df_arr)
        num_wildfire = 497 - MEAN
        num_storm = 430 - MEAN
        wildfire_counter =0
        storm_counter = 0
        for i in range(len(df_arr)):
            if df_arr[i][-1] == 1 and wildfire_counter < num_wildfire:
                mask[i] = False
                wildfire_counter += 1
            if df_arr[i][-6] == 1 and storm_counter < num_storm:
                mask[i] = False
                storm_counter += 1
            if sum(df_arr[i]) > 1:
                mask[i] = False

        # df_arr = np.delete(df_arr, [0, 1, 2 ,3 ,4 ,5, 6, 8, 9, 10, 11], axis=1)
        # mask = [not x for x in mask]
        df_arr = np.delete(df_arr, [2, 4], axis=1)

        df_arr = df_arr[mask, :]
        # print(df_arr.shape)


        # print('No of images in each class:', df_arr.sum(axis=0))
        # print('Mean:', np.mean(df_arr.sum(axis=0)), ' Median:', np.median(df_arr.sum(axis=0)))

        # print('\nThe distributions is now alomst symmetric because mean is very close to median')


        return df_arr, mask

    def get_X(self, idxs):
        features = np.load(FEATURES)
        X = features[idxs]
        X = X.reshape(-1, 7, 7, 1056).mean(axis=1).mean(axis=1)
        X = X / np.linalg.norm(X, axis=-1, keepdims=True)
        print(X.shape)
        # X = PCA(n_components=1446).fit_transform(X)
        # print(X.shape)
        return X

if __name__ == '__main__':
    y, mask = MultiLabels().get_y()
    X = MultiLabels().get_X(mask)
    print(y.shape, X.shape)