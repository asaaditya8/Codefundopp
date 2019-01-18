import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import azureml
from azureml.core import Workspace, Run
from classifier.create_ws import AZHelper


def trial():
    try:
        AZHelper.load_ws()
        print('Done')
    except:
        print('Failed')


def fresh_csv():
    CSV_PATH = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_urls.csv'
    df = pd.read_csv(CSV_PATH)

    # ADDING FILENAME TO THE CSV
    FILENAME_csv = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_urls_names.csv'
    df = df.rename(columns={'Unnamed: 0': 'filename'})
    df.to_csv(FILENAME_csv, index=False)

    # REMOVING MULTILABEL URLS
    SINGLE_CLASS_csv = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_urls_names_sing_cls.csv'
    df = pd.read_csv(FILENAME_csv)

    trash_idx = []
    for i in range(len(df)):
        if df.iloc[i, 2:].sum() != 1:
            trash_idx.append(i)

    df = df.drop(df.index[trash_idx])
    df.to_csv(SINGLE_CLASS_csv, index=False)

    # REMOVE CLASSES WITH SINGLE DIGIT IMAGE SIZE
    ClASS_13_csv = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_urls_names_13_sing_cls.csv'
    trash_cols = []
    for col in df.columns[2:]:
        if df[col].sum() < 10:
            trash_cols.append(col)

    df = df.drop(trash_cols, axis=1)
    df.to_csv(ClASS_13_csv, index=False)

    # REMOVING DISASTER COLUMNS WITH JUST CLASS INDEX
    FILE_CLS_csv = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_file_url_cls_outof_10.csv'
    disasters = df.columns[2:]
    cl_idxs = []
    for i in range(len(df)):
        cl_idxs.append(np.argmax(df.iloc[i, 2:].values))

    df['Class'] = pd.Series(cl_idxs, index=df.index)
    df = df.drop(disasters, axis=1)

    df.to_csv(FILE_CLS_csv, index=False)

if __name__ == '__main__':
    # check core SDK version number
    # print("Azure ML SDK Version: ", azureml.core.VERSION)

    FILE_CLS_csv = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_file_url_cls_outof_10.csv'
    df = pd.read_csv(FILE_CLS_csv)

