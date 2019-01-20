import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
import os
import shutil

from classifier.create_ws import AZHelper
import math

CUR_IDX = 0

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
    ClASS_10_csv = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_urls_names_10_sing_cls.csv'
    trash_cols = []
    for col in df.columns[2:]:
        if df[col].sum() < 10:
            trash_cols.append(col)

    df = df.drop(trash_cols, axis=1)
    df.to_csv(ClASS_10_csv, index=False)

    # REMOVING DISASTER COLUMNS WITH JUST CLASS INDEX
    FILE_CLS_csv = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_file_url_cls_outof_10.csv'
    disasters = df.columns[2:]
    cl_idxs = []
    for i in range(len(df)):
        cl_idxs.append(np.argmax(df.iloc[i, 2:].values))

    df['Class'] = pd.Series(cl_idxs, index=df.index)
    df = df.drop(disasters, axis=1)

    df.to_csv(FILE_CLS_csv, index=False)

    # REMOVING RECORDS WITH NON EXISTENT IMAGES
    FILE_CSL_jpg = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_file_url_cls_10_jpg.csv'

    root = '/home/aaditya/PycharmProjects/Codefundopp/data'
    existing_filenames = os.listdir(root)
    existing_filenames = [int(f.split('.')[0]) for f in existing_filenames if f.split('.')[-1] == 'jpg']

    trash_rows = []
    for i, f in enumerate(df['filename']):
        if f not in existing_filenames:
            trash_rows.append(i)

    df = df.drop(df.index[trash_rows])
    df.to_csv(FILE_CSL_jpg, index=False)

    # REMOVING OTHER TYPES OF IMAGES
    FILE_CLS_cleaned = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_file_url_cls_10_cleaned.csv'
    bekar_csv = '/home/aaditya/PycharmProjects/Codefundopp/bekar_images.csv'

    bad_df = pd.read_csv(bekar_csv)['filename']
    unique_bad_names = bad_df['filename'].unique() # list of int

    trash_ids = []
    for i in range(len(df)):
        if df.iloc[i, 0] in unique_bad_names:
            trash_ids.append(i)

    df = df.drop(df.index[trash_ids])
    df.to_csv(FILE_CLS_cleaned, index=False)

    # REMOVING MANMADE AND TEMPERATURE EXTREME CLASS
    FILE_CLS_8 = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_file_url_cls_8_cleaned.csv'

    df = df.drop(df.index[df['Class'] == 2])
    df = df.drop(df.index[df['Class'] == 6])

    cls = df.pop('Class').values
    cls_map = {
        0: 0,
        1: 1,
        3: 2,
        4: 3,
        5: 4,
        7: 5,
        8: 6,
        9: 7
    }

    new_cls = []
    for c in cls:
        new_cls.append(cls_map[c])

    df['Class'] = new_cls
    df.to_csv(FILE_CLS_8, index=False)

def image_ops():
    # Copying wildfire images to new destination
    FILE_CLS_8 = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_file_url_cls_8_cleaned.csv'

    df = pd.read_csv(FILE_CLS_8)

    root = '/home/aaditya/PycharmProjects/Codefundopp/data'
    dst = '/home/aaditya/PycharmProjects/Codefundopp/data_wf/present'

    wildfire_records = df[df['Class'] == 7]

    for i in range(len(wildfire_records)):
        fname = wildfire_records.iloc[i, 0]
        fsrc = f'{root}/{fname}.jpg'
        fdst = f'{dst}/{fname}.jpg'
        shutil.copy(fsrc, fdst)

    #Renaming no wildfire screenshots
    root = '/home/aaditya/PycharmProjects/Codefundopp/data_wf/absent'

    count = 0
    for f in os.listdir(root):
        fsrc = f'{root}/{f}'
        fdest = f'{root}/nowf_{count}.png'
        shutil.move(fsrc, fdest)
        count += 1

    #Converting them to jpg
    for f in os.listdir(root):
        fsrc = f'{root}/{f}'
        fdst = f'{root}/{f.split(".")[0]}.jpg'
        img = Image.open(fsrc)
        img.save(fdst)

    #Removing png
    for f in os.listdir(root):
        if f.split('.')[-1] == 'png':
            fsrc = f'{root}/{f}'
            os.remove(fsrc)

    # Drawing bounding box for cropping all the images
    f = os.listdir(root)[0]
    img = Image.open(f'{root}/{f}')

    plt.figure(figsize=(14, 14))
    ax = plt.gca()
    ax.imshow(img)

    rec = patches.Rectangle((235,50), 1310, 850, edgecolor='red', fill=None)
    ax.add_patch(rec)
    plt.show()

    # Cropping all the images
    for f in os.listdir(root):
        fsrc = f'{root}/{f}'
        img = Image.open(fsrc)
        img = img.crop((235,50, 1545, 900))
        img.save(fsrc)

def plot_samples(cls, num):
    """
    Plots random samples of a class
    int cls: Class index
    int num: Number of samples to plot
    """
    FILE_CLS_cleaned = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_file_url_cls_10_jpg.csv'
    df = pd.read_csv(FILE_CLS_cleaned)

    classes = ['Dust and Haze', 'Floods', 'Manmade', 'Sea and Lake Ice',
                    'Severe Storms', 'Snow', 'Temperature Extremes', 'Volcanoes',
                    'Water Color', 'Wildfires']

    # Find file names of a particular class
    records = df.loc[df['Class'] == cls]

    row = int(math.floor(math.sqrt(num)))
    col = num // row
    f, axarr = plt.subplots(row, col, figsize=(14, 14))
    f.suptitle(classes[cls])

    for ax in axarr.flat:
        root = '/home/aaditya/PycharmProjects/Codefundopp/data'
        idx = np.random.randint(len(records))
        try:
            sample = Image.open(f'{root}/{records.iloc[idx, 0]}.jpg')
        except FileNotFoundError:
            sample = Image.open(f'{root}/{records.iloc[idx, 0]}.png')
        except:
            continue

        ax.imshow(sample)
        ax.set_title(str(records.iloc[idx, 0]))

    plt.show()


def plot_all(num):
    """
    Plots all samples sequentially
    int num: Number of samples to plot
    """
    global CUR_IDX
    FILE_CLS_cleaned = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_file_url_cls_10_jpg.csv'
    df = pd.read_csv(FILE_CLS_cleaned)

    if num > len(df) - CUR_IDX - 1:
        num = len(df) - CUR_IDX - 1
        print('LAST BATCH')

    row = int(math.floor(math.sqrt(num)))
    col = num // row
    f, axarr = plt.subplots(row, col, figsize=(14, 14))

    idx = 0
    for ax in axarr.flat:
        root = '/home/aaditya/PycharmProjects/Codefundopp/data'
        try:
            sample = Image.open(f'{root}/{df.iloc[idx + CUR_IDX, 0]}.jpg')
        except FileNotFoundError:
            sample = Image.open(f'{root}/{df.iloc[idx + CUR_IDX, 0]}.png')
        except:
            continue

        ax.imshow(sample)
        ax.set_title(str(df.iloc[idx + CUR_IDX, 0]))
        idx += 1

    plt.show()
    CUR_IDX += idx


if __name__ == '__main__':
    # check core SDK version number
    # print("Azure ML SDK Version: ", azureml.core.VERSION)
    FILE_CLS_8 = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_file_url_cls_8_cleaned.csv'

    df = pd.read_csv(FILE_CLS_8)

    root= '/home/aaditya/PycharmProjects/Codefundopp/data_wf/absent'


