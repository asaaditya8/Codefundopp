import PIL
import numpy as np

import torch
from fastai import vision
from torchvision import models
from classifier.dataset import CustomDatasetFromCSV
from torchvision.transforms import transforms
from torch.utils.data import SubsetRandomSampler, DataLoader

BATCH_SIZE = 8
TRAIN_RATIO = 0.9
SHUFFLE = True
DEVICE = torch.device('cuda')


def mf(*args, **kwargs):
    return vision.models.wrn_22()

if __name__ == '__main__':
    csv_path = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_file_url_cls_8_cleaned.csv'
    full_dataset = CustomDatasetFromCSV(csv_path, transform=transforms.Compose([
        transforms.Resize((224, 224), PIL.Image.ANTIALIAS),
        transforms.ToTensor()
    ]))

    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    split = int(np.floor(TRAIN_RATIO * num_samples))

    if SHUFFLE:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)

    valid_sampler = SubsetRandomSampler(valid_idx)

    train_dl = DataLoader(full_dataset, BATCH_SIZE, sampler=train_sampler)
    valid_dl = DataLoader(full_dataset, BATCH_SIZE, sampler=valid_sampler)
    test_dl = DataLoader(full_dataset, BATCH_SIZE)

    # data = vision.ImageDataBunch(train_dl, valid_dl, test_dl=test_dl, device=DEVICE)

    # learn = vision.create_cnn(data, models.squeezenet1_0)
    model = vision.models.wrn_22()