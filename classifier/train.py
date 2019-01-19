from fastai import vision
from classifier.dataset import CustomDatasetFromCSV
import torch
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision.transforms import transforms
from torchvision import models
import PIL
import numpy as np

BATCH_SIZE = 8
TRAIN_RATIO = 0.9
SHUFFLE = True
DEVICE = torch.device('cuda')


def train(epoch, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                100. * batch_idx / len(train_loader), loss.item()))

            # log the loss to the Azure ML run


def test(test_loader):
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).sum().item()

    test_accuracy /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, 100. * test_accuracy))


if __name__ == '__main__':
    csv_path = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_file_url_cls_8_cleaned.csv'
    full_dataset = CustomDatasetFromCSV(csv_path, transform=transforms.Compose([
        transforms.Resize((224,224), PIL.Image.ANTIALIAS),
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

    model = models.squeezenet1_0()
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1, 2):
        train(epoch, train_dl, optimizer)
        test(valid_dl)

