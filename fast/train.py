from classifier.dataset import WFDataset
from classifier.xception import Xception
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import pyvarinf
from fastai import vision

BATCH_SIZE = 8
DEVICE = torch.device('gpu')

root= 'data_wf'
datasets = WFDataset(root)

train_dl = DataLoader(datasets.train_ds, BATCH_SIZE, shuffle=True)
valid_dl = DataLoader(datasets.valid_ds, BATCH_SIZE)
test_dl = DataLoader(datasets.test_ds, BATCH_SIZE)

PATH = 'classifier/weights/xception_imagenet.pth'

CKPT = 'weights/xception_wf_rvm_of.pth'
model = Xception(probs=[0.25, 0.5])
model.load_scratch(PATH)
# var_model = pyvarinf.Variationalize(model.classifier)
# var_model = model.classifier
# model.load_all(CKPT)
# model.features.to(DEVICE) # model.cuda will also work
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

class_weights = torch.Tensor(datasets.classw).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)
val_criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')

def train(epoch, train_loader, optimizer):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        # data = model.features(data)
        output = model(data)
        preds = output.data.max(1, keepdim=True)[1]
        loss = criterion(output, target)
        # loss_prior = var_model.prior_loss() / len(train_loader.dataset)
        # loss += loss_prior
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * data.size(0)
        running_corrects += preds.eq(target.data.view_as(preds)).sum().item()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [({:.0f}%)]\tLoss: {:.6f}\tAcc: {}'.format(
                epoch,
                100. * batch_idx / len(train_loader),
                (running_loss / ((1+batch_idx) * BATCH_SIZE) ),
                (running_corrects / ((1+batch_idx) * BATCH_SIZE))))

            # log the loss to the Azure ML run


def test(test_loader):
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        # output = [var_model(data).view(-1, 2, 1) for i in range(5)]
        # output = torch.mean(torch.cat(outputf, 2), 2)
        # sum up batch loss
        test_loss += val_criterion(output, target).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).sum().item()

    test_accuracy /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, 100. * test_accuracy))


for param in model.features.parameters():
    param.requires_grad = True

# data = vision.ImageDataBunch.from_folder('data_wf', ds_tfms=)
bunch = vision.DataBunch(train_dl, valid_dl, device=DEVICE)
learn = vision.Learner(bunch, model, optimizer, criterion, metrics=vision.accuracy)
learn.fit_one_cycle(1)
# for epoch in range(1, 4):
#     train(epoch, train_dl, optimizer)
#     test(valid_dl)

# torch.save(model.state_dict(), CKPT)
