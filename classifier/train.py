from classifier.dataset import WFDataset
from classifier.xception import Xception
import torch
from torch import nn
from torch.utils.data import DataLoader

BATCH_SIZE = 8
DEVICE = torch.device('cuda')

root= '/home/aaditya/PycharmProjects/Codefundopp/data_wf/'
datasets =WFDataset(root)

train_dl = DataLoader(datasets.train_ds, BATCH_SIZE)
valid_dl = DataLoader(datasets.valid_ds, BATCH_SIZE)
test_dl = DataLoader(datasets.test_ds, BATCH_SIZE)

PATH = '/home/aaditya/PycharmProjects/Codefundopp/weights/xception_imagenet.pth'

CKPT = '/home/aaditya/PycharmProjects/Codefundopp/weights/xception_train2.pth'
model = Xception()
model.load_scratch(PATH)
model.to(DEVICE) # model.cuda will also work

optimizer = torch.optim.Adam(model.parameters())

class_weights = torch.Tensor(datasets.classw).cuda()
criterion = nn.CrossEntropyLoss(weight=class_weights)
val_criterion = nn.CrossEntropyLoss(weight=class_weights, size_average=False)

def train(epoch, train_loader, optimizer):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        preds = torch.max(output, 1)[1]
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * data.size(0)
        running_corrects += torch.sum(preds == target.data)

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
        test_loss += val_criterion(output, target).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).sum().item()

    test_accuracy /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, 100. * test_accuracy))


for epoch in range(1, 3):
    train(epoch, train_dl, optimizer)
    test(valid_dl)

torch.save(model.state_dict(), CKPT)
