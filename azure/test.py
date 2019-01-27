from dataset import WFDataset
from xception import Xception
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import pyvarinf

BATCH_SIZE = 8
DEVICE = torch.device('cuda')

root= '/home/asaaditya8/PycharmProjects/Codefundopp/data'
datasets = WFDataset(root)

test_dl = DataLoader(datasets.test_ds, BATCH_SIZE)

CKPT = '/home/asaaditya8/PycharmProjects/Codefundopp/weights/xception_wf_rvm_2.pth'
model = Xception(num_classes=6, probs=[0.25, 0.5])
model.load_all(CKPT)
model.to(DEVICE)


def test(model, test_loader):
    model.eval()
    test_accuracy = 0.
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        # get the index of the max probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).sum().item()

    test_accuracy /= len(test_loader.dataset)

    print('\nTest set: Accuracy: {:.2f}%\n'.format(
        100. * test_accuracy))

test(model, test_dl)