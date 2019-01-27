from dataset import WFDataset
from xception import Xception
from learner import Learner
import torch
from torch import nn
from torch.utils.data import DataLoader

BATCH_SIZE = 16
DEVICE = torch.device('cuda')

root= '/home/asaaditya8/PycharmProjects/Codefundopp/data'
datasets = WFDataset(root)

train_dl = DataLoader(datasets.train_ds, BATCH_SIZE, shuffle=True)
valid_dl = DataLoader(datasets.valid_ds, BATCH_SIZE)
test_dl = DataLoader(datasets.test_ds, BATCH_SIZE, shuffle=True)

PATH = '/home/asaaditya8/PycharmProjects/Codefundopp/classifier/weights/xception_imagenet.pth'

CKPT = '/home/asaaditya8/PycharmProjects/Codefundopp/weights/xception_all_rvm_2.pth'
OUTPUT = './outputs/xception_all_rvm_2.pth'

model = Xception(num_classes=6, probs=[0.25, 0.1])
model.load_scratch(PATH)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

class_weights = torch.Tensor(datasets.classw).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
val_criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')

learn = Learner(model, optimizer, test_dl, valid_dl, criterion, val_criterion, DEVICE)

for epoch in range(1, 2):
    learn.train(epoch)
    learn.test()

for param in model.features.parameters():
    param.requires_grad = True

for epoch in range(2, 12):
    learn.train(epoch)
    learn.test()

torch.save(model.state_dict(), CKPT)
torch.save(model.state_dict(), OUTPUT)