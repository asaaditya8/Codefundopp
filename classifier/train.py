from classifier.dataset import WFDataset
from classifier.xception import Xception
from classifier.learner import Learner
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import pyvarinf

BATCH_SIZE = 8
DEVICE = torch.device('cuda')

root= 'data_wf'
datasets = WFDataset(root)

train_dl = DataLoader(datasets.train_ds, BATCH_SIZE, shuffle=True)
valid_dl = DataLoader(datasets.valid_ds, BATCH_SIZE)
test_dl = DataLoader(datasets.test_ds, BATCH_SIZE)

PATH = 'classifier/weights/xception_imagenet.pth'

CKPT = 'weights/xception_all0.pth'
model = Xception(probs=[0.25, 0.5])
# model.load_all(CKPT)
model.load_scratch(PATH)
# var_model = pyvarinf.Variationalize(model.classifier)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

class_weights = torch.Tensor(datasets.classw).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
val_criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')

learn = Learner(model, optimizer, train_dl, valid_dl, criterion, val_criterion, DEVICE)

# for param in model.features.parameters():
#     param.requires_grad = True

for epoch in range(1, 4):
    learn.train(epoch)
    learn.test()

torch.save(model.state_dict(), CKPT)
