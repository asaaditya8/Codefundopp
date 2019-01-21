import PIL
import numpy as np

import torch
from torch import nn
from fastai import vision
from torchvision import models
from classifier.dataset import CustomDatasetFromCSV
import pretrainedmodels
from classifier.xception import Xception

def model_building():
    PATH = '/home/aaditya/PycharmProjects/Codefundopp/weights/xception_imagenet.pth'

    inp = torch.Tensor(4, 3, 299, 299).uniform_(-1, 1)
    model = XModel()
    lrs = model.layers[:5] + [model.layers[2]] + model.layers[5:17] + [model.layers[2]] + model.layers[17:-2]

    m = nn.Sequential(*lrs)
    # torch.save(the_model.state_dict(), PATH)
    m2 = Xception()

    print(m.state_dict().keys())
    print(m2.features.state_dict().keys())

    m2.features.load_state_dict(m.state_dict())

    torch.save(m2.features.state_dict(), PATH)


class XModel(nn.Module):
    def __init__(self):
        super(XModel, self).__init__()
        self.top = vision.create_head(nf=1024, nc=8)
    
        self.img_model = pretrainedmodels.models.xception()
        self.layers = []

        for c in self.img_model.named_children():
            self.layers.append(c[1])


    def forward(self, input):
        x = self.body(input)
        return self.top(x)


if __name__ == '__main__':
    pass