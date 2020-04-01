import base64
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import json

from azureml.core.model import Model

class Flatten(nn.Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"

    def __init__(self, full=False):
        super().__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=2, probs=(0.25, 0.5)):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes
        relu = nn.ReLU(inplace=True)
        self.body = [
            nn.Conv2d(3, 32, 3,2, 0, bias=False),
            nn.BatchNorm2d(32),
            relu,

            nn.Conv2d(32,64,3,bias=False),
            nn.BatchNorm2d(64),
            relu,

            Block(64,128,2,2,start_with_relu=False,grow_first=True),
            Block(128,256,2,2,start_with_relu=True,grow_first=True),
            Block(256,728,2,2,start_with_relu=True,grow_first=True),

            Block(728,728,3,1,start_with_relu=True,grow_first=True),
            Block(728,728,3,1,start_with_relu=True,grow_first=True),
            Block(728,728,3,1,start_with_relu=True,grow_first=True),
            Block(728,728,3,1,start_with_relu=True,grow_first=True),

            Block(728,728,3,1,start_with_relu=True,grow_first=True),
            Block(728,728,3,1,start_with_relu=True,grow_first=True),
            Block(728,728,3,1,start_with_relu=True,grow_first=True),
            Block(728,728,3,1,start_with_relu=True,grow_first=True),

            Block(728,1024,2,2,start_with_relu=True,grow_first=False),

            relu,
            SeparableConv2d(1024,1536,3,1,1),
            nn.BatchNorm2d(1536),

            SeparableConv2d(1536,2048,3,1,1)
        ]

        self.features = nn.Sequential(*self.body)

        #feature output is 2048 x 10 x 10
        self.top = [
            # nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1),
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(num_features=4096),
            nn.Dropout(p=probs[0]),
            nn.Linear(in_features= 4096, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(p=probs[1]),
            nn.Linear(in_features=512, out_features=self.num_classes, bias=True)
        ]

        self.classifier = nn.Sequential(*self.top)

    def forward(self, input):
        x = self.features(input)
        x = self.classifier(x)
        return x

    def load_scratch(self, PATH):
        self.features.load_state_dict(torch.load(PATH))

        for param in self.features.parameters():
            param.requires_grad = False

    def load_all(self, PATH):
        self.load_state_dict(torch.load(PATH, map_location='cpu'))

def b64_2_img(mystr):
    data = mystr.encode('utf-8')
    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff)

def init():
    global model
    model_path = Model.get_model_path('xception_all_1')
    model = Xception(num_classes=6, probs=[0.25, 0.1])
    model.load_all(model_path)
    model.eval()

def run(input_data):
    input_data = b64_2_img(json.loads(input_data)['data'])

    try:
        SIZE = 299
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

        tmfs = transforms.Compose([
            transforms.Resize(SIZE),
            transforms.CenterCrop(SIZE),
            transforms.ToTensor(),
            normalize
        ])

        input_data = tmfs(input_data)

        # get prediction
        with torch.no_grad():
            output = F.softmax(model(input_data.view(1, 3, 299, 299)), dim=1)
            classes = ['Dust and Haze', 'Sea and Lake Ice', 'Severe Storms', 'Volcanoes', 'Water Color', 'Wildfires']
            # For just one sample
            pred_probs = output.numpy()[0]
            index = torch.argmax(output, 1)

        result = {"label": classes[index], "probability": str(pred_probs[index])}
        return result
    except Exception as e:
        result = str(e)
        return {"error": result}