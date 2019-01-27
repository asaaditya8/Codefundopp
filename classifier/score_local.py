import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import json
from classifier.xception import Xception
from PIL import Image
import numpy as np
import os

def initialize():
    global model
    model_path = 'weights/xception_wf0.pth'
    model = Xception()
    model.load_all(model_path)
    model.eval()

def run(input_data):

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
            classes = ['absent', 'present']
            # For just one sample
            pred_probs = output.numpy()[0]
            index = torch.argmax(output, 1)

        result = {"label": classes[index], "probability": str(pred_probs[index])}
        return result
    except Exception as e:
        result = str(e)
        return {"error": result}

if __name__ == '__main__':
    initialize()
    PATH = 'test'
    for f in os.listdir(PATH):
        img = Image.open(f'{PATH}/{f}')
        print(run(img))