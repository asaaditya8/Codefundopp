import base64
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import json
from xception import Xception
from PIL import Image
import json

import numpy as np
import os
def im_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    new_str = img_str.decode('utf-8')
    return new_str

def b64_2_img(mystr):
    data = mystr.encode('utf-8')
    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff)

def initialize():
    global model
    model_path = '/home/asaaditya8/PycharmProjects/Codefundopp/weights/xception_wf_rvm_2.pth'
    model = Xception(num_classes=2)
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
            print('output', output)
            # print()
            # print('softmax', F.softmax(output, dim=1))
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
    PATH = '/home/asaaditya8/PycharmProjects/Codefundopp/data_wf/test'
    # for f in os.listdir(PATH):
    f = 'absent/6_e7.jpg'
    img = Image.open(f'{PATH}/{f}')
    inp = im_2_b64(img)
    inp_data = json.dumps({'data': inp})
    print(run(inp_data))