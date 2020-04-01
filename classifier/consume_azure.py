import base64
from io import BytesIO

import requests
import json
from PIL import Image
import os

def im_2_b64(image):
    buff = BytesIO()
    image = image.convert('RGB')
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    new_str = img_str.decode('utf-8')
    return new_str

def test_image(PATH):
    img = Image.open(PATH)
    inp = im_2_b64(img)
    inp_data = json.dumps({'data': inp})
    return inp_data

scoring_uri = 'http://104.45.187.156:80/score'
headers = {'Content-Type': 'application/json'}

input_data = test_image('/home/aaditya/PycharmProjects/Codefundopp/cropped_tasmanian_wf.png')
result = requests.post(scoring_uri, input_data, headers=headers)
print(result.text)

# fpaths = sorted(os.listdir('../other_data/aa'))
# for f in fpaths:
#     input_data = test_image(f'../other_data/aa/{f}')
#     result = requests.post(scoring_uri, input_data, headers=headers)
#     print(f, result.text)