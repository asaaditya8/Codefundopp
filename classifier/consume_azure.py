import requests
import json
import numpy as np
from PIL import Image

def test_image(PATH):
    img = Image.open(PATH)
    inp = np.array(img, dtype='float')
    inp_data = json.dumps({'data': inp.tolist()})
    return inp_data

scoring_uri = 'http://104.45.178.57:80/score'
input_data = test_image('/home/aaditya/PycharmProjects/Codefundopp/data_wf/val/absent/nowf_36.jpg')
headers = {'Content-Type': 'application/json'}
result = requests.post(scoring_uri, input_data, headers=headers)
print(result.text)