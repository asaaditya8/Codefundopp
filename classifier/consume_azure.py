import base64
from io import BytesIO

import requests
import json
from PIL import Image

def im_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    new_str = img_str.decode('utf-8')
    return new_str

def test_image(PATH):
    img = Image.open(PATH)
    inp = im_2_b64(img)
    inp_data = json.dumps({'data': inp})
    return inp_data

scoring_uri = 'http://20.185.214.218:80/score'
input_data = test_image('data_wf/test/absent/6_e7.jpg')
headers = {'Content-Type': 'application/json'}

result = requests.post(scoring_uri, input_data, headers=headers)
print(result.text)