import os
from PIL import Image

PATH = '/home/aaditya/Pictures/'

fnames = sorted(next(os.walk(PATH))[2])
imgpath = fnames[]
newpath = imgpath.split('.')[0] + '_1.png'
print(imgpath)
print(newpath)
img = Image.open(PATH + imgpath)
# print(img.size)
img = img.resize((1600, 900))
print(img.size)
# img.save(PATH + newpath)