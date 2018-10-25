import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def open_pil_image(path):
    return Image.open(path)


def open_image(path):
    return np.array(Image.open(path), dtype='float32')


def resize_pil_image(Img, size):
    return Img.resize(size, Image.ANTIALIAS)


def resize_image(img, size):
    return np.array(resize_pil_image(Image.fromarray(img.astype('uint8')), size), dtype='float32')


def rescale_pil_image(Img, size):
    W, H = size
    w, h = Img.size
    factor = np.min([W/w, H/h])
    nw = int(w*factor)
    nh = int(h*factor)
    return resize_pil_image(Img, (nw,nh))


def rescale_image(img, size):
    return np.array(rescale_pil_image(Image.fromarray(img.astype('uint8')), size), dtype='float32')


def pad_image(img, size):
    wp, hp = size
    wp1 = int(np.ceil(wp/2)); wp2 = int(np.floor(wp/2))
    hp1 = int(np.ceil(hp/2)); hp2 = int(np.floor(hp/2))
    H, W = img.shape[:-1]
    w = W + wp
    h = H + hp
    new_img = np.zeros((h,w,3), dtype='float32')
    new_img[hp1:h-hp2, wp1:w-wp2, :] = img
    return new_img


def pad_pil_image(Img, size):
    return Image.fromarray(pad_image(np.array(Img, dtype='float32'), size).astype('uint8'))


def rescale_pad(img, size):
    img = rescale_image(img, size)
    W, H = size
    h, w = img.shape[:-1]
    return pad_image(img, (W-w, H-h))


def rescale_pad_pil(Img, size):
    return Image.fromarray(rescale_pad(np.array(Img, dtype='float32'), size).astype('uint8'))


if __name__ == '__main__':
    img = open_image('data/0.jpg')
    ni = rescale_pad(img, (224, 224)).astype('uint8')
    plt.imshow(ni)
    plt.show()