import numpy as np
from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def scaleimg(size, img):
    img.thumbnail(size, Image.ANTIALIAS)
    return img