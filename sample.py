import os
import numpy as np
from PIL import Image

sample_dir = 'sample'
sample_size = (28, 28)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def get_classes():
    return os.listdir(sample_dir)

def get_class_samples(c):
    p = os.path.join(sample_dir, c)
    f = os.listdir(p)
    samples = []

    for x in f:
        im = Image.open(os.path.join(p, x))
        im.thumbnail(sample_size, Image.ANTIALIAS)
        im_gray = rgb2gray(np.array(im.getdata()))
        samples.append(im_gray / 255)

    return np.array(samples)

def get_all_samples():
    classes = get_classes()
    samples = None
    labels = []

    for c in classes:
        class_samples = get_class_samples(c)
        labels = labels + ([c] * class_samples.shape[0])

        if (samples is None):
            samples = class_samples
        else:
            samples = np.concatenate((samples, class_samples))

    return samples, np.array(labels)

# samples, labels = get_all_samples()

# print(samples.shape)
# print(labels)