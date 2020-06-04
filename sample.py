import os
import numpy as np
from PIL import Image
import utils

sample_dir = 'sample'
sample_size = (28, 28)


def get_classes():
    return os.listdir(sample_dir)

def get_class_samples(c):
    p = os.path.join(sample_dir, c)
    f = os.listdir(p)
    samples = []

    for x in f:
        im = Image.open(os.path.join(p, x))
        im = utils.scaleimg(sample_size, im)
        samples.append(np.array(im.convert('1')) / 255)

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
