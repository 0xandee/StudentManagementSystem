import os
import numpy as np
import cv2

class Loader:

    def __init__(self):
        self.dir = 'sample'
        self.size = (28, 28)

    def get_classes(self):
        return os.listdir(self.dir)

    def load_samples_of_class(self, c):
        p = os.path.join(self.dir, c)
        f = os.listdir(p)
        samples = list()

        for x in f:
            im = cv2.imread(os.path.join(p, x), cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, self.size)
            samples.append(im / 255)

        return np.array(samples)

    def load_samples(self):
        classes = self.get_classes()
        samples = None
        labels = []

        for c in classes:
            class_samples = self.load_samples_of_class(c)
            labels = labels + ([c] * class_samples.shape[0])

            if (samples is None):
                samples = class_samples
            else:
                samples = np.concatenate((samples, class_samples))

        return samples, np.array(labels)
