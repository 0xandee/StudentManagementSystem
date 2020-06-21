import os
import numpy as np
import cv2
import src.BgSub as BgSub

class Loader:

    def __init__(self):
        self.dir = 'sample'
        self.size = (64, 64)

    def get_classes(self):
        dirs = os.listdir(self.dir)
        for d in dirs:
            if '.' in d:
                dirs.remove(d)
        return dirs

    def load_samples_of_class(self, c):
        p = os.path.join(self.dir, c)
        f = os.listdir(p)
        samples = list()

        for x in f:
            im = cv2.imread(os.path.join(p, x))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, self.size)
            im = BgSub.remove_bg(im)
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
