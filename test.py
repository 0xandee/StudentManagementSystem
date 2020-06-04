import cv2
import datetime
import sample
import utils
import numpy as np
from PIL import Image
from src.Conv2D import Conv2D
from src.Pooling import Pooling
from src.SoftmaxReg import SoftmaxReg
X, y = sample.get_all_samples()
num_features = X.shape[1]
num_classes = len(set(y))
new_y = utils.encode(y)

conv = Conv2D(8)
pool = Pooling()
out = conv.forward(X)
print(out.shape)
out = pool.forward(out)
print(out.shape)

softmax = SoftmaxReg(391, num_classes)
out = softmax.forward(out)
print(out.shape)