from src.Loader import Loader
from src.Conv2D import Conv2D
from src.Pooling import Pooling
from src.SoftmaxReg import SoftmaxReg
from src import Utils

loader = Loader()
samples, labels = loader.load_samples()
n_classes = len(set(labels))
y_encoded = Utils.encode(labels)

print("Number of classes:", n_classes)

conv = Conv2D(8)
pool = Pooling()
soft = SoftmaxReg(13 * 13 * 8, n_classes)

for image in samples:
    out = conv.forward(image)
    out = pool.forward(out)
    out = soft.forward(out)
    soft.backward(y_encoded, 0.01)
