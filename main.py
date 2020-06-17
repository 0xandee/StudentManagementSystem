from src.FaceDetection import FaceDetection
from src.Loader import Loader
from src.SoftmaxReg import SoftmaxReg
from src.Utils import encode

loader = Loader()
X, y = loader.load_samples()
num_classes = len(loader.get_classes())

# print("Number of features:", num_features)
# print("Number of classes:", num_classes)

softmax = SoftmaxReg(loader.size, num_classes)
softmax.fit(X, encode(y), 0.01, 100)

# for img in X:
#     softmax.predict(img)

fd = FaceDetection(softmax)
fd.open()