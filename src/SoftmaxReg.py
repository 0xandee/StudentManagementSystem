import numpy as np
import src.BgSub as BgSub
import cv2

class SoftmaxReg:

    def __init__(self, size, num_classes):
        (w, h) = size
        self.size = size
        self.num_features = w * h
        self.num_classes = num_classes
        self.weights = np.random.randn(self.num_features, self.num_classes)
        self.input = None

    def normalize(self, X):
        max = np.max(X)
        return X / max

    def predict(self, X):
        if (len(X.shape) == 3):
            X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
        X = cv2.resize(X, self.size)
        X = BgSub.remove_bg(X)
        X = self.normalize(X.flatten())
        pred = self.compute_h(X)
        index = pred.argmax()
        return index

    def compute_gradient(self, X, error):
        X = X.reshape(self.num_features, 1)
        error = error.reshape(1, self.num_classes)
        return X.dot(error)
    
    def softmax_activation(self, x):
        ex = np.exp(x)
        return ex / np.sum(ex, axis = 0)
    
    def compute_h(self, input):
        return self.softmax_activation(np.dot(input, self.weights))

    def forward(self, image):
        self.input = image.flatten()
        return self.compute_h(self.input)
    
    def backward(self, h, y, learning_rate):
        error = h - y
        gradient = self.compute_gradient(self.input, error)
        # print(self.weights.shape,"vs",gradient.shape)
        self.weights += -learning_rate * gradient
        return error, gradient

    def fit(self, X, y, learning_rate, epochs):
        for _ in range(epochs):
            for i in range(len(X)):
                h = self.forward(X[i])
                self.backward(h, y[i], learning_rate)


    

