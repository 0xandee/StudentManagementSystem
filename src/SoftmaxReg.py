import numpy as np

class SoftmaxReg:

    def __init__(self, num_features, num_classes):
        self.weights = np.random.randn(num_features, num_classes)
        self.input = None

    def predict(self, X):
        pred = self.compute_h(X)
        return np.argmax(pred, axis=1)

    def compute_gradient(self, X, error):
        return X.T.dot(error)
    
    def softmax_activation(self, x):
        ex = np.exp(x)
        return ex / np.sum(ex, axis = 0)
    
    def compute_h(self, input):
        return self.softmax_activation(np.dot(input, self.weights))

    def forward(self, image):
        self.input = image.flatten()
        return self.compute_h(self.input)
    
    def backward(self, y, learning_rate):
        h = self.compute_h(self.input)
        error = h - y
        print(self.input.shape, error.shape)
        gradient = self.compute_gradient(self.input, error)
        self.weights += -learning_rate * gradient
        return error


    

