import tensorflow as tf
import numpy as np
import cv2
from src.Loader import Loader
from tensorflow.keras import datasets, layers, models

class CNN:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(num_classes))
        self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    def fit(self, train_images, train_labels):
        train_labels = np.unique(train_labels, return_inverse=True)[1]
        history = self.model.fit(train_images, train_labels, epochs=50)

    def predict(self, X):
        if (len(X.shape) == 3):
            X = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
        X = cv2.resize(X, (64, 64))
        X = X / 255
        X = X.reshape((1, 64, 64, 1))
        return self.model.predict_classes(X)[0]
