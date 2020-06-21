 from sklearn import svm

 class SVM:
     def __init__(self):
        self.clf = svm.SVC(decision_function_shape='ovo')

    def fit(self, trainX, trainY):
        self.clf.fit(trainX, trainY)

    def predict(self, X):
        print(self.clf.predict(X))
        return 0
