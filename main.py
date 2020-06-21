from src.FaceDetection import FaceDetection
from src.Loader import Loader
from src.SoftmaxReg import SoftmaxReg
from src.Utils import encode
from src.Cnn import CNN

def add_student():
    loader = Loader()
    X, y = loader.load_samples()
    num_classes = len(loader.get_classes())

    # print("Number of features:", num_features)
    # print("Number of classes:", num_classes)

    # softmax = SoftmaxReg(loader.size, num_classes)
    # softmax.fit(X, encode(y), 0.01, 100)
    cnn = CNN()
    cnn.fit(X, y, num_classes)
    # for img in X:
    #     softmax.predict(img)

    fd = FaceDetection(cnn)
    fd.open2_face()

def check_attendance():
    loader = Loader()
    X, y = loader.load_samples()

    num_classes = len(loader.get_classes())


    # print("Number of features:", num_features)
    # print("Number of classes:", num_classes)
    print(X.shape, y.shape)
    cnn = CNN(num_classes)
    cnn.fit(X, y)

    # for img in X:
    #     softmax.predict(img)

    fd = FaceDetection(cnn)
    fd.open()

ans=True
while ans:
    print ("""
    1.Add a Student
    2.Check Attendance
    3.Exit
    """)
    #add_student()
    ans=input("What would you like to do? ") 
    if ans=="1": 
        print("\n Adding Student...")
        add_student()
    elif ans=="2":
        print("\n Checking Attendance...")
        check_attendance()
    elif ans=="3":
        print("\n Exiting...")
        break
    