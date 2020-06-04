# import cv2
# import datetime
# import sample
# import utils
# import numpy as np
# from PIL import Image
# from sklearn.linear_model import LogisticRegression
# from src import SoftmaxReg

# cascPath = "haarcascade_frontalface_default.xml"

# faceCascade = cv2.CascadeClassifier(cascPath)

# X, y = sample.get_all_samples()
# logmodel = LogisticRegression(multi_class='multinomial').fit(X, y)
# print(logmodel.predict(X))

# def detectNearestFace(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30),
#         flags = cv2.CASCADE_SCALE_IMAGE
#     )
    
#     img = [0, 0, 0, 0]
    
#     for (x, y, w, h) in faces:
#         if (w > img[2] and h > img[3]):
#             img = [x, y, w, h]
        
#     (x, y, w, h) = img
#     crop = frame[y:y+h, x:x+w]
#     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     return image, crop

# cam = cv2.VideoCapture(0)
# index = 0
# while True:
#     ret, frame = cam.read()
#     if not ret:
#         break
    
#     image, face = detectNearestFace(frame)
    
#     cv2.imshow("", image)
    
#     if (len(face) != 0):
#         scale = utils.scaleimg(sample.sample_size, Image.fromarray(face))
#         img = utils.rgb2gray(np.array(scale.getdata())) / 255
#         print(logmodel.predict(np.array([img])))
    
#     k = cv2.waitKey(1)
#     if k%256 == 27:
#         # ESC pressed
#         print("Escape hit, closing...")
#         break
#     elif k%256 == 32:
#         #imgname = str(datetime.datetime.now()) + ".png"
#         imgname = "sample/" + str(index) + ".png"
#         index += 1
#         print("Save new image " + imgname)
#         print(face)
#         cv2.imwrite(imgname, face)
#         print(cv2.imwrite(imgname, face))
    

# cam.release()
# cv2.destroyAllWindows()
