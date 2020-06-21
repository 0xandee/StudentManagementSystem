import cv2
import datetime
import src.BgSub as BgSub
from src.Loader import Loader

class FaceDetection:

    def __init__(self, model):
        self.img_index = 0
        self.model = model
        self.classes = Loader().get_classes()
        self.cam = cv2.VideoCapture(0)
        self.cascade_path = "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)

    def __del__(self):
        self.cam.release()
        cv2.destroyAllWindows()

    def draw_rectangle(self, image, coords):
        (x, y, w, h) = coords
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    def draw_text(self, image, text, frame):
        (x, y, _, h) = frame
        center = (x, int(y + h + 60 / 2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text, center, font, 0.8, (0, 255 , 0), 2, cv2.LINE_AA)

    def open(self):
        cv2.namedWindow("acd_face")
        listimg = list()
        maxcount = 20
        counts = [0] * len(self.classes)
        last_class = self.classes[0]

        while True:
            ret, frame = self.cam.read()

            if not ret:
                break

            image, faces, frame = self.detect_faces(frame)

            if (cv2.getWindowProperty("acd_face", 1) == -1):
                break

            if (len(faces) == 1):
                listimg.append(faces[0])
                print(self.model.predict(faces[0]))

            if (len(listimg) == maxcount):
                for f in listimg:
                    counts[self.model.predict(f)] += 1
                
                index = counts.index(max(counts))
                print(self.classes[index])
                for i in range(len(self.classes)):
                    print(self.classes[i] , ":", counts[i] / maxcount)
                listimg.clear()
                counts = [0] * len(self.classes)
                exit(0)

            cv2.imshow("acd_face", image)
            cv2.waitKey(1)
    

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coord_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        faces = []
        frame = None

        for (x, y, w, h) in coord_faces:
            temp = image[y:y+h, x:x+w]
            faces.append(temp)
            frame = (x, y, w, h)
        return image, faces, frame
    
    def detect_faces2(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coord_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )
        
        faces = []
        temp = 0
        for (x, y, w, h) in coord_faces:
            temp = image[y:y+h, x:x+w]
            faces.append(temp)
            pred = self.model.predict(temp)
            self.draw_text(image, self.classes[pred], (x, y, w, h))
            self.draw_rectangle(image, (x, y, w, h))
            
        return image, temp

    def save_img(self, face):
        imgname = "sample/" + str(self.img_index) + ".png"
        self.img_index += 1
        print("Save new image " + imgname)
        cv2.imwrite(imgname, face)
        cv2.imwrite(imgname, face)


    def open2_face(self):
        cv2.namedWindow("acd_face")
        index = 0
        while True:
            ret, frame = self.cam.read()

            if not ret:
                break

            image, faces = self.detect_faces2(frame)

            
            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                #imgname = str(datetime.datetime.now()) + ".png"
                imgname = "sample/" + str(index) + ".png"
                index += 1
                print("Save new image " + imgname)
                print(faces)
                cv2.imwrite(imgname, faces)
                print(cv2.imwrite(imgname, faces))

            if (cv2.getWindowProperty("acd_face", 1) == -1):
                break
            
            cv2.imshow("acd_face", frame)
            cv2.waitKey(1)
