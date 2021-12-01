import os
import cv2
import matplotlib.pyplot as plt
training_data_path = "dataset/training-data"
lis=os.listdir(training_data_path)
print(lis)

lbpcascade_frontalface = 'opencv_xml_files/lbpcascade_frontalface.xml'



def detect_face(input_img):
    image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    # cv::CascadeClassifier::load to load a .xml classifier file. It can be either a Haar or a LBP classifier
    #In this file i have used cascade classifier
    face_cascade = cv2.CascadeClassifier(lbpcascade_frontalface)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
    if (len(faces) == 0):
        return -1, -1
    (x, y, w, h) = faces[0]
    return image[y:y+w, x:x+h], faces[0]

image = cv2.imread('dataset/training-data/1/Alvaro_Uribe_0002.jpg')
face, rect = detect_face(image)
plt.figure()
plt.show(face)