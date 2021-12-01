import cv2
#cv2.imread()
#cv2.resize()
#cv2.CascadeClassifier()
#cv2.cvtColor
import os
# os module creating,removing directory, fetching from the directory
#os.getcwd() - get current working directory
#os.mkdir("file name")  - create file in current working directory
#os.chdir() - change current working directory
#os.listdir(path) - returns the list of all files and directories in the specified directory
import numpy as np
import matplotlib.pyplot as plt

training_data_path = "dataset/training-data"

testing_data_path = "dataset/test-data"

random_image = cv2.imread('dataset/training-data/3/George_W_Bush_0020.jpg')

plt.figure()

fig = plt.figure()

ax1=fig.add_axes((0.1,0.2,0.8,0.7))

ax1.set_title('Image from category 3')

#  local binary pattern 
lbpcascade_frontalface = 'opencv_xml_files/lbpcascade_frontalface.xml'

#lbpcascade_frontalface.xml  - This xml defines a frontal-face layout in terms of the dots

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
	
def prepare_training_data(training_data_folder_path):
    detected_faces = []
    face_labels = []
    traning_image_dirs = os.listdir(training_data_path)
    
    for dir_name in traning_image_dirs:
        label = int(dir_name)
        training_image_path = training_data_path + "/" + dir_name
        training_images_names = os.listdir(training_image_path)
        
        for image_name in training_images_names:
            image_path = training_image_path  + "/" + image_name
            image = cv2.imread(image_path)
            face, rect = detect_face(image)
            if face is not -1:
                resized_face = cv2.resize(face, (121,121), interpolation = cv2.INTER_AREA)
                detected_faces.append(face)
                face_labels.append(label)

    return detected_faces, face_labels
	
detected_faces, face_labels = prepare_training_data("dataset/training-data")

print("Total faces: ", len(detected_faces))
print("Total labels: ", len(face_labels))

lbphfaces_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8)

lbphfaces_recognizer.train(detected_faces,np.array(face_labels))

def draw_rectangle(test_image,rect):
    (x,y,w,h)=rect
    cv2.rectangle(test_image,(x,y),(x+w,y+h),(0,255,0),2)
	
def draw_text(text_image,label_text,x,y):
    cv2.putText(test_image,label_text,(x,y),cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,0),2)
	
def predict(test_image):
    face,rect =detect_face(test_image)
    label = lbphfaces_recognizer.predict(face)
    label_text = tags[label[0]]
    draw_rectangle(test_image,rect)
    draw_text(test_image,label_text,rect[0],rect[1]-5)
    return test_image,label_text
	
tags = ['0','1','2','3','4']



test_image = cv2.imread('dataset/test-data/6/Ana_Guevara_0007.jpg')

plt.imshow(test_image)

predicted_image,label = predict(test_image)

fig = plt.figure()
ax1 = fig.add_axes((0.1,0.2,0.8,0.7))
ax1.set_title('actual class:'+tags[3]+'|'+'predicted class'+label)
plt.axis('off') 
imgplot = plt.imshow(cv2.cvtColor(predicted_image,cv2.COLOR_BGR2RGB))
plt.show()

