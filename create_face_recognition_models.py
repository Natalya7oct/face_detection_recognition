import face_recognition
from sklearn import svm
import os
import cv2
import numpy as np
from PIL import Image
from joblib import dump, load
import pandas as pd
from sklearn import neighbors

'''
Training and saving recognition models:

dlib_svm_face_recognition - build with face_recognition lib embading and C-Support Vector Classification (sklearn, trained)

dlib_knn_face_recognition - build with face_recognition lib embading and k-nearest neighbors vote (sklearn, trained)

cv2_face_recognition - build with Haar feature-based cascade classifiers (cv2) and retrained Local Binary Patterns recognizer (cv2, trained)


'''


path = "./Training" # folder for training samples
train_dir = os.listdir(path)


encodings = []
names = []

for person in train_dir:
    pix = os.listdir(path + '/' + person)
    for person_img in pix:
        face = face_recognition.load_image_file("./Training/" + person + "/" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            encodings.append(face_enc)
            names.append(person)
        else:
            print(person + "/" + person_img + " was skipped and can't be used for training")

clf = svm.SVC(gamma='scale', probability=True)
clf.fit(encodings,names)
dump(clf, "./dlib_svm_face_recognition.joblib") 


knn_clf = neighbors.KNeighborsClassifier(algorithm='ball_tree', weights='distance')
knn_clf.fit(encodings,names)
dump(knn_clf, "./dlib_knn_face_recognition.joblib") 


recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    folderPaths = [os.path.join(path,f) for f in os.listdir(path)] 
    faceSamples=[]
    names = []
    ids = []
    id = 0
    for folderPath in folderPaths:
        name = folderPath.split('/')[-1]
        path = folderPath
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') # grayscale
            img_numpy = np.array(PIL_img,'uint8')
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
                names.append(name)
        id = id + 1
    return faceSamples, names, ids
faces,names,ids = getImagesAndLabels(path)
d = dict(zip(ids, names))
faces_dict = {}
[faces_dict.update({k:v}) for k,v in d.items() if v not in faces_dict.values()]

recognizer.train(faces, np.array(ids))
recognizer.write("./cv2_face_recognition.yml") 
