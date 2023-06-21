import face_recognition
from sklearn import svm
import os
import cv2
import numpy as np
from PIL import Image
from joblib import dump, load
import pandas as pd


'''
Functions 'show_faces' and 'show_faces_cv2' do pipline:
1. Check all videos in all folders
2. Divide video into pictures
3. Detecte all faces for every picture
4. Recognize every face 
5. Show every pictures with dedicated and signes faces
6. Save results of detection and recognition

'show_faces' uses face_recognition and sklearn libs
'show_faces_cv2' uses open_cv lib

Function 'result' gets the result from txt descriptor.

Function 'check' compares the result from txt descriptor and the result of the model.

'''


def show_faces(video_path, clf):
    cap = cv2.VideoCapture(video_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    count = 0
    period = -1
    faces_list = []
    periods = []

    while cap.isOpened():
        ret, img = cap.read()

        if ret:
            
            period = period + 1
            face_locations = face_recognition.face_locations(img)
            img_enc = face_recognition.face_encodings(img, face_locations)
            
            if len(img_enc)>0:
                
                face_names_prob = list(clf.predict_proba(list(img_enc)))
                face_names = []

                for i in range(len(face_names_prob)):
                    d = dict(zip(clf.classes_, face_names_prob[i]))
                    d_sorted = sorted(d.items(), key=lambda i: i[1])[-1]
                    if d_sorted[1]>0.6:
                        face_names.append(d_sorted[0])
                    else:
                        face_names.append('unknown')

                        
                
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    if not name:
                        continue

                    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

                    cv2.rectangle(img, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                                
                    faces_list.append([name, top, right, bottom, left])
                    periods.append(period)
            
            cv2.imshow('camera',img) 
            if cv2.waitKey(10) == ord('q'):
                break
            '''n = video_path.split('/')[-2]
            i = video_path.split('/')[-1][:-4]
            filename = '/home/green/find_faces/results/' + str(n) + " " + str(i) + " " + str(period) + ".jpg"
            cv2.imwrite(filename, img)'''
            count += 10 
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        else:
            cap.release()
            break
        
    cv2.destroyAllWindows()
    faces = list(zip(periods, faces_list))

    return faces


def result(path):
    my_file = open(path, "r")
    data = my_file.read()
    data_into_list = data.split("\n")
    my_file.close()
    data = []
    periods = []
    period = -1
    for i in data_into_list:
        if i!='':
            try:
                period = int(i)
            except:
                data.append(i)
                periods.append(period)
    periods = [int(i) for i in periods]
    faces_list = []
    for i in data:
        res = []
        res.append(i.split(' ')[0])
        res = res+i.split(' ')[1].split(',')
        faces_list.append(res)
    faces = list(zip(periods, faces_list))
    return faces


def check(data1,data2):
    TP=0
    FP=0
    for i in range(len(data1)):
        try:
            true = data1[i][1]
            predict = data2[i][1]
            check0 = (true[0]==predict[0])
            check1 = (abs(int(true[1])-int(predict[1]))<50)
            check2 = (abs(int(true[2])-int(predict[2]))<50)
            check3 = (abs(int(true[3])-int(predict[3]))<50)
            check4 = (abs(int(true[4])-int(predict[4]))<50)
            if check0+check1+check2+check3+check4 == 5:
                TP=TP+1
            else:
                FP=FP+1
        except:
            FP=FP+1
    return(TP,FP)



def show_faces_cv2(video_path,faces_dict):

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("./cv2_face_recognition.yml")
    cascadePath = "./haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    cap = cv2.VideoCapture(video_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    count = 0
    period = -1
    faces_list = []
    periods = []

    while cap.isOpened():
        ret, img = cap.read()

        if ret:
            
            period = period + 1
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            face_locations = faceCascade.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 5)
            
            if len(face_locations)>0:

                face_names = []
                
                for(x,y,w,h) in face_locations:

                    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                    id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
                    if (confidence < 100):
                        name = faces_dict[id]
                    else:
                        name = "unknown"
                    
                    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.rectangle(img, (x, y+h - 25), (x+w, y+h), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, name, (x + 6, y+h - 6), font, 0.5, (255, 255, 255), 1)
                                
                    faces_list.append([name, y, x+w, y+h, x])
                    periods.append(period)
            
            cv2.imshow('camera',img) 
            if cv2.waitKey(10) == ord('q'):
                break
            '''n = video_path.split('/')[-2]
            i = video_path.split('/')[-1][:-4]
            filename = '/home/green/find_faces/results/' + str(n) + " " + str(i) + " " + str(period) + ".jpg"
            cv2.imwrite(filename, img)'''
            count += 10 
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        else:
            cap.release()
            break
        
    cv2.destroyAllWindows()
    faces = list(zip(periods, faces_list))

    return faces