import numpy as np
import cv2
import face_recognition # This project has been made using face_recognition library
import os
from _datetime import datetime

mylist = os.listdir("Images");

ImgList = []
Names = []

for im in mylist:
    cur = cv2.imread(f'Images/{im}')
    ImgList.append(cur)
    Names.append(os.path.splitext(im)[0])

def encodings(ImgL):
    encodelist = []
    for img in ImgL:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)[0]
        encodelist.append(enc)
    return encodelist

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        nameList = f.readlines()
        namesL = []
        for n in nameList:
            n = n.split(',')
            namesL.append(n[0])
        if name not in namesL:
            time = datetime.now().strftime('%H:%M:%S')
            f.writelines(f'\n{name},{time}')

print("Encodings complete")

EncodeList = encodings(ImgList)

cap = cv2.VideoCapture(0)

while True:
    suc, img = cap.read()
    imgr = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)

    curFrameLoc = face_recognition.face_locations(imgr)
    encodeCurFrame = face_recognition.face_encodings(imgr, curFrameLoc)

    for encodeF, faceLoc in zip(encodeCurFrame, curFrameLoc):
        match = face_recognition.compare_faces(EncodeList, encodeF)
        faceDis = face_recognition.face_distance(EncodeList, encodeF)

        index = np.argmin(faceDis)

        if match[index]:
            name = Names[index].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)


    cv2.imshow("Webcam",img)
    cv2.waitKey(1)





