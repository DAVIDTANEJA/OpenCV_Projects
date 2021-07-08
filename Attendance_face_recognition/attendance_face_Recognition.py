# pip install cmake dlib==19.18.0 face-recognition
# Steps : 1.load image and test with other images , 2.find faces and encoding of images , 3.compare faces and find distance b/w them

# import cv2
# import numpy as np
# import face_recognition

# # 1. load images and convert to RGB
# img = face_recognition.load_image_file("Attendance_Images/elon_musk.jpg")
# img = cv2.cvtColor(cv2.resize(img, (400,400)), cv2.COLOR_BGR2RGB)
# test = face_recognition.load_image_file("Attendance_Images/bill_gates.jpg")     # 1st pass elon musk other images , 2nd bill gates or else to test.
# test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)

# # 2.detect face and encoding
# faceloc = face_recognition.face_locations(img)[0]      # detect, gives 4 points (x1,y1,x2,y2) use to draw rectangle
# encodeimg = face_recognition.face_encodings(img)[0]    # encoding
# cv2.rectangle(img, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255,0,0), 2)

# faceloctest = face_recognition.face_locations(test)[0]      # detect, gives 4 points (x1,y1,x2,y2) use to draw rectangle
# encodeimgtest = face_recognition.face_encodings(test)[0]    # encoding
# cv2.rectangle(test, (faceloctest[3], faceloctest[0]), (faceloctest[1], faceloctest[2]), (255,0,0), 2)

# # 3.compare faces
# results = face_recognition.compare_faces([encodeimg], encodeimgtest)   # we have only 1 image in list increase it later.
# # when we have lot of images we have to find best match. so we find distance (lower distance the best match)
# facedis = face_recognition.face_distance([encodeimg], encodeimgtest)
# print(results, facedis)
# cv2.putText(test, f"{results} {round(facedis[0], 2)}", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)


# cv2.imshow('Image', img)
# cv2.imshow('test', test)

# cv2.waitKey(0)


# ------------------------------------------------------------------------------------------------------------------------
# Attendance
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# 1.load images acc. to names , to add more just add images into folder with names
path = "Attendance_Images"
images = []            # path oof image
classNames = []        # name of image
mylist = os.listdir(path)
for i in mylist:
    img = cv2.imread(f"{path}/{i}")
    images.append(img)
    classNames.append(os.path.splitext(i)[0])
# print(mylist)
# print(classNames)

# 2.find encoding for faces
def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # convert to RGB
        encode = face_recognition.face_encodings(img)[0]    # encoding
        encodelist.append(encode)
    return encodelist

encodelistknown = findEncodings(images)
print("Completed.")

# write attendance names into csv file(create file 'Attendance.csv' and add index : Name,Time) and write time they arrived
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()    # here we read data bcoz if someone arrived don't want to repeat it.
        nameList = []                 # in list append 'name' and 'time'
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])    # name
        if name not in nameList:         # check name present in list or not
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# 3.find match b/w encode , using webcam
# Remember - when we use webcam we can find multiple faces, so we find location of faces and send these locations of faces to encodings function.
cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)    # 1/4 of size , making small image to work fast
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)       # convert to RGB

    # find faces location and send to encodings
    facesCurFrame = face_recognition.face_locations(imgS)    # detect face location , gives 4 points (x1,y1,x2,y2) use to draw rectangle
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)      # encoding

    # now find matches b/w faces, iterate through all faces from current frame(webcam) and compare with encodings we found before (images)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodelistknown, encodeFace)
        faceDis = face_recognition.face_distance(encodelistknown, encodeFace) 
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4      # multiply with 4 bcoz we have resized image 1/4th
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
 
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)


