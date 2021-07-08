# 1. find face and create bbox around it  ,  2.find landmarks on face  
# 3.create function "createBox()" help in find particular parts of face using 'crop' and also using 'masking' for parts
# 4. also using trackbar for real-time functionality like changing color of lips.
# 5. we can also use webcam.
# Also download the file : "shape_predictor_68_face_landmarks.dat"

import cv2
import numpy as np
import dlib

# find/detect face and landmarks
detector = dlib.get_frontal_face_detector()    # objects
predictor = dlib.shape_predictor("files/shape_predictor_68_face_landmarks.dat")    # file 68 landmarks

# 5.we can also add webcam functionality
webcam = True                 # if don't want to use webcam make it  "False"
cap = cv2.VideoCapture(0)

# 4.
# we can also use trackbar to change color of lips
def empty(a):
    pass
cv2.namedWindow("BGR")
cv2.resizeWindow("BGR", 640, 240)
cv2.createTrackbar("Blue", "BGR", 0, 255, empty)
cv2.createTrackbar("Green", "BGR", 0, 255, empty)
cv2.createTrackbar("Red", "BGR", 0, 255, empty)


# 3.
# function where we pass image , landmarks points and give output for particular face part eyes, nose ..
# like we want to crop eye we have 6 points which give location of eye we create bbox around it based on points
# 1st we create bbox and then crop that part and put image
# def createBox(img, points, scale=5):          # points - for particular part eye, nose
#     bbox = cv2.boundingRect(points)  # based on 'points' it will give 4 points to create bbox
#     x,y,w,h = bbox
#     imgCrop = img[y:y+h, x:x+w]    # its small so we resize it
#     imgCrop = cv2.resize(imgCrop, (0,0), None, scale, scale)    # using scale bcoz user can give value acc. it will enlarge image part
#     return imgCrop

# but if we want to color particular part like eye, lips..  we need exact location instead of rectangle(bbox) we need 'polygon' exact points
# And also we are performing cropping (bbox) part in this by using condition
# here we pass mask image , then put features in mask image
# applying condition for mask, crop : if want to then make it True / False 
def createBox(img, points, scale=5, masked=False, cropped=True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [points], (255,255,255))   # fill exact points, with white color
        # cv2.imshow("Mask", mask)
        # we need actual color of lips/eyes acc. to  points not white masking color
        img = cv2.bitwise_and(img, mask)    # so we use  "bitwise_and()"
        # cv2.imshow("Mask", img)

    if cropped:
        bbox = cv2.boundingRect(points)
        x,y,w,h = bbox
        imgCrop = img[y:y+h, x:x+w]    # its small so we resize it
        imgCrop = cv2.resize(imgCrop, (0,0), None, scale, scale)    # using scale bcoz user can give value acc. it will enlarge image part
        return imgCrop

    # if we are not cropping
    else:
        return mask


# 1.
while True:
    if webcam: success, img = cap.read()                 # if want to use webcam
    else : img = cv2.imread("files/elon_musk.jpg")
    img = cv2.resize(img, (0,0), None, 1,1)    # 1/4th of image 0.5, 0.5  instead of 1,1
    imgOriginal = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # gray image
    faces = detector(imgGray)     # detect faces

    # 2.create bbox around face and find landmarks
    for face in faces:
        x1,y1 = face.left(), face.top()
        x2,y2 = face.right(), face.bottom()
        # imgOriginal = cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)    # draw rectangle
        # detect facial landmarks , using : img, face object, by these landmarks we can crop/separate these parts
        landmarks = predictor(imgGray, face)
        myPoints = []    # put x,y in list to use easily
        for n in range(68):              # draw landmarks
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x,y])
            # create circle and put text to know points 'comment it out' after knowing points
            # cv2.circle(imgOriginal, (x,y), 2, (50,50,255), -1)
            # cv2.putText(imgOriginal, str(n), (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.2, (0,0,0), 1)
        # print(myPoints)

        # we have to 1st convert 'myPoints' into numpy array
        myPoints = np.array(myPoints)
        # left eye
        # imgLeftEye = createBox(img, myPoints[36:42])   # 42 excluded , create box around left eye
        # cv2.imshow("Left Eye",imgLeftEye)
        # lips
        imglips = createBox(img, myPoints[48:61], 10, masked=True, cropped=False)

        # color the lips , using trackbar values
        imgColorLips = np.zeros_like(imglips)
        b = cv2.getTrackbarPos("Blue", "BGR")
        g = cv2.getTrackbarPos("Green", "BGR")
        r = cv2.getTrackbarPos("Red", "BGR")
        imgColorLips[:] = b,g,r
        # merging mask lips image and color lips
        imgColorLips = cv2.bitwise_and(imglips, imgColorLips)
        
        # Now we add original image and color lips image
        imgColorLips = cv2.GaussianBlur(imgColorLips, (7,7), 10)    # adding blur at edges so it will look good not so sharp
        imgOriginalGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)   # converting original img into gray scale so we can see results clearly for lips color , gray scale is 1 channel
        imgOriginalGray = cv2.cvtColor(imgOriginalGray, cv2.COLOR_GRAY2BGR)    # we will convert back to BGR into 3 channel image so we can add both images
        imgColorLips = cv2.addWeighted(imgOriginalGray, 1, imgColorLips, 0.4, 0)    # add images acc. to weight : 1st original image full '1' , 2nd color lips image '0.4' 40% , gamma = 0
        # cv2.imshow("Colored",imgColorLips)
        cv2.imshow("BGR",imgColorLips)             # using trackbar

        # cv2.imshow("Lips",imglips)


    cv2.imshow("image", imgOriginal)
    if cv2.waitKey(1) == 27:
        break

