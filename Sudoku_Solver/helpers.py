import cv2
import numpy as np
from tensorflow.keras.models import load_model



# download from here : https://github.com/murtazahassan/OpenCV-Sudoku-Solver/blob/main/Resources/myModel.h5
# intialize the digits classification model
def intializePredictionModel():
    model = load_model('files/myModel.h5')
    return model


# 1.preprocess image(grayscale, blur, threshold) , we can also add dilation / erosion as well
def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)  # APPLY ADAPTIVE THRESHOLD
    return imgThreshold


# reorder() - reorder points for warp perspective
# we need points in this manner to get proper image : [0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]               # (0,0)
    myPointsNew[3] =myPoints[np.argmax(add)]                # add points (width, height)
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]               # take difference (width,0)
    myPointsNew[2] = myPoints[np.argmax(diff)]              # (height,0) / (width,0) depends on difference
    return myPointsNew


# find the biggest contour , loop through all contours by max_area and biggest contour of rectangle/square in every loop
def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:                # loop through all contours
        area = cv2.contourArea(i)              # we will check area , small is noise
        if area > 50:                          # so take condition
            peri = cv2.arcLength(i, True)      # find perimiter
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)    # and how many corners does it have
            if area > max_area and len(approx) == 4:           # we check for max_area and len= '4' here rectangle/square
                biggest = approx                               # biggest corners points -rectangle/square
                max_area = area                                # update max_area every loop
    return biggest,max_area


# 4 - split the 9x9 image into single 81 boxes / images , using vertical and horizontal split
def splitBoxes(img):
    rows = np.vsplit(img,9)     # 1st split vertical into 9 rows
    boxes=[]
    for r in rows:               # then each row split horizontally into 9 boxes
        cols= np.hsplit(r,9)
        for box in cols:
            boxes.append(box)     # and append all boxes
    return boxes


# 4 - get prediction on all images , we will take that digit which has 80% above prediction otherwise make it blank
def getPrediction(boxes, model):             # boxes - 81 image
    result = []
    for image in boxes:
        # preprocess images
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        img = cv2.resize(img, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)

        predictions = model.predict(img)             # prediction methods
        classIndex = model.predict_classes(img)
        classIndex = np.argmax(predictions, axis=-1)   # classification done from 0 to 9, need to know which class it belong to 'digits'
        probabilityValue = np.amax(predictions)        # probability value for prediction

        if probabilityValue > 0.8:              # digit which has 80% above prediction we append into 'results'
            result.append(classIndex[0])
        else:
            result.append(0)
    return result


# 4,5 -  display the solution on the image
def displayNumbers(img,numbers,color = (0,255,0)):       # img -on which image to display
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),
                               (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img


# 6 - draw grid / lines in matrix solution image
def drawGrid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)
    return img


# stack images
def stackImages(imgArray,scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    return ver