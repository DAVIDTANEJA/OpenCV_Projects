import cv2
import numpy as np


# display images
def stackImages(imgArray,scale,lables=[]):
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
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        #print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver


# reorder of points ((0,0)(w,0),(0,h),(w,h))  for warping
# biggest contour has points (4,1,2) - we need (4,2) and we use it acc. and create new matrix (4,1,2) and after reordering fix it this way
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))   # reshape contours points (4,2) : 4 rectangle contour and each contour has 2 points x,y
    # print(myPoints)
    myPointsNew = np.zeros((4, 1, 2), np.int32)  # new matrix
    add = myPoints.sum(1)    # it will add 4 (x,y) points and we check points
    # print(add)             # this will show us list with smallest, biggest points and we do indexing acc. [0] [1] [2] [3]
    # print(np.argmax(add))
    myPointsNew[0] = myPoints[np.argmin(add)]  #[0,0]   np.argmin()- smallest sum will be this
    myPointsNew[3] =myPoints[np.argmax(add)]   #[w,h]   np.argmax()- highest sum
    diff = np.diff(myPoints, axis=1)
    # print(diff)
    myPointsNew[1] =myPoints[np.argmin(diff)]  #[w,0]    # similarly find diference and remaining with +ve one is width
    myPointsNew[2] = myPoints[np.argmax(diff)] #[h,0]    # and -ve  one height

    return myPointsNew


# 2 - filter for rectangle contours checking if it has 4 points, by this we can get largset or 2nd largest rectangle contours/corner points by checking 'area'
def rectContour(contours):
    rectCon = []        # list of all rectangle contours
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)    # find perimeter for contour 'i' and  True - for closed one
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # polygon -how may corner points it has , resolution : 0.02*peri -change acc., True - for closed contour
            # print(len(approx))
            if len(approx) == 4:                              # len = 4 for rectangle, append in a list
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)    # sort biggest area rectangle, reverse=True -descending order
    #print(len(rectCon))
    return rectCon

# 2 - find corner points for rectangle - biggest , 2nd largest
def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)    # length of contour
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)    # approximate polygon for corner points
    return approx

# 4 - split boxes to check which circless are marked and which are not marked.
def splitBoxes(img):
    rows = np.vsplit(img,5)  # rows split, 5 - no. of splits
    boxes=[]
    for r in rows:               # now split rows horizontally and save into list
        cols= np.hsplit(r,5)     # 5 - no. of splits
        for box in cols:
            boxes.append(box)
    return boxes

def drawGrid(img,questions=5,choices=5):
    secW = int(img.shape[1]/questions)
    secH = int(img.shape[0]/choices)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)

    return img

# 5 - display answers on (img, user answers index no. , grading , actual answers, quest., choices)
def showAnswers(img, myIndex, grading, ans, questions=5, choices=5):
    secW = int(img.shape[1]/questions)  # img=500x500 , divide by 5 , become width=100 , so 
    secH = int(img.shape[0]/choices)    # height=100 , so image of box will be 100x100
    # Now we can create circle at that point, and show ans color green if correct , otherwise red
    for x in range(0,questions):
        myAns= myIndex[x]                   # answer index no.
        cX = (myAns * secW) + secW // 2     # cX, cY - center point for circle
        cY = (x * secH) + secH // 2
        if grading[x]==1:                     # if grading = 1 -its correct ans and show green , else red
            myColor = (0,255,0)    # green
            #cv2.rectangle(img,(myAns*secW,x*secH),((myAns*secW)+secW,(x*secH)+secH),myColor,cv2.FILLED)
            cv2.circle(img,(cX,cY),50,myColor,cv2.FILLED)
        else:
            myColor = (0,0,255)    # red , display wrong 'ans' in red
            #cv2.rectangle(img, (myAns * secW, x * secH), ((myAns * secW) + secW, (x * secH) + secH), myColor, cv2.FILLED)
            cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)

            # also display correct answer in green
            myColor = (0, 255, 0)
            correctAns = ans[x]
            cv2.circle(img,((correctAns * secW)+secW//2, (x * secH)+secH//2), 40,myColor,cv2.FILLED)
