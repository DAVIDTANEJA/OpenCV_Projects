# Document scanner : 
# 1st take image -> then grayscale it -> then apply edge detector find edges -> find contours -> filter biggest contour
# -> then with biggest contour take corner points and use "warp perspective" and get the desired image.
# -> Then apply "Adaptive thresholding"  get the scanned paper like  and save image.
# Also use Trackbar to find threshold value


import cv2
import numpy as np
import utlis     # file created with functions we use here

########################################################################
webCamFeed = True
pathImage = "1.jpg"      # we can also image

# We are using mobile camera
camera = "http://100.67.110.153:8080/video"     # ip address shown in app, video - just name given
cap = cv2.VideoCapture(0)    # cv2.CAP_DSHOW  - pass this parameter if any error shows
cap.open(camera)

cap.set(10,160)        # brightness
heightImg = 640        # increase resolution for better result 1280 x 720
widthImg  = 480
########################################################################

utlis.initializeTrackbars()             # Remember : it will create "Trackbar" use it to find threshold value
count=0

while True:
    _, img = cap.read()

    # if webCamFeed:
    #     success, img = cap.read()           # either use webcam
    # else:
    #     img = cv2.imread(pathImage)         # or take image
    
    # 1.Preprocessing
    img = cv2.resize(img, (widthImg, heightImg))             # RESIZE IMAGE
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8)   # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)          # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)           # ADD GAUSSIAN BLUR

    thres=utlis.valTrackbars()                               # get 2 threshold values using trackbar and change acc.
    imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1])      # APPLY CANNY edge detector

    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) # Then apply dilation  and erosion
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # EROSION

    # 2.FIND ALL COUNTOURS
    imgContours = img.copy()    # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy()  # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)    # on "imgContours" - DRAW ALL DETECTED CONTOURS


    # 3.FIND THE BIGGEST COUNTOUR - like document
    biggest, maxArea = utlis.biggestContour(contours) # FIND THE BIGGEST CONTOUR
    if biggest.size != 0:                      # if find out the biggest contour
        biggest=utlis.reorder(biggest)        # "reorder()" the points in what we need
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)    # DRAW THE BIGGEST CONTOUR
        imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest)     # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))         # then we apply warp perspective

        # REMOVE 20 PIXELS FORM EACH SIDE - to remove edges from image
        imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))

        # 4.APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)           # grayscale 
        imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)    # it will give binary image
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)                      # reverse it make all 1 -> 0  and  all 0 -> 1
        imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)                       # remove noise if any, and then give scanned image

        # Image Array for Display
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigContour,imgWarpColored, imgWarpGray,imgAdaptiveThre])

    else:
        imageArray = ([img,imgGray,imgThreshold,imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])       # if don't find any contours

    # LABELS FOR DISPLAY
    labels = [["Original","Gray","Threshold","Contours"],
              ["Biggest Contour","Warp Prespective","Warp Gray","Adaptive Threshold"]]

    stackedImage = utlis.stackImages(imageArray, 0.5, labels)     # stack images (image array, scale images in %, labels)
    cv2.imshow("Result",stackedImage)

    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg",imgWarpColored)     # save the image in folder 'Scanned'
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1

    # if cv2.waitKey(1) == 27:
    #     break








