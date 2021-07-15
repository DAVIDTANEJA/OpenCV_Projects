# 1.img preprocess grayscale, blur, find edges
# 2.then find contours : biggest contour /rectangle corner points
# 3.warp perspective : to apply warping we need : 1st("reorder" of points (0,0)(w,0)(0,h)(w,h)) -then pts1, pts2 and transformation matrix
# 4. apply threshold : 
# which circles are marked have more pixels value and circles not marked will have less pixels values.
# find the User answers and put them in a list , iterate through all questions/rows , then 
# compare/check with original answers (user answers index no.)  and give grading acc.
# 5. display answers  and  grading marks on final image.


import cv2
import numpy as np
import helpers


########################################################################
webcam = True         # if want to use webcam otherwise make it "False"

pathImage = "5.jpg"
cap = cv2.VideoCapture(1)
cap.set(10,160)
heightImg = 700
widthImg  = 700
questions=5           # total no. of questions
choices=5             # no. of choices 
ans= [1,2,0,2,4]      # correct answers acc. to Index no.
########################################################################

count=0

while True:
    # 1.image preprocess
    if webcam: success, img = cap.read()
    else: img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg))     # resize    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale img
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)   # add blur
    imgCanny = cv2.Canny(imgBlur, 10, 70)            # find edges (img, threshold)

    imgFinal = img.copy()   # copy of original img - to display answers-grade marks
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # just for testing

    # if webcam not find contours/image try-except
    try:
        # 2.find contours
        imgContours = img.copy()    # for display
        imgBigContour = img.copy()  # for display
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # find all contours, cv2.RETR_EXTERNAL - find outer edges , no need of chain approximation
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)    # draw contours
        # filter for rectangle contours from 'helpers' functions and then find corner points for biggest and second largest rectangle contours
        rectCon = helpers.rectContour(contours)              # rectangle contours
        biggestPoints= helpers.getCornerPoints(rectCon[0])   # biggest rectangle corner points - which has all markings
        gradePoints = helpers.getCornerPoints(rectCon[1])    # second largest rectangle - grading marks

        # if the biggest rectangle , 2nd largest rectangle detected
        if biggestPoints.size != 0 and gradePoints.size != 0:
            # 3.warp perspective - biggest rectangle warping
            biggestPoints=helpers.reorder(biggestPoints)     # "reorder" of points ((0,0)(w,0)(0,h)(w,h))  for warping
            cv2.drawContours(imgBigContour, biggestPoints, -1, (0, 255, 0), 20)   # draw biggest contour
            # to apply warping we need pts1, pts2 and transformation matrix
            pts1 = np.float32(biggestPoints) 
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)    # transformation matrix
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))    # apply warp perspective

            # 2nd largest rectangle warping - grading marks "gradePoints"
            cv2.drawContours(imgBigContour, gradePoints, -1, (255, 0, 0), 20)  # draw 2nd largest contour
            gradePoints = helpers.reorder(gradePoints)    # reorder points
            # to apply warping we need pts1, pts2 and transformation matrix
            ptsG1 = np.float32(gradePoints)   # prepare for warping
            ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])  # we are not taking original width, height -take any acc. for this
            matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)    # get transformation matrix
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))  # apply warp perspective

            # 4.apply threshold - which circles are marked have more pixels value and circles not marked will have less pixels values.
            # Now we can find marking points and check answers correct / not and give grading marks acc. to that
            imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)               # grayscale
            imgThresh = cv2.threshold(imgWarpGray, 170, 255,cv2.THRESH_BINARY_INV )[1]  # threshold, change values acc. 170
            # split boxes and check which one are marked and which one are not marked
            boxes = helpers.splitBoxes(imgThresh)    # splitBoxes()
            # cv2.imshow("Split Test ", boxes[3])
            # it shows marked one has more pixels and not marked has less pixels , and we take them into array and check correct/not
            # print(cv2.countNonZero(boxes[1], cv2.countNonZero(boxes[2])))
            # Now we iterate through all boxes/images and check pixels values and store in array 5x5 of non-zero pixels values
            countR=0    # rows
            countC=0    # columns
            myPixelVal = np.zeros((questions,choices))  # and we store here into 5x5 array non-zero pixel values (5-questions , 5-circle answers)
            for image in boxes:                                   # iterate through all boxes/images
                #cv2.imshow(str(countR)+str(countC),image)
                totalPixels = cv2.countNonZero(image)             # and check pixels value
                myPixelVal[countR][countC]= totalPixels           
                countC += 1                                    # count column/image +1, whenever count reach to 5 it changes row
                if (countC==choices): countC=0; countR+=1     # count column = choices=5 it change row countR+=1 and iterate again countC=0
            # print(myPixelVal)

            # find the User answers and put them in a list , iterate through all questions/rows and find max. pixels value / marked circle and store its "index value" in list
            myIndex=[]      # store index no. of correct answers
            for x in range (0,questions):          # iterate each rows
                arr = myPixelVal[x]                # get each row in 'arr'
                myIndexVal = np.where(arr == np.amax(arr))  # find max pixel value in each row 'arr'
                myIndex.append(myIndexVal[0][0])            # append into list
            # print("USER ANSWERS",myIndex)

            # Now compare/check with original answers (user answers index no.)  and give grading acc. to that
            grading=[]
            for x in range(0,questions):
                if ans[x] == myIndex[x]:       # match actual 'ans' with user's 'answers'
                    grading.append(1)          # if ans is correct append 1 otherwise 0
                else:
                    grading.append(0)
            # print("GRADING", grading)
            score = (sum(grading)/questions)*100    # final score in %
            # print("SCORE", score)


            # 5.display answers
            helpers.showAnswers(imgWarpColored,myIndex,grading,ans)  # showAnswers()
            helpers.drawGrid(imgWarpColored)  # draw grid b/w rows-columns answer sheets
            imgRawDrawings = np.zeros_like(imgWarpColored) # new blank image of warp image size, to display answers colored
            helpers.showAnswers(imgRawDrawings, myIndex, grading, ans) # it will show only colored part
            # and we will take this new image "imgRawDrawings" and take inverse perspective then apply on original img.
            invMatrix = cv2.getPerspectiveTransform(pts2, pts1)    # inverse transformation matrix: pts1 as pts2 and pts2 as pts1
            imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg))  # inervse img warp

            # 5.display grade marks - blank image, put text grade marks on it, inverse perspective and add to original image
            imgRawGrade = np.zeros_like(imgGradeDisplay, np.uint8)  # new blank image of 'grade image size'
            cv2.putText(imgRawGrade,str(int(score))+"%",(70,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3)  # grade marks
            invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1)  # take inverse perspective
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))  # inverse img warp

            # show answers colored and grade marks on final image
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1,0)          # add original img , imgInvWarp of answers
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1,0)  # similarly of 'grade' marks

            # image array to display using 'stackImages'
            imageArray = ([img,imgGray,imgCanny,imgContours], [imgBigContour,imgThresh,imgWarpColored,imgFinal])
            cv2.imshow("Final Result", imgFinal)
    except:
        imageArray = ([img,imgGray,imgCanny,imgContours], [imgBlank, imgBlank, imgBlank, imgBlank])

    # labels for "stackImages"
    lables = [["Original","Gray","Edges","Contours"], ["Biggest Contour","Threshold","Warpped","Final"]]
    stackedImage = helpers.stackImages(imageArray, 0.5, lables)
    cv2.imshow("Result",stackedImage)

    # save the image in folder when 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("files/Saved/myImage"+str(count)+".jpg",imgFinal)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1



