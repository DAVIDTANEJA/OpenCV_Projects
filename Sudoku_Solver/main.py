# all the maths funcctionalities performed in  "math_functions.py"  to solver sudoku problem.
# all funcions needed for image process are in  "helpers.py". Remember to create model / download  "digits classification model".
# In "main.py" we solve the problem and use all functionalities here from 'helpers.py' and 'math_functions.py'

# Steps : 
# 1.preprocess image(grayscale, threshold, blur) we can also add dilation / erosion as well
# 2.find contours
# 3.find biggest contour sudoku image, then reorder points and apply warp perspective to get perfect square image
# 4.classify all digits which already filled in sudoku (Its important otherwise we will get wrong answers / not any answer)
# like : split the 9x9 image into single 81 boxes and using digit classification model to predict/find each available answer 1 to 9 we don't have 0 but blank
# 5.find solution and fill spaces with solution digits
# 6.overlay solution to the original image.

# to load tensorflow
print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from helpers import *
import math_functions

########################################################################
pathImage = "files/1.jpg"     # load the image and it has to be square so height, width should be same
heightImg = 450
widthImg = 450
model = intializePredictionModel()    # digits classification model
########################################################################


# 1.preprocess images
img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))             # resize to square image
imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
imgThreshold = preProcess(img)


# 2.find contours
imgContours = img.copy()      # copy image
imgBigContour = img.copy()
contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_EXTERNAL - method for outer contours in image
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)     # draw contours


# 3.find biggest contour(sudoku image) and reorder points and apply warp perspective
biggest, maxArea = biggestContour(contours)    # biggest contour/corner points , max area of contours - rectangle/square
print(biggest)

if biggest.size != 0:
    biggest = reorder(biggest)    # reorder()  points for biggest contour
    # print(biggest)
    cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25)    # draw biggest contour

    pts1 = np.float32(biggest)    # prepare points for warp
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])  # prepare points for warp
    matrix = cv2.getPerspectiveTransform(pts1, pts2)    # find the matrix
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))       # warp perspective
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

    # 4. split the 9x9 image into single 81 boxes and using digit classification 'model' to predict/find each available answer 1 to 9 we don't have 0 but blank
    imgSolvedDigits = imgBlank.copy()
    boxes = splitBoxes(imgWarpColored)    # splitBoxes() into 81 boxes / images
    # print(len(boxes))
    # cv2.imshow("Sample",boxes[65])
    numbers = getPrediction(boxes, model)    # getPrediction() - send it here to predict each image
    # print(numbers)
    imgDetectedDigits = imgBlank.copy()
    imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))  # display numbers on blank image
    
    numbers = np.asarray(numbers)     # convert into array
    # find all numbers where 'number > 0' / where already number it replaces with 0 and where blank puts 1 - later we find soltuion for these 1's
    posArray = np.where(numbers > 0, 0, 1)  # we need this bcoz we want to display on image otherwise no need of this
    # print(posArray)


    # 5. find solution of board sudoku and fill with those digits in place of 1 , here we use "math_functions"
    board = np.array_split(numbers, 9)    # 1st create board with numbers=0,1
    # print(board)
    try:
        math_functions.solve(board)
    except:
        pass
    # print(board)    # after applying solve() method

    flatList = []            # after finding solution, convert this array into list and later we will display numbers
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    # posarray has 0,1 : when 0 multiply with numbers becomes 0 and it makes it blank space, and when multiply 1 with 'solution' digits replace with those numbers
    solvedNumbers =flatList * posArray
    imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)  # display solution numbers/answers


    # 6. overlay solution on original image
    # we will flip 'pts1' and 'pts2' to fill with solution by creating new matrix
    pts2 = np.float32(biggest)    # this was earlier 'pts1' above
    pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))  # warp perspective with new matrix

    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)    # add both 'imgInvWarpColored'  and  'img'
    imgDetectedDigits = drawGrid(imgDetectedDigits)    # drawGrid() - draw lines
    imgSolvedDigits = drawGrid(imgSolvedDigits)

    imageArray = ([img, imgThreshold, imgContours, imgBigContour],
                  [imgDetectedDigits, imgSolvedDigits, imgInvWarpColored, inv_perspective])
    stackedImage = stackImages(imageArray, 1)
    cv2.imshow('Stacked Images', stackedImage)

else:
    print("No Sudoku Found")        # if don't find biggest contour

cv2.waitKey(0)

