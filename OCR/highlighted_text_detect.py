# we will detect text from image which is highlighted and save text it to another file.
# Also created 'utlis.py' file for some functionalities.
# we are using 'color_picker.py' to detect highlighted text color we need and by getting hsv values we pass here.

from utlis import *
import pytesseract

# 1.define image and HSV color for highlighted text
path = 'test.png'
hsv = [0, 65, 59, 255, 0, 255]    # HSV[1st 3 lower, then higher values] : this is highlighted text color we picked using trackbar from "color_picker.py"

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'

# 2.read image
img = cv2.imread(path)

# 3.detect color from 'utlis'
imgResult = detectColor(img, hsv)

# 4. we find contours of highlighted text and create bbox around it, using 'utlis'
imgContours, contours = getContours(imgResult, img, showCanny=False, minArea=1000, filter=4, cThr=[100,150], draw=True)  # it should have minArea otherwise it will detect noise, filter - how many corners it should have here for rectangle need 4, canny threshold, draw bbox if True on img
# cv2.imshow("imgContours", imgContours)
print(len(contours))

# 5.crop this contours / higlighted text regions and convert into individual images - send to 'pytesseract' detect text.
roiList = getRoi(img, contours)    # from 'utlis' , contours - 3rd element of contours is bbox , apply to 'img' and get our 'roi' images, so roiList has all images of roi
# cv2.imshow("Test", roiList[1])   # 'roiList' has all images of roi
roiDisplay(roiList)    # function creted to display all roi images


# 6.Take all roi images and send to pytesseract and give text output
highlightedText = []    # add text values
for x, roi in enumerate(roiList):
    highlightedText.append(pytesseract.image_to_string(roi))    # 'pytesseract' read text
    # print(pytesseract.image_to_string(roi))

# 7.function to write/save text into 'csv' file
saveText(highlightedText)

# stack images if want to
# imgStack = stackImages(0.7, ([img, imgResult, imgContours]))
# cv2.imshow("Stacked images", imgStack)

cv2.waitKey(0)
