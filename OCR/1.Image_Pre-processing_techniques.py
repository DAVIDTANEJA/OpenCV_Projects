import cv2
import numpy as np

img = cv2.imread('files/page.jpg')
# img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # blur used not for text, but to find structure of image so we can cut out required part columns/rows acc.

# 1.inverted image : make white into black part and black into white part
invert_img = cv2.bitwise_not(img)


# 2.Binarization - black-white image using threshold
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh, img_bw = cv2.threshold(gray_img, 200, 230, cv2.THRESH_BINARY)  # img_bw - holds the black-white image
# thresh, img_bw2 = cv2.threshold(gray_img, 200, 230, cv2.THRESH_BINARY_INV)  # inverted threshold image


# Note : when font looks thick / thin use -> dilate and erode ,  But 'morphologyEx and medianBlur' used when noise in background.

# 3.Noise removal
def noise_removal(image):
    kernel = np.ones((1,1), np.uint8)  # kernel (1,1) (3,3) (5,5)
    image = cv2.dilate(image, kernel, iterations=1)  # dilate
    kernel = np.ones((1,1), np.uint8)  # kernel (1,1) (3,3) (5,5)
    image = cv2.erode(image, kernel, iterations=1)   # erode, iterations - is for how many times this kernel run over on image
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # close morphology , open morphology cv2.MORPH_CLOSE
    image = cv2.medianBlur(image, 3)
    return image

noise_remove_img = noise_removal(img_bw)    # here we are passing binary image "img_bw"


# 4.Dilate and erode : adjust the font sizes 'thick / thin fonts' ,  Remember : To use this we need "inverted image black-white"
# in "dilate" - 'kernel' increase thickness of font  ,  in "erode"  - 'kernel' decrease thickness of font
# erode
def thin_font(image):
    image = cv2.bitwise_not(image)    # invert image
    kernel = np.ones((2,2), np.uint8)     # increase the kernel size it will more thin font
    image = cv2.erode(image, kernel, iterations=1)               # if increase number of iteration make more thin font
    image = cv2.bitwise_not(image)     # re invert back image
    return image

# dilate
def thick_font(image):
    image = cv2.bitwise_not(image)    # invert image
    kernel = np.ones((2,2), np.uint8)     # increase the kernel size it will more thin font
    image = cv2.dilate(image, kernel, iterations=1)               # if increase number of iteration make more thin font
    image = cv2.bitwise_not(image)     # re invert back image
    return image


erode_img = thin_font(noise_remove_img)     # using noise removed image
dilate_img = thick_font(noise_remove_img)     # using noise removed image


# 5.rotation / deskewing : mostly used for "pdf / image file"
# 1st Calculate skew angle of an image
def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

# 2nd Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# 3rd Deskew image : calls both above function
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)


# finally
fixed_img = deskew("rotate_img.jpg")     # use here rotated image, to startighten it




# 6.remove borders / create borders
# remove border
def remove_border(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntSorted = sorted(contours, key=lambda x : cv2.contourArea(x))
    cnt = cntSorted[-1]
    x,y,w,h = cv2.boundingRect(cnt)
    crop_img = image[y:y+h, x:x+w]
    return crop_img

remove_border_img = remove_border(noise_remove_img)

# create border
brdr = cv2.copyMakeBorder(img, 20, 10, 5, 5, cv2.BORDER_CONSTANT, value=(0,125,176))  # (image, top, bottom, left, right, bordertype, color value)


# cv2.imshow("Original image", img)
# cv2.imshow("invert image", invert_img)
cv2.imshow("binary image", img_bw)
cv2.imshow("noise removal", noise_remove_img)
cv2.imshow("erode image", erode_img)
cv2.imshow("dilate image", dilate_img)
cv2.waitKey(0)


# -----------------------------------------------------------------------------------------------
# Create bounding box  : 1.set kernel values acc.  and  2.adjust height and width acc.
import pytesseract
import cv2
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' 
tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'

# 1.Here we take an image and separate that into 3 sections divided using bounding box
image = cv2.imread("files/sample.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,10))  #(col, row) (3,30) (3,40) as required increase acc. it will create bounding box.
dilate = cv2.dilate(thresh, kernel, iterations=1)
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if h < 20 and w > 250:                              # Also adjust the height and width acc. : h < 20 , h > 100
        roi = image[y:y+h, x:x+w]
        cv2.rectangle(image, (x,y), (x+w, y+h), (36, 255, 12), 2)


cv2.imshow("output", cv2.resize(image,(700,700)))
cv2.waitKey(0)


# -----------------------------------------------------------------------------------------------
# Text from PDF documents
from pdf2image import convert_from_path    # pip install pdf2image
from IPython.display import display, Image

images = convert_from_path('functionalSample.pdf')


display(images[0])

