# 1.Here we take an image and create bounding box and separate that into 3 sections divided
# 2.store into 'list' separate
# 3.then take text out of it and then use NLP (Named entity Recognition)

import pytesseract
import cv2
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' 
tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'


# 1.Here we take an image and create bounding box separate that into 3 sections divided using bounding box
image = cv2.imread("files/index.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7,7), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))  # (col, row) (3,30) (3,40) as required increase acc. it will create bounding box.
dilate = cv2.dilate(thresh, kernel, iterations=1)
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])

# for c in cnts:
#     x,y,w,h = cv2.boundingRect(c)
#     if h > 200 and w > 250:
#         roi = base_image[y:y+h, x:x+w]
#         cv2.rectangle(image, (x,y), (x+w, y+h), (36, 255, 12), 2)


# 2.store into 'list' separate
results  = []  # list
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if h > 200 and w > 20:
        roi = image[y:y+h, x:x+h]
        cv2.rectangle(image, (x, y), (x+w, y+h), (36, 255, 12), 2)
        ocr_result = pytesseract.image_to_string(roi, config=tessdata_dir_config)
        ocr_result = ocr_result.split("\n")
        for item in ocr_result:
            results.append(item)

            
cv2.imshow("Output image", cv2.resize(image, (500,500)))  # just before display resize image
cv2.waitKey(0)



# 3.then take text out of it and then use NLP (Named entity Recognition)
entities = []
for item in results:
    item = item.strip().replace("\n", "")
    item = item.split(" ")[0]
    if len(item) > 2:
        if item[0] == "A" and "-" not in item:                                 # taking only capital "A" 
            item = item.split(".")[0].replace(",", "").replace(";", "")
            entities.append(item)

print(entities)  # it has same names several times

entities = list(set(entities))  # it will remove copies / several names and get the single name
entities.sort()
