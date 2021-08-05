import cv2
import numpy as np

# image1 = cv2.imread("files/1.png")
# image2 = cv2.imread("files/copy2.jpg")
# image1 = cv2.resize(image1, (500,500))
# image2 = cv2.resize(image2, (500,500))

# image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# print(image1.shape)
# print(image2.shape)

# difference = cv2.subtract(image1, image2)

# result = np.any(difference)  # if images are same / difference is '0' it will return 'False'

# if result:
#     cv2.imwrite("results.jpg", difference)
#     print("Images are different")
# else:
#     print("Images are same")
  
# cv2.waitKey(0)


# -----------------------------------------------------------------------------------------------------
# 2nd method

# from skimage.metrics import structural_similarity
# import imutils

# image1 = cv2.imread("files/1.png")
# image2 = cv2.imread("files/copy2.jpg")
# image1 = cv2.resize(image1, (500,200))       # 1.Make sure images are equal so resize it.
# image2 = cv2.resize(image2, (500,200))
# # print(image1.shape)
# # print(image2.shape)

# # start_frame = cv2.vconcat((image1, image2))
# # cv2.imshow("concat", start_frame)

# gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)      # 2.convert into grayscale
# gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# print(gray_image1.shape)
# print(gray_image2.shape)

# (score, diff) = structural_similarity(gray_image1, gray_image2, full=True)
# diff = (diff * 255).astype("uint8")
# print("Similarity index : ", score)

# thresh = cv2.threshold(diff, 0, 100, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]        # change threshold value acc.
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)

# no_of_diff = 0
# for c in cnts:
#     (x,y,w,h) = cv2.boundingRect(c)
#     area = w*h
#     if area > 10:
#         no_of_diff += 1
#         cv2.rectangle(image1, (x,y), (x+w, y+h), (0,0,255), 2)
#         cv2.rectangle(image2, (x,y), (x+w, y+h), (0,0,255), 2)

# print("Number of difference", no_of_diff)

# # if score < 1:
# #     cv2.imwrite("image1.png", image1)
# #     cv2.imwrite("image2.png", image2)
# #     cv2.imwrite("differ.png", diff)

# # final_frame = cv2.vconcat((image1, image2))
# # cv2.imshow("Output", final_frame)

# cv2.imshow("image 1", image1)
# cv2.imshow("image 2", image2)
# cv2.imshow("diff image", diff)

# cv2.waitKey(0)

# -----------------------------------------------------------------------------------------------------

from PIL import Image ,ImageChops
img1 = Image.open("files/1.png")
img2 = Image.open("files/copy2.jpg")
img1 = img1.resize((500,200))       # 1.Make sure images are equal so resize it.
img2 = img2.resize((500,200))
print(img1.size)
print(img2.size)

diff = ImageChops.difference(img1,img2)
if diff.getbbox():
	diff.show()
