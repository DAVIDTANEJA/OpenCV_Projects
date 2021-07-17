# Now we test the model , use new image and classify the image using model.
from _typeshed import HasFileno
import numpy as np
import cv2
import pickle


width = 640
height = 480
cap = cv2.VideoCapture(0)      # using webcam
cap.set(3, width)
cap.set(4, height)
threshold = 0.8         # 80%    -change acc.


# load the model
pickle_model = open("digits_classification.p", "rb")
model = pickle.load(pickle_model)

# preprocess the image
def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)    # equalize the image (makes the light of image distribute evenly)
    img = img/255    # normalize the image 0 to 1 , bcoz in grayscale image 0 to 255 so divide by 255
    return img


while True:
    _, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)     # convert image into array
    img = cv2.resize(img, (32,32))    # resize 32x32
    img = preProcess(img)             # preProcess()
    cv2.imshow("Processed Image", img)
    
    # Predict
    img = img.reshape(1,32,32,1)  # reshape image before the predictor
    classIndex = int(model.predict_classes(img))
    # print(classIndex)
    predictions = model.predict(img)    # give predictions values
    probVal = np.amax(predictions)      # get the highest value

    if probVal > threshold:       # change threshold value acc.
        cv2.putText(imgOriginal, str(classIndex)+" | " + str(probVal), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)


    cv2.imshow("Original Image", imgOriginal)
    if cv2.waitKey(1) == 27:
        break

