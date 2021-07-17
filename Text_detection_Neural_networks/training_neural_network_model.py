# we have 0 to 9 images into separate folders acc. to numbers 0 to 9 , total around 10000 images
# 1.Read the images folder and put all images into 1 list , Remeber we have (180x180) image we will resize it
# 2.convert images into numpy array
# 3.split the Data (training, testing, validation)
# 4.Plot the numofSamples (total images of each class) , Check Data is distributed evenly
# 5.preprocess the image : we will preprocess 'x_train' images using "map()" which run over list / array of elements
# 6.Add depth of image , for convolution network
# 7.image augmentation (add rotation, zoom, shift, translation to image)
# 8."One Hot encoding" of matrix
# 9.create model , compile the model , fit the model (Note : change parameters acc. "batch size, epochs, steps") , check score of the model
# 10.save the model (use it later) in -> "detect_using_model.py"



import numpy as np
import cv2
import os
from numpy.core.fromnumeric import size
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
import pickle


# 1.Read the images folder and put all images into 1 list , Remeber we have (180x180) image we will resize it
path = "myData"
myList = os.listdir(path)
# print(myList)
noOfClasses = len(myList)
# it reads all folders images and 1st resize it and store in a list, # store class ID for each image acc. to folder 0,1,2..
images = []
classNo = []    # x = 0 it will take class Id for image 0 , if x=1 then class Id = 1 , and so on..
for x in range(0, noOfClasses):     # x = 0,1,2.... all folders
    myPicList = os.listdir(path+"/"+str(x))    # and it will have path of all images of each folder
    # print(len(myPicList))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)   # it have all images
        curImg = cv2.resize(curImg, (32,32))         # resize image
        images.append(curImg)
        classNo.append(x)      # store class ID for each image acc. to folder 0,1,2..
# print(len(images))
# print(len(classNo))


# 2.convert images into numpy array
images = np.array(images)
classNo = np.array(classNo)
# print(images.shape)          # (total images , 32,32, 3)  3- RGB colored img


# 3. split the Data (training, testing, validation)
testRatio = 0.2    # 20%
valRatio = 0.2
x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)    # test_size=20%, train_size=80% of data images, (x_train have images, y_train have Ids of each image)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=valRatio)  # now we split 80% training data for validation
# print(x_train.shape, x_test.shape, x_validation.shape)

# np.where(y_train == 0)  # give index where 0 is present
# print(len(np.where(y_train==0)[0]))  # total no. of 0 images
numOfSamples = []
for x in range(0, noOfClasses):
    numOfSamples.append(len(np.where(y_train==x)))    # total no. of images of each class 0,1,2....

# 4.Plot the numofSamples (total images of each class) , Check Data is distributed evenly
plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses), numOfSamples)
plt.title("No. of Images for each class")
plt.xlabel("Class Id")
plt.ylabel("Total no. of Images")
plt.show()


# 5.preprocess the image
def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)    # equalize the image (makes the light of image distribute evenly)
    img = img/255    # normalize the image 0 to 1 , bcoz in grayscale image 0 to 255 so divide by 255
    return img

# img = preProcess(x_train[25])     # now we can look at any image
# img = cv2.resize(img, (300,300))  # resizing bcoz image is 32x32
# cv2.imshow("Preprocessed", img)
# cv2.waitKey(0)

# Now we will preprocess 'x_train', 'x_test', 'x_validation' - images using "map()" which run over list / array of elements
x_train = np.array(list(map(preProcess, x_train)))  # take image 1 by 1 from x_train and preprocess it and store in list. Note when we have images we have to convert into numpy array
x_test = np.array(list(map(preProcess, x_test)))
x_validation = np.array(list(map(preProcess, x_validation)))
# print(x_train[25].shape)   # before preprocessing its 3 channel, but now its grayscale so its 1 channel

# 6.Add depth of image , for convolution network
# print(x_train.shape)    # before the depth
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)  # initial parameters remain same, and in 3rd one we add 1
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)  # initial parameters remain same, and in 3rd one we add 1
x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)  # initial parameters remain same, and in 3rd one we add 1
# print(x_train.shape)    # after adding depth


# 7.image augmentation (add rotation, zoom, shift, translation to image)
dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)  # 0.1 = 10% shift , rotation range = 10 degree
# calculate statistics before transformation
dataGen.fit(x_train)  # we do not generate images before the training , we generate them during the process , So we want to know generator about dataset before training process, images request will be in batches

# 8."One Hot encoding" of matrix
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)


# 9.create model , compile the model , fit the model , check score of the model
imageDimensions = (32,32,3)
def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNode = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0], imageDimensions[1], 1), activation='relu')))  # 1st add convolutional layer, 
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))  # 2nd convolutional layer, 

    model.add(MaxPooling2D(pool_size=sizeOfPool))                          # pooling layer

    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))  # 3rd convolutional layer , noOfFilters//2 
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))  # 4th convolutional layer

    model.add(MaxPooling2D(pool_size=sizeOfPool))                          # pooling layer
    model.add(Dropout(0.5))    # dropout layer , 50%

    model.add(Flatten())        # flatten
    model.add(Dense(noOfNode, activation='relu'))  # dense layer
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])      # compile the model

    return model

# call the model
model = myModel()
# print(model.summary())     # model summary 

# fit the model -start the training , (Note : change parameters acc. "batch size, epochs, steps")
batchSizeVal = 50
epochsVal = 10
stepsPerEpoch = 2000
history = model.fit_generator(dataGen.flow(x_train, y_train, batch_size=batchSizeVal), steps_per_epoch=stepsPerEpoch, epochs=epochsVal, validation_data=(x_validation, y_validation), shuffle=1)     # fit the model


# plot the loss , accuracy
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title("Loss")
plt.xlabel("No. of epochs")

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title("Accuracy")
plt.xlabel("No. of epochs")
plt.show()

# check score of the model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test Score : ", score[0])
print("Accuracy : ", score[1])


# 10.save the model
pickle_out = open("digits_classification.p", "wb")    # filename
pickle.dump(model, pickle_out)
pickle_out.close()




