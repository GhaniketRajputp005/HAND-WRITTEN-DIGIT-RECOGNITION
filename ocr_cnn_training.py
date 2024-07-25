
#                                  HAND WRITTEN DIGIT RECOGNITION PYTHON PROJECT
                                    #******************************************#


 #CNN -> Convolutional Neural Networks are a type of Deep Learning Algorithm that take the image as an input and learn the various features of the image through filters.
 #This allows them to learn the important objects present in the image, allowing them to discern one image from the other


import numpy as np # python library used for working with arrays
import cv2  # designed to solve computer vision problems
import os   #Python os system function allows us to run a command in the Python script
from sklearn.model_selection import train_test_split  #Split arrays or matrices into random train and test subsets.
import matplotlib.pyplot as plt  #pyplot is mainly intended for interactive plots
from keras.preprocessing.image import ImageDataGenerator # takes i/p of original data, then transform it on a random basis,returning the o/p
from keras.utils.np_utils import to_categorical # to categorical converts classes to binary matrix
from keras.models import Sequential # it is a model each layer has exactly one input and output and is stacked together to form the entire network.
from keras.layers import Dense #the regular deeply connected neural network layer,to classify image based on output from convolutional layers
from keras.optimizers import Adam # algorithm that can be used  to update network weights iterative based in training data
from keras.layers import Dropout, Flatten #A Simple Way to Prevent Neural Networks from Overfitting,
#Flattens the input. Does not affect the batch size
from keras.layers.convolutional import Conv2D, MaxPooling2D 
#This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of output,
# #Downsamples the input along its spatial dimensions
import pickle #converting a Python object into a byte stream to store it in a file/database.


#   Keras is a high-level, deep learning API developed by Google for implementing neural networks.
#  It is written in Python and is used to make the implementation of neural networks easy.
#  It also supports multiple backend neural network computation.
#The TensorFlow platform helps you implement best practices for data automation, model tracking,
#  performance monitoring, and model retraining

###############################################################
path = 'myData'
pathLabels = 'Labels.csv'
count = 0
Images = []
classNo = []
noOfSamples = []
test_ratio = 0.2
validation_ratio = 0.2
imageDimensions = (32, 32, 3)
batchSizeValue = 50
epochsvalue =50
stepsPerEpochVal = 2000

##############################################################


################## WORKING ON DATASET #########################

myList = os.listdir(path)
print("Total No of Classes Detected = ", len(myList))
noOfClasses = len(myList)

print("******Importing Classes******")
for x in range(0, noOfClasses):
    # INSIDE MYDATA FOLDER GO INTO ZERO AND OTHER FOLDERS AND
    # INSIDE WHATEVER NAMES YOU FIND PUT  THEM IN A LIST

    myPicList = os.listdir(path + '/' + str(count))
    # THIS FUNCTION OPENS THE FOLDERS IN MYDATA FOLDER

    for y in myPicList:
        currentImg = cv2.imread(path + '/' + str(count) + '/' + y)
        # THIS STATEMENT READS IMAGES FROM FOLDER
      #  currentImg = cv2.resize(currentImg, (imageDimensions[0], imageDimensions[1]))
        currentImg = cv2.resize(currentImg, (32,32))

        # RESIZING OUR IMAGE COZ RIGHT NOW OUR IMAGES ARE 128*128 AND
        # IS TOO LARGE FOR NETWORK AND COMPUTATIONALLY EXPENSIVE
        Images.append(currentImg)
        classNo.append(count)
        # SAVE THE CLASS ID OF EACH OF THESE IMAGES
    print(count, end=" ")
    count += 1
print(" ")
print("Total Images in Images List = ", len(Images))
print("Total IDS in classNo List= ", len(classNo))

images = np.array(Images)
classNo = np.array(classNo)
# INITIALLY IMAGES WAS A LIST OF IMAGES BUT NP.ARRAY() CHANGES IT TO TUPLE
# TABLE->ELEMENTS->OF SAME TYPE->INDEXED BY TUPLE OF POSITIVE INTEGERS
print(images.shape)
print(classNo.shape)

########   SPLITTING THE DATA   #########

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=test_ratio)
# SPLITS THE ORIGINAL DATA RANDOMLY AND EVENLY ,
# FOR BETTER PROCESSING, 80% FOR TRAINING 20% FOR TESTING IF (0.2)*
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_ratio)
# SPLITS THE TRAINING DATA FOR VALIDATION
# 80% OF PREVIOUS TRAINING DATA TO CURRENT TRAINING DATA REST FOR VALIDATION
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(X_validation.shape)

for x in range(0, noOfClasses):
    # LOOKING FOR NUMBER OF IMAGES OF EACH CLASSES AS IT IS NECESSARY TO CHECK BIASING
    # SINCE X_TRAIN CONTAINS IMAGES AND Y_TRAIN CONTAINS ID'S OF IMAGES
    # SO IT'LL BE EFFICIENT TO CHECK USING Y_TRAIN
    noOfSamples.append(len(np.where(y_train == x)[0]))
    # np.where() RETURNS A LIST OF INDICES WHERE THE CONDITIONS HAVE BEEN MET
    # len(np.where(y_train == x)[0]) ,
    # IT'LL GIVE ME THE NUMBER OF SAME TYPE OF IMAGE (0->9) PRESENT IN Y_TRAIN AS LIST
print(noOfSamples)

######## CREATING A BAR GRAPH ##########
plt.figure(figsize=(10, 5))
plt.bar(range(0, noOfClasses), noOfSamples)
plt.title("No of Images for each Class")
plt.xlabel("Classes ID")
plt.ylabel("No of Images")
plt.show()


######## PREPROCESS ALL THE  IMAGES #######

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CONVERTING TO GRAY MAKES THE LIGHTING OF THE IMAGE TO DISTRIBUTE EVENLY
    img = cv2.equalizeHist(img)
    # GRAY IMAGE HAS SINGLE CHANNEL ,USE TO IMPROVE CONTRAST IN IMAGES
    img = img / 255
    #  When using the image as it is and passing through a Deep Neural Network
    #  The computation of high numeric values may become more complex
    #  To reduce this we can normalize the values to range from 0 to 1.
    #  Numbers will be small and the computation becomes easier and faster
    return img


X_train = np.array(list(map(preprocessing, X_train)))
X_test = np.array(list(map(preprocessing, X_test)))
X_validation = np.array(list(map(preprocessing, X_validation)))
# MAP FUCTION PASSES X_TRAIN TO PREPROCESSING FUNCTION,
# RETURNS->LIST , NP.ARRAY CONVERTS LIST TO TUPLE
print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
# adding depth of 1(channel->1),required for neural network to run properly
print(X_train.shape)

##### GENERATING AUGMENTED IMAGES #####
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

dataGen.fit(X_train) #Fits the data generator to some sample data

y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
# to categorical converts classes to binary matrix

def mymodel():  # using LENET CNN
    noOffilters = 60
    sizeOffilter1 = (5, 5)
    sizeOffilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNode = 500

    model = Sequential()  # building model
    model.add((Conv2D(noOffilters, sizeOffilter1, input_shape=(imageDimensions[0], imageDimensions[1], 1),
                      activation='relu'))) #Adds a layer instance on top of the layer stack.
    # relu is an activation function , it does not activate all the neurons at the same time
    model.add((Conv2D(noOffilters, sizeOffilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOffilters // 2, sizeOffilter2, activation='relu')))
    model.add((Conv2D(noOffilters // 2, sizeOffilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))
    # dropout layers helps in making data  more generic and reduce overfitting
    model.add(Flatten())
    model.add(Dense(noOfNode, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation="softmax"))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy']) #Configures the model for training.

    return model


model = mymodel()
print(model.summary())


# this will start the training for us
history = model.fit_generator(dataGen.flow(X_train, y_train,  #Takes data & label arrays, generates batches of augmented data.
                                 batch_size=batchSizeValue),
                    steps_per_epoch=stepsPerEpochVal,
                    validation_data=(X_validation, y_validation),
                    epochs= epochsvalue)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test, y_test, verbose=0) #Returns the loss value & metrics values for the model in test mode.
print('Test Score = ', score[0])
print("Test Accuracy = ", score[1])

pickle_out = open("model_trained_50.p", "wb")#wb=write bytes,A .P file is a pickle file, used for the input or output of data in a pictured format
pickle.dump(model, pickle_out) #dump() function to store the object data to the file.
pickle_out.close()