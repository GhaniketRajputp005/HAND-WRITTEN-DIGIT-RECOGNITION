import numpy as np
import cv2
import pickle

width = 640
height = 480
threshold = .75


cap = cv2.VideoCapture(0)
cap.set(3,1260)
cap.set(4,720)

pickle_in = open("model_trained_50.p","rb")

model = pickle.load(pickle_in)

def preProcessing (img) :
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist (img) # GRAY IMAGE HAS SINGLE CHANNEL ,USE TO IMPROVE CONTRAST IN IMAGES
    img = img/255
    return img

while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32,32))
    img = preProcessing(img)
    cv2.imshow("Processsed Image", img)
    img = img.reshape(1,32,32,1)

#Predict

    classIndex = int(model.predict_classes(img))
    #print(classIndex)
    predictions = model.predict(img)
    #print(predictions)
    probVal= np.amax(predictions)  # Return the maximum of an array or maximum along an axis
    newprobVal=round(probVal,2)    # Round a number to a given precision in decimal digits
    print(classIndex,newprobVal)


    if probVal>threshold:
        cv2.putText(imgOriginal, " Digit = "+str(classIndex) + "   "+"Accuracy = " f"{100*probVal:.2f}"+str("%") , (5,50) ,cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0) ,2)


    cv2.imshow("Digit Prediction Corner...",imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

    