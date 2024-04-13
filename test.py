import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
from collections import Counter 
import math
import time

cap = cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
classifier=Classifier("ML Project\Hand Sign Detection(ASL)\Model\keras_model.h5",
                                        "ML Project\Hand Sign Detection(ASL)\Model\labels.txt")

offset=20
imgSize=300

# folder="Data/Z"
counter=0

labels=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

l = []
res = ''
letters = cv2.imread('ML Project\Hand Sign Detection(ASL)\The_26_letters.jpg')
cv2.imshow("ML Project\Hand Sign Detection(ASL)\The_26_letters.jpg", letters)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if img is None:
        break
    imgResult = img.copy()
    imgOutput = img.copy()
    hands, img=detector.findHands(img)
    if hands:
        hand=hands[0]
        x, y, w, h=hand['bbox']
        imgWhite=np.ones((imgSize,imgSize,3), np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x + w+offset]

        imgCropShape=imgCrop.shape

        aspectRatio=h/w

        if aspectRatio > 1:
            k=imgSize/h
            wCal=math.ceil(k*w)
            if imgCrop.size==0:
                pass
                # print("Image is empty.")
            else:
                imgResize=cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape=imgResize.shape
                wGap=math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap: wCal + wGap]=imgResize
                prediction, index=classifier.getPrediction(imgWhite)
            # print(prediction,index)
            
        else:
            k=imgSize/w
            hCal=math.ceil(k*h)
            if imgCrop.size==0:
                pass
                # print("Image is empty.")
            else:
                imgResize=cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape=imgResize.shape
                hGap=math.ceil((imgSize-hCal)/2)
                imgWhite[hGap: hCal + hGap, :]=imgResize
                prediction, index=classifier.getPrediction(imgWhite)
        
        cv2.putText(imgOutput, labels[index], (x,y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.putText(imgOutput, res, (0,30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        l.append(labels[index])
        if len(l)==50:
            d = sorted(Counter(l).items())
            res += d[0][0]
            l.clear()
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w+offset, y + h+offset), (255, 0, 255), 4)
        # if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0 and imgWhite.shape[0] > 0 and imgWhite.shape[1] > 0:
        #     cv2.imshow("ImageCrop", imgCrop)
        #     cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()




