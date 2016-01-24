import pyscreenshot as ImageGrab
import cv2
import sys
import numpy as np
import os
from PIL import Image
import re
from time import sleep
import subprocess
#Image.open('prev.gif').convert('RGB').save('prev.jpg')
cascPath = 'haarcascade_frontalface_alt_tree.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
recognizer = cv2.createFisherFaceRecognizer(threshold=600.0)
Images = []
Labels = []


def MonitorScreen():
    im=ImageGrab.grab()
    SingleImage = np.array(im) 
    # Convert RGB to BGR 
    SingleImage = SingleImage[:, :, ::-1].copy()
    faces = FindFaces(SingleImage)
    PredictArray = []
    nbr_predicted = 0
    for (x, y, w, h) in faces:
        ResizedImage = cv2.resize(SingleImage[y: y + h, x: x + w],(100,100))
        gray = cv2.cvtColor(ResizedImage, cv2.COLOR_BGR2GRAY)
        PredictArray.append(np.asarray(gray,dtype=np.uint8))
        nbr_predicted, conf = recognizer.predict(PredictArray[0])
        print nbr_predicted
    return nbr_predicted
        
def FindFaces(InputImage):
    ImageNumpy = np.array(InputImage,dtype=np.uint8)
    gray = cv2.cvtColor(ImageNumpy, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        #scaleFactor=1.1,
        #minNeighbors=5,
        #minSize=(0, 0),
        flags=cv2.cv.CV_HAAR_DO_CANNY_PRUNING#.cv.CV_HAAR_SCALE_IMAGE
    )
    return faces

def TrainOtherImages(FolderPath):
    for filename in os.listdir(FolderPath):
        SingleImage = cv2.imread(os.path.join(FolderPath,filename))
        #cv2.imshow("image",SingleImage)
        #cv2.waitKey(0)
        faces = FindFaces(SingleImage)
        for (x, y, w, h) in faces:
            ResizedImage = cv2.resize(SingleImage[y: y + h, x: x + w],(100,100))
            gray = cv2.cvtColor(ResizedImage, cv2.COLOR_BGR2GRAY)
            Images.append(np.asarray(gray,dtype=np.uint8))
            #cv2.imshow("Image",gray)
            #print gray.shape
            #cv2.waitKey(0)
            Labels.append(0)
            sleep(0.5)

def TrainCenaImages(FolderPath):
    for filename in os.listdir(FolderPath):
        SingleImage = cv2.imread(os.path.join(FolderPath,filename))
        #cv2.imshow("image",SingleImage)
        #cv2.waitKey(0)
        faces = FindFaces(SingleImage)
        for (x, y, w, h) in faces:
            ResizedImage = cv2.resize(SingleImage[y: y + h, x: x + w],(100,100))
            gray = cv2.cvtColor(ResizedImage, cv2.COLOR_BGR2GRAY)
            Images.append(np.asarray(gray,dtype=np.uint8))
            Labels.append(1)
           
def TestMethod(FolderPath):
     for filename in os.listdir(FolderPath):
        SingleImage = cv2.imread(os.path.join(FolderPath,filename))
        #cv2.imshow("image",SingleImage)
        #cv2.waitKey(0)
        faces = FindFaces(SingleImage)
        PredictArray = []
        for (x, y, w, h) in faces:
            ResizedImage = cv2.resize(SingleImage[y: y + h, x: x + w],(100,100))
            gray = cv2.cvtColor(ResizedImage, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Image",gray)
            print gray.shape
            cv2.waitKey(0)
            PredictArray.append(np.asarray(gray,dtype=np.uint8))
            nbr_predicted, conf = recognizer.predict(PredictArray[0])
            print filename
            print nbr_predicted

def TrainModel():
    TrainOtherImages("OtherFacesJpg")
    TrainCenaImages("JohnCenaTrain")
    recognizer.train(Images,np.asarray(Labels))
    recognizer.save('FisherFace')

def LoadModel():
    recognizer.load('FisherFace')
    #TestMethod("TestFace")
    #TestMethod("RandomFaces")

while 1:
    LoadModel()
    Result = MonitorScreen()
    if Result == 1:
        with open(os.devnull, 'wb') as devnull:
            subprocess.check_call(['mplayer', 'cena.wav'], stdout=devnull, stderr=subprocess.STDOUT)
        break



