# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 11:12:57 2022

@author: MiskinLab
"""

import cv2
import os, time
import numpy as np
from PIL import Image
# Camera Imports
from pypylon import pylon
from pypylon_opencv_viewer import BaslerOpenCVViewer
# SLM Imports
import detect_heds_module_path
from holoeye import slmdisplaysdk
from showSLMPreview import showSLMPreview
from GS_DraftWORKING_cal import *
from feedback import *
import math
import cmath
import random
import matplotlib.pyplot as plt 

def genCoefficients(ptsArray,expPtsArray):
    from_pt = ptsArray
    to_pt =   expPtsArray

    # Fill the matrices
    A_data = []
    for pt in from_pt:
      A_data.append( [-pt[1], pt[0], 1, 0] )
      A_data.append( [ pt[0], pt[1], 0, 1] )

    b_data = []
    for pt in to_pt:
      b_data.append(pt[0])
      b_data.append(pt[1])

    # Solve
    A = np.matrix( A_data )
    b = np.matrix( b_data ).T
    c = np.linalg.lstsq(A, b)[0].T
    c = np.array(c)[0]

    print("Solved coefficients:")
    print(c)
      
    return c

def transformPoints(c,from_pt):
    to_pt = []
    for pt in from_pt:
      to_pt.append([(
        c[1]*pt[0] - c[0]*pt[1] + c[2],
        c[1]*pt[1] + c[0]*pt[0] + c[3] )])
      
    return to_pt

def imageInit(imgToCheck):
    
    # CONVERTING IMAGE TO BINARY
    imgToCheck = cv2.cvtColor(imgToCheck, cv2.COLOR_BGR2GRAY)
    #imgToCheck = cv2.blur(imgToCheck, (5, 5))
    kernel = np.ones((5, 5), np.uint8)
    imgToCheck = cv2.adaptiveThreshold(imgToCheck, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 10) #21
    #imgToCheck = cv2.dilate(imgToCheck, kernel, iterations=1)
    imgToCheck = cv2.dilate(imgToCheck, kernel, iterations=2)
    imgToCheck = cv2.erode(imgToCheck,kernel,iterations = 1)
    
    # MORPH OPEN
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    imgToCheck = cv2.morphologyEx(imgToCheck, cv2.MORPH_OPEN, kernel, 
                                  iterations=4)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    imgToCheck = cv2.morphologyEx(imgToCheck, cv2.MORPH_CLOSE, kernel, 
                                  iterations=10)
    
    #cv2.imshow("Camera Preview", imgToCheck)
    #cv2.waitKey(1)
    return imgToCheck

########
slm = slmdisplaysdk.SLMInstance()
if not slm.requiresVersion(3): # checking library version
     exit(1)
  
# DETECT SLM AND OPEN WINDOW
error = slm.open()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# OPEN SLM IN PREVIEW WINDOW IN NON-SCALED MODE

########################### Initialising Camera ###############################           
# CREATING AND OPENING CAMERA OBJECT
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
viewer = BaslerOpenCVViewer(camera)
# SETTING PARAMETERS
countOfImagesToGrab = 100
camera.MaxNumBuffer = 5
targetPhase = 2 * np.pi * np.random.rand(1080, 1920) #- np.pi # Phase guess

from showSLMPreview import showSLMPreview
showSLMPreview(slm, scale=1.0)

steeringX = 0
steeringY = 0
aInc = 1
aMax = 540
# Bringing number DOWN makes spot go RIGHT
ydeg = 325 #302 #258
blackBG = np.zeros(shape=[1080, 1920], dtype=np.uint8)
cv2.circle(blackBG, (960, 540), 15, (127, 127, 127), -1)
#blackBG = cv2.rotate(blackBG, cv2.cv2.ROTATE_90_CLOCKWISE)
#cv2.imwrite("SLMimage1.png", blackBG) # Image to be loaded onto SLM

cv2.imwrite("SLMimage.png", blackBG)  
sourceField = gerchbergSaxtonAlgorithm("SLMimage.png", targetPhase)
imgToCheck = viewer.get_image()
imgToCheckInit = cv2.cvtColor(imgToCheck,cv2.COLOR_BGR2GRAY)
while ydeg < aMax:
    ydeg += aInc
    # bringing number DOWN makes spot go UP
    xdeg = 174 #178
    while xdeg < aMax:
        imgToCheck = viewer.get_image()
        origPhase = np.angle(sourceField)
        #print(origPhase)
        adjPhase = np.zeros((1080,1920))
        #print(origPhase)
        xshift,yshift = np.mgrid[-1079/2:1081/2, -1919/2:1921/2]
        xshift = (np.pi/180.0)*xshift*xdeg #3.2
        yshift = (np.pi/180.0)*yshift*ydeg #.7
        adjPhase = 1*(origPhase + (xshift+yshift))%(2*np.pi)
        #print(adjPhase)
        #adjPhase = adjPhase*180/np.pi
        error = slm.showPhasevalues(adjPhase)
        assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
    
        imgToCheckInit = viewer.get_image()
        imgToCheckInit = cv2.cvtColor(imgToCheck,cv2.COLOR_BGR2GRAY)
        #imgToCheckInit = imageInit(imgToCheck)

        circles = cv2.HoughCircles(imgToCheckInit,cv2.HOUGH_GRADIENT,1,350,
                      param1=60,param2=40,minRadius=5,maxRadius=150)

        cv2.circle(imgToCheckInit, (2660,1516), 100, (127, 127, 127), 3)
        cv2.namedWindow("main", cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('main', 900, 900) 
        cv2.imshow('main',imgToCheckInit)
        cv2.waitKey(1)
        
        # Draw the circles
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
            # draw the outer circle
                cv2.circle(imgToCheckInit,(i[0],i[1]),i[2],(0,255,255),2)
            # draw the center of the circle
                cv2.circle(imgToCheckInit,(i[0],i[1]),2,(0,255,255),3)
                cv2.imshow('main',imgToCheckInit)
                cv2.waitKey(1)
                if (math.isclose(i[0],2660,abs_tol=50)) & (math.isclose(i[1],1516,abs_tol=50)):
                    steeringX = xdeg
                    steeringY = ydeg
                    xdeg = aMax
                    ydeg = aMax
                    print("", steeringX)
                    print("", steeringY)
        xdeg += aInc

        #Image.fromarray(imgToCheckInit).save(r"C:\Users\MiskinLab\Desktop\tst\pol" + str(xdeg) + ".tif")
         
#creating 3 points to det transformation matrix
#ptsArray = np.float32([[540,960],[random.randint(0, 1080), random.randint(0, 1920)],[random.randint(0, 1080), random.randint(0, 1920)]])
#ptsArray = np.float32([[673,669], [913,425],[1237,581]])
#ptsArray = np.float32([[1167,657],[961,381],[701,577]])
#ptsArray = np.float32([[961,381],[1220,581],[760,660]])
#ptsArray = np.float32([[759,659],[957,386],[1220,581]])
#ptsArray = np.float32([[957,386],[759,659],[1220,581]])
#ptsArray = np.float32([[961,381],[701,577],[1167,657]])

#triasc2

#ptsArray = np.float32([[941,681],[1165,489],[737,429]])
#ptsArray = np.float32([[1165,489],[765,443],[939,679]])
#ptsArray = np.float32([[607,641],[1241,575],[869,515],[941,691],[1177,441],[675,413],[1043,347],[1309,663]])
#ptsArray = np.float32([[873,513],[937,693],[1233,577],[609,637],[669,409],[1045,345],[1177,437],[1305,661]])
#ptsArray = np.float32([[941,691],[869,515],[609,639],[1179,441],[1235,577],[673,413],[1307,665],[1045,347]])
#ptsArray = np.float32([[869,515],[609,639],[941,691],[1235,575],[1179,441],[673,413],[1045,347],[1307,665]])
#ptsArray = np.float32([[1235,575],[1179,441],[1307,665],[1045,347],[869,515],[941,691],[609,639],[673,413]])
#ptsArray = np.float32([[709,581],[857,705],[961,385],[1217,577]])
ptsArray = np.float32([[741,601],[1187,667],[989,419]])

# blackBG = np.zeros(shape=[1080, 1920], dtype=np.uint8)
# #drawing the points onto a black background
# for x in ptsArray:
#     cv2.circle(blackBG, (int(x[0]), int(x[1])), 20, (127, 127, 127), -1)
# blackBG = cv2.rotate(blackBG, cv2.cv2.ROTATE_90_CLOCKWISE)
# blackBG = cv2.rotate(blackBG, cv2.cv2.ROTATE_90_CLOCKWISE)
#blackBG = cv2.flip(blackBG, 1)
#cv2.imwrite("SLMimage.png", cv2.resize(blackBG, (0,0), fx = 0.5, fy = 0.5)) # Image to be loaded onto SLM

#cv2.imwrite("SLMimage.png", blackBG) # Image to be loaded onto SLM

#GS the image and do the steering with values found above
sourceField = gerchbergSaxtonAlgorithm("triasc2-2.png", targetPhase)
origPhase = np.angle(sourceField)
adjPhase = np.zeros((1080,1920))
xshift,yshift = np.mgrid[-1079/2:1081/2, -1919/2:1921/2]
xshift = (np.pi/180.0)*xshift*steeringX
yshift = (np.pi/180.0)*yshift*steeringY
adjPhase = 1*(origPhase + (xshift+yshift))%(2*np.pi)
error = slm.showPhasevalues(adjPhase)
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
cv2.imshow('main',imgToCheckInit)
cv2.waitKey(10)
time.sleep(1)
#take microscope image (hopefully) with 3 dots in the FOV  
imgToCheck = viewer.get_image()
# kernel = np.ones((5, 5), np.uint8)
imgToCheckInit = cv2.cvtColor(imgToCheck,cv2.COLOR_BGR2GRAY)
# imgToCheckInit = cv2.adaptiveThreshold(imgToCheckInit, 255,
#                                cv2.ADAPTIVE_THRESH_MEAN_C, 
#                                cv2.THRESH_BINARY_INV, 21, 10) #21
# imgToCheckInit = cv2.dilate(imgToCheckInit, kernel, iterations=2)
# imgToCheckInit = cv2.erode(imgToCheckInit,kernel,iterations = 2)

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
# imgToCheckInit = cv2.morphologyEx(imgToCheckInit, cv2.MORPH_CLOSE, kernel, 
#                               iterations=10)
cv2.imshow('main',imgToCheckInit)
cv2.waitKey(10)
#initialize image and find/draw contours
circles1 = cv2.HoughCircles(imgToCheckInit,cv2.HOUGH_GRADIENT,1,550,
             param1=60,param2=20,minRadius=20,maxRadius=350)
    
#populate an array with experimental CoM values for the dots projected above
expPtsArray = []
if circles1 is not None:
    print(circles1)
    for i in circles1[0,:]:
    # draw the outer circle
        cv2.circle(imgToCheckInit,(int(i[0]),int(i[1])),50,(127,127,127),-1)
    # draw the center of the circle
        cv2.circle(imgToCheckInit,(int(i[0]),int(i[1])),2,(127,127,127),-1)
        cv2.imshow('main',imgToCheckInit)
        cv2.waitKey(1000)
        expPtsArray.append([i[0],i[1]])

cv2.imshow('main',imgToCheckInit)
cv2.waitKey(1000)
#now we have the projected pt locations and exp pt locations-- solve for matrix
#M = cv2.getAffineTransform(ptsArray,np.float32(expPtsArray))
M = cv2.estimateAffine2D(np.float32(expPtsArray),ptsArray)
#m, _ = cv2.estimateAffinePartial2D(ptsArray, np.float32(expPtsArray))


#########################
# Input points

#########################
# preview = viewer.get_image()
# preview = imageInit(preview)

# #imgToCheck = cv2.imread("moreshapes.png",0) 
# #rows, cols = imgToCheck.shape[:2]
# dst = cv2.warpAffine(preview, M,(1920,1080))
# #dst = cv2.flip(dst,1)
# #cv2.imwrite("SLMimage.png",dst)
# cv2.imwrite("SLMimage.png", dst)

# #GS the image and do the steering with values found above
# sourceField = gerchbergSaxtonAlgorithm("SLMimage.png", targetPhase)
# origPhase = np.angle(sourceField)
# adjPhase = np.zeros((1080,1920))
# xshift,yshift = np.mgrid[-1079/2:1081/2, -1919/2:1921/2]
# xshift = (np.pi/180.0)*xshift*steeringX
# yshift = (np.pi/180.0)*yshift*steeringY
# adjPhase = -1*(origPhase + (xshift+yshift))%(2*np.pi)

# error = slm.showPhasevalues(adjPhase)
# assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

preview = viewer.get_image()
cv2.imshow('main', preview)
cv2.waitKey(1)


