# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:42:07 2023

@author: MiskinLab
"""

# General Imports
import cv2
import numpy as np
import os
from PIL import Image
from scipy.spatial import distance as dist
from collections import OrderedDict
import glob
import math
import matplotlib.pyplot as plt
from GS_DraftWORKING_ import *
import time
# Camera Imports
from pypylon import pylon
from pypylon_opencv_viewer import BaslerOpenCVViewer
# SLM Imports
import detect_heds_module_path
from holoeye import slmdisplaysdk
from showSLMPreview import showSLMPreview

#log the mouse clicks into a point matrix
def mousePoints(event,x,y,flags,params):
    global counter
    global start

    # Left button click
    if event == cv2.EVENT_LBUTTONDOWN:
        point_matrix[counter] = x,y
        #counter = counter + 1
        #cv2.circle(preview, (x,y), 200, (255,0,0), 6)
        start = True

#takes in a pixel array representation of an image (such as the BlackBG image), does the GS, applies steering, and displays on SLM    
def dispSLM(dst):
    steeringX = 176
    steeringY = 318
    
    #sourceField = gerchbergSaxtonAlgorithm("SLMimage.png", targetPhase)
    origPhase = gerchbergSaxtonAlgorithm(dst, targetPhase)
    #origPhase = np.angle(sourceField)
    adjPhase = np.zeros((1080,1920))
    xshift,yshift = np.mgrid[-1079/2:1081/2, -1919/2:1921/2]
    xshift = (np.pi/180.0)*xshift*steeringX
    yshift = (np.pi/180.0)*yshift*steeringY
    adjPhase = 1*(origPhase + (xshift+yshift))%(2*np.pi)
    
    error = slm.showPhasevalues(adjPhase)
    assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

#initialize SLM
slm = slmdisplaysdk.SLMInstance()
if not slm.requiresVersion(3): # checking library version
     exit(1)
  
# DETECT SLM AND OPEN WINDOW
error = slm.open()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

########################### Initialising Camera ###############################           
# CREATING AND OPENING CAMERA OBJECT
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
viewer = BaslerOpenCVViewer(camera)

# SETTING PARAMETERS
camera.MaxNumBuffer = 5
targetPhase = 2 * np.pi * np.random.rand(1080, 1920) - np.pi # Phase guess

#open a CV2 preview window called "main". If this window is clicked, call the "mousePoints" fxn
cv2.namedWindow("main", cv2.WINDOW_NORMAL)
cv2.setMouseCallback('main', mousePoints)

# make a master image
#blackBG = np.zeros(shape=[3032, 5320], dtype=np.uint8)

#keep track of where we clicked
point_matrix = np.zeros((1,2),int)

M = np.array([[ 1.58950531e-01, -3.11168052e-04,  5.39237462e+02],
        [ 4.57916442e-03, -1.36700095e-01,  7.35889447e+02]])
start = False
counter = 0
cnt = 1

blackBG = np.zeros(shape=[3032, 5320], dtype=np.uint8)

t0 = time.time()

showSLMPreview(slm, scale=1.0)
while 1:
    im = viewer.get_image()
    
    if start:
        blackBG = np.zeros(shape=[3032, 5320], dtype=np.uint8)
        start = False
    a = point_matrix[0]
    cv2.rectangle(blackBG, ((a[0]-1),(a[1]-1)), ((a[0]+1),(a[1]+1)), (255,255,255), -1)
 #   cv2.circle(blackBG, (a[0],a[1]),3, (127,127,127), -1)


    #need add a pop after drawing the circle 
    #for a in point_matrix:
    # if cnt == 3:
    #     print("here")
    #     a = point_matrix[0]
    #     cv2.rectangle(blackBG, ((a[0]-50),(a[1]-50)), ((a[0]+50),(a[1]+50)), (255,255,255), -1)
    #     cnt = 1
    # elif cnt == 1:
    #     a = point_matrix[0]
    #     cv2.rectangle(blackBG, ((a[0]-50),(a[1]-50)), ((a[0]+50),(a[1]+50)), (127,127,127), -1)
    # elif cnt == 2:
    #     a = point_matrix[1]
    #     cv2.rectangle(blackBG, ((a[0]-50),(a[1]-50)), ((a[0]+50),(a[1]+50)), (127,127,127), -1)

    dst = cv2.warpAffine(blackBG,M,(1920,1080))

    dispSLM(dst)
    
    cv2.imshow('main',im)
    cv2.waitKey(1)

   # Image.fromarray(im).save(r"E:\Legged Microbots\Gaitbot_FaceUp_20xZoom_PointandClick_DoubleActuation2\n" + str(cnt) + ".tif")
    cnt = cnt + 1



