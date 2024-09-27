# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:26:18 2023

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
from scipy.spatial import KDTree
import time 
# Camera Imports
from pypylon import pylon
from pypylon_opencv_viewer import BaslerOpenCVViewer
# SLM Imports
import detect_heds_module_path
from holoeye import slmdisplaysdk
from showSLMPreview import showSLMPreview
# Tracker Imports
import norfair
from norfair import Detection, Tracker, Video, draw_tracked_objects
 
class Robot:  
    # Instance attribute
    def __init__(self, ID, pos):
        self.ID = ID
        self.pos = pos
    
    def getID(self):
        return self.ID
   
    def getPos(self):
        return self.pos

    def updatePos(self, newPos):
        self.pos = newPos
        
class Piloted(Robot):
    # Instance attributes
    engineProportion = (.50, .50)
    prevHeading = 0

    def __init__(self, ID, pos, waypoint):
        super(Piloted, self).__init__(ID, pos)
        self.heading = 0
        self.waypoint = waypoint
        # these arrays are going to log the data for plotting later
        self.engine1Array = []
        self.engine2Array = []
        self.angleArray = []
        self.weightArray = []
        self.positionArray = []
        self.velocityArray = []
        self.pXArray = []
        self.pYArray = []
        self.angleAdj = 0
        self.unadjAngles = []
        self.widthheight = []
        self.curvatureArray = []
        self.contour = []

    def updateWP(self, wp):
        self.waypoint = wp
        
    def updateAngleAdj(self, adj):
        self.angleAdj = adj
        
    def updateHeading(self, newHeading):
        self.heading = newHeading

    def addEngine1(self, value):
        self.engine1Array.append(value)

    def addEngine2(self, value):
        self.engine2Array.append(value)

    def addAngle(self, value):
        self.angleArray.append(value)

    def addWeight(self, value):
        self.weightArray.append(value)

    def getWaypoint(self):
        return self.waypoint

    def getEngineProportion(self):
        return self.engineProportion

    def getPrevHeading(self):
        return self.prevHeading
    
    def getAngleAdj(self):
        return self.angleAdj
    
def gerchbergSaxtonAlgorithm(old_im, targetPhase):
    w = 1920
    h = 1080
    new_size = (w, h)
    new_im = old_im
    img = np.array(new_im)

    targetAmplitude = np.sqrt(img)

    sourceAmplitude = np.ones((h, w)) 
    sourcePhase = np.ones((h, w))  
    sourceField = np.fft.fft2(targetAmplitude * np.exp(targetPhase * 1j)) #fft2 is the 2D fourier
    sourceField = np.fft.fftshift(sourceField)

    sourcePhase = np.angle(sourceField)

    return sourcePhase

#find the contours on a processed image (i.e. from robContours)
def contours(imgToCheckInit):
    cnts = cv2.findContours(imgToCheckInit, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    if (len(cnts) == 2):
        cnts = cnts[0]
    else:
        cnts = cnts[1]

    # only keep smaller contours
    cnts_append = []
    for c in cnts:
        if ((cv2.contourArea(c) > 10000)): #was 100, changed to 200 during troubleshooting with dummy image
            cnts_append.append(c)
    testArray = []    
    for i in cnts_append:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            testArray.append(np.asarray((cx,cy)))
    return cnts_append, testArray

#processes the images for ENTIRE DEVICE DETECTION. Assumes auto-exposure at min. top down illumination.
def robContours(imgToCheck):
    imgToCheck = cv2.cvtColor(imgToCheck, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    imgToCheck = cv2.adaptiveThreshold(imgToCheck, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10) #21
    imgToCheck = cv2.dilate(imgToCheck, kernel, iterations=2)
    imgToCheck = cv2.erode(imgToCheck,kernel,iterations = 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    imgToCheck = cv2.morphologyEx(imgToCheck, cv2.MORPH_CLOSE, kernel,
                                  iterations=10)
   
    cnts_append, robArray = contours(imgToCheck)
   
    return cnts_append, robArray

def engineLocs(c):
    minBound, (width,height), AoR = cv2.minAreaRect(c)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    points = np.int0(box)
    
    if (math.dist(points[0],points[1])) > (math.dist(points[0],points[-1])):
        #find the points on the top face corresponding to 1/3 and 2/3 dist.
        floated = ((2/3)*points[0][0] + (1/3)*points[-1][0])
        one_thirdX = round(((2/3)*points[0][0] + (1/3)*points[-1][0]))
        one_thirdY = round(((2/3)*points[0][1] + (1/3)*points[-1][1]))
        two_thirdsX = round(((1/3)*points[0][0] + (2/3)*points[-1][0]))
        two_thirdsY = round(((1/3)*points[0][1] + (2/3)*points[-1][1]))
        
        #get the corresponding points on the opposite face of the rectangle
        one_thirdX_bot = round(((2/3)*points[1][0] + (1/3)*points[2][0]))
        one_thirdY_bot = round(((2/3)*points[1][1] + (1/3)*points[2][1]))
        two_thirdsX_bot = round(((1/3)*points[1][0] + (2/3)*points[2][0]))
        two_thirdsY_bot = round(((1/3)*points[1][1] + (2/3)*points[2][1]))
       
        #save the engine points
        engine1 = [(points[0][0],points[0][1]),(points[1][0],points[1][1]),(one_thirdX_bot,one_thirdY_bot),(one_thirdX,one_thirdY)]
        engine1cnt = np.array(engine1).reshape((-1,1,2)).astype(np.int32)
        engine1_centerofvector = [((points[0][0]+points[1][0])/2,(points[0][1]+points[1][1])/2)]
       
        engine2 = [(points[-1][0],points[-1][1]),(two_thirdsX,two_thirdsY),(two_thirdsX_bot,two_thirdsY_bot),(points[2][0],points[2][1])]
        engine2cnt = np.array(engine2).reshape((-1,1,2)).astype(np.int32)
        engine2_centerofvector = [((points[-1][0]+points[2][0])/2,(points[-1][1]+points[2][1])/2)]
    
        # Difference in x coordinates
        dx = points[0][0] - points[1][0]

        # Difference in y coordinates
        dy = points[0][1] - points[1][1]

        # Angle between p1 and p2 in deg
        heading = np.rad2deg(math.atan2(dy, dx))
    else:
        floated = ((2/3)*points[0][0] + (1/3)*points[1][0])
        one_thirdX = round(((2/3)*points[0][0] + (1/3)*points[1][0]))
        one_thirdY = round(((2/3)*points[0][1] + (1/3)*points[1][1]))
        two_thirdsX = round(((1/3)*points[0][0] + (2/3)*points[1][0]))
        two_thirdsY = round(((1/3)*points[0][1] + (2/3)*points[1][1]))
       
        #get the corresponding points on the opposite face of the rectangle
        one_thirdX_bot = round(((2/3)*points[2][0] + (1/3)*points[-1][0]))
        one_thirdY_bot = round(((2/3)*points[2][1] + (1/3)*points[-1][1]))
        two_thirdsX_bot = round(((1/3)*points[2][0] + (2/3)*points[-1][0]))
        two_thirdsY_bot = round(((1/3)*points[2][1] + (2/3)*points[-1][1]))
       
        #save the engine points
        engine1 = [(points[0][0],points[0][1]),(points[-1][0],points[-1][1]),(two_thirdsX_bot,two_thirdsY_bot),(one_thirdX,one_thirdY)]
        engine1cnt = np.array(engine1).reshape((-1,1,2)).astype(np.int32)
        engine1_centerofvector = [((points[0][0]+points[-1][0])/2,(points[0][1]+points[-1][1])/2)]
       
        engine2 = [(points[1][0],points[1][1]),(points[2][0],points[2][1]),(one_thirdX_bot,one_thirdY_bot),(two_thirdsX,two_thirdsY)]
        engine2cnt = np.array(engine2).reshape((-1,1,2)).astype(np.int32)
        engine2_centerofvector = [((points[1][0]+points[2][0])/2,(points[1][1]+points[2][1])/2)]
       
        dx = points[0][0] - points[-1][0]

        # Difference in y coordinates
        dy = points[0][1] - points[-1][1]
            
        # Angle between p1 and p2 in deg
        heading = np.rad2deg(math.atan2(dy, dx))
    
    return engine1, engine1cnt, engine1_centerofvector,engine2, engine2cnt, engine2_centerofvector, heading, dx, dy

def dispSLM(dst):
    #steering values from calibration
    steeringX = 187
    steeringY = 302 
    
    origPhase = gerchbergSaxtonAlgorithm(dst, targetPhase)

    adjPhase = np.zeros((1080,1920))
    xshift,yshift = np.mgrid[-1079/2:1081/2, -1919/2:1921/2]
    xshift = (np.pi/180.0)*xshift*steeringX
    yshift = (np.pi/180.0)*yshift*steeringY
    adjPhase = 1*(origPhase + (xshift+yshift))%(2*np.pi)

    error = slm.showPhasevalues(adjPhase)
    assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)


##############################################################################
# MAIN CODE #
##############################################################################

#initialize camera object
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
viewer = BaslerOpenCVViewer(camera)
targetPhase = 2 * np.pi * np.random.rand(1080, 1920) - np.pi # Phase guess

#initialize SLM instance
slm = slmdisplaysdk.SLMInstance()
if not slm.requiresVersion(3): # checking library version
     exit(1)
  
#detect SLM, throw an error if can't
error = slm.open()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

#initialize the tracker
tracker = Tracker(distance_function="euclidean", distance_threshold=500)

#create a viewing window
cnt = 1
cv2.namedWindow("main", cv2.WINDOW_NORMAL)
showSLMPreview(slm, scale=1.0)

#transformation matrix M from calibration.py. Last updated 05/05/2023
M = np.array([[ 1.70386410e-01,  2.50124660e-03,  4.90698632e+02],
        [ 2.10809673e-03, -1.42391427e-01,  7.68385307e+02]])

#initialize a hash table that will have (Key:Value as ID:RobObject)
hash_table = {}
while 1:
    #grab a frame:
    im = viewer.get_image()
    t0 = time.time()
    
    #detect the robots in the frame
    cnts, detections = robContours(im)
    
    #send these detections to Norfair for persistent tracking, velocity, etc.
    norfair_detections = [Detection(points) for points in detections]
    tracked_objects = tracker.update(detections=norfair_detections)

    #set static waypoints here (ex: fig8)
    waypoints = [[2225,1000],[3350,1500],[3350,1000],[2225,1500],[2225,1000],[2225,1000],[3350,1500],[3350,1000],[2225,1500],[2225,1000]]
    
    #use this for initializing robots
    prevPos = [0,0]

    for a in tracked_objects:
        ID_key = a.id
        position = a.estimate
        # if this is a new ID, we don't have this robot in the list need to add it
        if ID_key not in hash_table:
            waypoint = (0, 0)
            prevPos = position
            # initialize the robot object with ID, position, and waypoint
            hash_table[ID_key] = Piloted(ID_key, position, waypoint)

    #now, we loop over each object in the hashtable and add the computed laser spots to a "master" image.
    #first, create a monochrome master image
    blackBG = np.zeros(shape=[3032, 5320], dtype=np.uint8)

    #loop over each object in the hash
    for key in hash_table:
        #get the waypoint and pos for the first robot 
        wp = hash_table[key].getWaypoint()
        position = hash_table[key].getPos()
        
        #check which detected contour contains the CoM of this robot. It seems like there should be a better way to do
        #this than looping through each contour every time, so this might need to be changed if it's slow.
        assoc_contour = []
        #loop through the contours and see where the point sits
        for i in cnts:
            if (cv2.pointPolygonTest(i,(position[0][0],position[0][1]),False)) > 0:
                assoc_contour = i
                print("failed to match contours")
                break
            
        #if the CoM wasn't associated with any contour, start over-- something probably went weird.
        if len(assoc_contour) == 0:
            continue

        #link this contour to hash key
        hash_table[key].contour.append(assoc_contour)

        #generate the minimum bounding rectangle for this contour and get the center
        minBound, (width,height), AoR = cv2.minAreaRect(assoc_contour)
        centerBounded = (int(minBound[0][0]),int(minBound[0][1]))
        hash_table[key].updatePos([centerBounded])
        
        #pass the contour to engineLocs function to find engine locations
        e1, engine1cnt, e1Center, e2, engine2cnt, e2Center, heading, dx, dy = engineLocs(assoc_contour)
        
        #now, we decide whether to shoot both engines or just one.
        #grab the heading vector of the robot
        headingVector = [dx,dy]
        
        #next, get the desired heading vector (center of robot to waypoint)                
        centerToWaypoint = [(round(centerBounded[0] - wp[0]),round(centerBounded[1] - wp[1]))]
    
        #make unit vectors
        unit_vector_1 = headingVector / np.linalg.norm(headingVector)
        unit_vector_2 = centerToWaypoint / np.linalg.norm(centerToWaypoint)
        
        #now, we want to find two angles:
        # theta1 is the angle from the desired heading vector to the x axis
        # theta2 is the angle from the actual heading vector to the x axis
        # theta3 is the relevant value and is found by subtracting theta1-theta2
        theta1 = math.atan2(unit_vector_1[1], unit_vector_1[0])
        theta2 = math.atan2(unit_vector_2[0][1], unit_vector_2[0][0])
        theta3 = math.degrees(theta2-theta1)
        angle = theta3
        
        #we want to define the robot's heading from -pi/pi
        if angle < -180:
            angle = angle + 180
        elif angle > 180:
            angle = angle - 180

        #get the previous heading angle and switch it out with the new one.
        prevAngle = hash_table[key].getPrevHeading()
        setattr(hash_table[key],'prevHeading',angle)
        
        #as 127 is max intensity, we can set the angle @ which max. turning occurs.
        #for example, here any angles greater than 60 deg. get maximum weight
        weight = abs(angle) + 67
        if weight > 127:
            weight = 127
        
        #we're close enough to the waypoint. don't do anything.
        if math.dist(wp, centerBounded) < 350:
            continue
        
        #if the angle is small and prev angle is less than 7.5, shoot both engines
        if (abs(angle) < 7.5) and (prevAngle < 7.5):
            cv2.drawContours(blackBG, [engine1cnt], -1,(weight,weight,weight),-1)
            cv2.drawContours(blackBG, [engine2cnt], -1,(weight,weight,weight),-1)
            cv2.drawContours(im,[engine1cnt],-1,(0,0,255),3)
            cv2.drawContours(im,[engine2cnt],-1,(0,0,255),3)
            cv2.line(im, wp, centerBounded, (0,255,0), 6)
            cv2.line(im, wp, centerBounded, (0,255,0), 6)     
        
        #if the angle is not small, we need to shoot one of the engines with a power weighted by the heading error.
        #as 127 is the pixel value corresponding to maximum intensity, we can tweak the robot's behavior by changing the constant
        #in the weight equation above.
        else:
            #we choose which engine to shoot based on which one is farther away
            if math.dist(wp, e1Center) > math.dist(wp, e2Center):
                cv2.drawContours(blackBG,[engine1cnt],0,(weight,weight,weight),-1)
                cv2.drawContours(im,[engine1cnt],-1,(0,0,255),3)
                eP1 = weight
                eP2 = 0
                cv2.line(im, wp, centerBounded, (255,0,0), 6)
            else:
                cv2.drawContours(blackBG,[engine2cnt],0,(weight,weight,weight),-1)
                cv2.drawContours(im,[engine2cnt],-1,(0,0,255),3)
                eP2 = weight
                eP1 = 0
                cv2.line(im, wp, centerBounded, (255,0,0), 12)
        
        #now append all these values to the robot object's arrays so we have them to plot in the future
        hash_table[key].addEngine1(eP1)
        hash_table[key].addEngine2(eP2)
        hash_table[key].addAngle(angle)
        hash_table[key].addWeight(weight)
   
    #take the final image (blackBG) and transform it with the matrix M.
    dst = cv2.warpAffine(blackBG,M,(1920,1080))

    #display on SLM    
    dispSLM(dst)

    ## draw waypoints on the preview and show
    for w in waypoints:
        im = cv2.circle(im, w,200,(255,0,0),6)
    draw_tracked_objects(im, tracked_objects)
    cv2.imshow('main',im)
    cv2.waitKey(1)

    #save images
    #Image.fromarray(im).save(r"D:\swimming\program save\n" + str(cnt) + ".tif")
    #Image.fromarray(im2).save(r"D:\swimming\program save no w\n" + str(cnt) + ".tif")
    cnt += 1


    