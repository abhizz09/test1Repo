# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:35:10 2019

@author: Shikha
"""
import Templatematching
import math
import cv2
from math import sqrt

import numpy as np
import time
from collections import defaultdict
#############################################################################
def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented
######################################################################
    #to calculate the intersection of lines
def intersectLines( pt1, pt2, ptA, ptB ): 
    
    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1;   x2, y2 = pt2
    dx1 = x2 - x1;  dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA;   xB, yB = ptB;
    dx = xB - x;  dy = yB - y;

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where DET = (-dx1 * dy + dy1 * dx)
    #
    # if DET is too small, they're parallel
    #
    DET = (-dx1 * dy + dy1 * dx)
    if(DET != 0):
        
        

#    if math.fabs(DET) < DET_TOLERANCE: return (0,0,0,0,0)

    # now, the determinant should be OK
        DETinv = 1.0/DET
    
        # find the scalar amount along the "self" segment
        r = DETinv * (-dy  * (x-x1) +  dx * (y-y1))
    
        # find the scalar amount along the input line
        s = DETinv * (-dy1 * (x-x1) + dx1 * (y-y1))
        xi=(x1 + r*dx1 + x + s*dx)/2.0
    #    print(xi)
        # return the average of the two descriptions
        xi = int((x1 + r*dx1 + x + s*dx)/2.0)
    #    print(xi)
    
        yi = int((y1 + r*dy1 + y + s*dy)/2.0)
        return [[xi, yi]]

###################################################################

def testIntersection( line1, line2 ):
    for x1,y1,x2,y2 in line1:
        pt1=(x1,y1)
        pt2=(x2,y2)
        
    for x1,y1,x2,y2 in line2:
        ptA=(x1,y1)
        ptB=(x2,y2)

    result = intersectLines( pt1, pt2, ptA, ptB )
    return result 
################################################################### 
    #to calculate the slope
def slope(x1, y1, x2, y2): 
    return (float)(y2-y1)/(x2-x1)
###################################################################
def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                
                    intersections.append(testIntersection(line1, line2)) 

    return intersections
###################################################################
def calculateDistance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist 
#####################################################################
def intersect_distance(newarray):
    dist = lambda p1, p2: sqrt(((p1-p2)**2).sum())
    dm = np.asarray([[dist(p1, p2) for p2 in newarray] for p1 in newarray])
    return dm
######################################################################
    #to calculate  parallel lines
def calculate_parallel(line1, line2):
    x1=line1[0]
    y1=line1[1]
    x2=line1[2]
    y2=line1[3]
    if((x1 != x2)):
        m=slope(x1,y1,x2,y2)
        m=(math.floor(m*100)/100)
        b1=y1-(m*x1);
        b1=math.floor(b1*100)/100
    x1=line2[0]
    y1=line2[1]
    x2=line2[2]
    y2=line2[3]
    if((x1 != x2)):
        m2=slope(x1,y1,x2,y2)
        m2=(math.floor(m2*100)/100)
        b2=y1-(m*x1);
        b2=math.floor(b2*100)/100
    if((abs(m-m2))<0.33):
        
        distance=(dist(m, b1, b2))
        maxdist.append(distance)
#        print("line1=",line1,"line2=",line2,"m1=",m,"m2=",m2,"b1=",b1,"b2=",b2,"distane=",distance,"                                                     ")
    return maxdist
############################################################################  
    
#distance between parallellines
def distance_parallel(parallellines):
    """Finds the intersections between groups of lines."""
#    print(parallellines)
    for i, group in enumerate(parallellines[:-1]):
        for next_group in parallellines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    distance=(calculate_parallel(line1, line2)) 
    
        
    return distance
##########################################################################
img = cv2.imread("EXTRACTION.jpg")
#cv2.namedWindow("INPUT", cv2.WINDOW_NORMAL)
#cv2.imshow("INPUT",img)
#cv2.waitKey(0)
maxdist=[]
img1=img.copy() 
height = img.shape[0]
width = img.shape[1]
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.namedWindow("GRAY", cv2.WINDOW_NORMAL)
#cv2.imshow("GRAY",gray)
#cv2.waitKey(0)
th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,99,14)
#cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
#cv2.imshow("thresh", th3)
#cv2.waitKey(0)
retval, labels = cv2.connectedComponents(th3)
#print(labels)
##################################################
ts = time.time()
num = labels.max()
#print("num=",num)
N = 50
for i in range(1, num+1):
    pts =  np.where(labels == i)
    if len(pts[0]) < N:
        labels[pts] = 0

#print("Time passed: {:.3f} ms".format(1000*(time.time()-ts)))
# Time passed: 4.607 ms

##################################################

# Map component labels to hue val
label_hue = np.uint8(179*labels/np.max(labels))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
#cv2.namedWindow('labeled.png', cv2.WINDOW_NORMAL)
#cv2.imshow('labeled.png', labeled_img)
#cv2.waitKey()
# cvt to BGR for display
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

# set bg label to black
labeled_img[label_hue==0] = 0
#cv2.namedWindow('labeled.png', cv2.WINDOW_NORMAL)
#cv2.imshow('labeled.png', labeled_img)
#cv2.waitKey()

edges = cv2.Canny(labeled_img,50,255)
#cv2.namedWindow("EDGES", cv2.WINDOW_NORMAL)
#cv2.imshow("EDGES", edges)
#cv2.waitKey(0)
kernel = np.ones((2,2), np.uint8) 
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel,iterations=2)
#cv2.namedWindow("CLOSING", cv2.WINDOW_NORMAL)
#cv2.imshow("CLOSING",closing)
#cv2.waitKey(0)
lines = cv2.HoughLinesP(closing, rho=0.53, theta=0.7* np.pi / 180, 
threshold=105, minLineLength=120, maxLineGap=140)
font = cv2.FONT_HERSHEY_SIMPLEX
print("\nNUMBER OF LINES:=",len(lines))
count1=0
#mindist=0
#print("\nTotal lines:",lines)
slopelist=[]
for i in lines:
    for x1, y1, x2, y2 in i:
        if((x1 != x2)):
            
            m=slope(x1,y1,x2,y2)
            b=y1-(m*x1);
#            print("m=",m)
            m=math.floor(m*100)/100
#            print("m=",m)

            b=math.floor(b*100)/100
            slopelist.append(m)
#            print(x1, y1, x2, y2)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.namedWindow("houghlines5.jpg", cv2.WINDOW_NORMAL)
cv2.imshow('houghlines5.jpg', img)
cv2.waitKey(0)
############################################################
# to calculate intersection point and ploting
       
segmented = segment_by_angle_kmeans(lines)
#print("segmented=",segmented)
intersections = segmented_intersections(segmented)    
distance = [] 
new_intersect=[]
#print(type(intersections))
#print(intersections)
for i in intersections:
    if(i!= None):
#        print(i)
        if i[0][0]>0 and  i[0][1]>0 and i[0][0]<width and i[0][1]<height :
            new_intersect.append(i)
            
print("\nIntersection point coordiantes:\n",new_intersect)
######################################################################
#After getting intersection point ...calculating length and width
count1=0
sortedpoint = sorted(new_intersect, key = lambda x: x[0][1])
#print(sortedpoint)
x1=0
y1=0
for i in sortedpoint:
    count1=count1+1
    if(count1==1):
        x1=i[0][0]
        y1=i[0][1]
        cv2.circle(img,(x1,y1), 1, (0,0,255), thickness=5, lineType=8, shift=0)
    else:
        x2=i[0][0]
        y2=i[0][1]
        cv2.circle(img,(x2,y2), 1, (0,0,255), thickness=5, lineType=8, shift=0)
        caldis=calculateDistance(x1,y1,x2,y2)
        x1=x2
        y1=y2
        if(count1==2):
            print("\n\nLength of the slab=",caldis)
        elif(count1==3):
            print("\n\nWidth of the slab=",caldis)
      
cv2.namedWindow("point", cv2.WINDOW_NORMAL)
cv2.imshow("point", img) 
cv2.waitKey(0)
########################################################
#to calculate the lines which are parallel


list1=[]  
def dist(m, b1, b2):
    d = abs(b2 - b1) / sqrt(((m * m) + 1)); 
    return d; 

#print(slopelist)            
for i in range(len(slopelist)):
    for j in range(i+1,len(slopelist)):
        if abs(slopelist[i]-slopelist[j])<0.03 :
            list1.append(slopelist[i])
            list1.append(slopelist[j])
#print(list1)                       
new_list=set(list1)
#print(new_list)
parallellines=[]
for i in  (lines):
    for x1, y1, x2, y2 in i:
        if((x1 != x2)):
            m1=slope(x1,y1,x2,y2)
            m1=math.floor(m1*100)/100
            for m2 in new_list:
#                print(m1,m2)
                if m1 == m2:
                    parallellines.append(i)
                    break
parallellines = np.asarray(parallellines)                 
#print(("parallel lines=",parallellines))  

  
#################################################



parallel_linesdist=distance_parallel(parallellines)
#print("\n\ndistance between parallel lines:\n",parallel_linesdist)
print("\n\nthikness of slab=",max(parallel_linesdist))
#print("LINES",lines)
#print("\n\n\n\n\nseg",segmented)
k=input("\nPress any key to exit")