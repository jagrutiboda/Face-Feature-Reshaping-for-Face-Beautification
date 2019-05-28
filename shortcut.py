import numpy as np
import cv2
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import os
import sys
import random
import matplotlib.pyplot as plt
import time

from img_utils import (mls_affine_deformation, mls_affine_deformation_inv,
                       mls_similarity_deformation, mls_similarity_deformation_inv,
                       mls_rigid_deformation, mls_rigid_deformation_inv)
					   
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
parameters=np.load('sliderpoint.npz')
a=parameters['arr_0']

cap = cv2.VideoCapture(0)
out=cv2.VideoWriter('output.avi',-1, 20.0,(640,480))

count = 0 
countt =0
p=[[0]*2 for x in range(68)]
q=[[0]*2 for x in range(68)]
list_of_lists = []
with open('sliderpoint.txt') as f:
    for line in f:
        #inner_list = [elt.strip() for elt in line.split(' ')]
        # in alternative, if you need to use the file content as numbers
        inner_list = [int(elt.strip()) for elt in line.split(' ')]
        list_of_lists.append(inner_list)

#a[37][0]=list_of_lists[0][0]
#a[37][1]=list_of_lists[0][1]
#a[40][0]=list_of_lists[1][0]
#a[40][1]=list_of_lists[1][1]
		

while(True):
    # Capture frame-by-fram
    start=time.time()
    countt=0
    ret, image = cap.read()
    cv2.imwrite("framesave/frame%d.jpg" % count,image)
    count +=1
   

    # Our operations on the frame come here
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = plt.imread(os.path.join(sys.path[0], "testimage.jpg"))
             #code here use image
			 
    dets = detector(gray)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame',gray)
    rects = detector(gray, 0)
		# Saves the frames with frame-count 
    #cv2.imshow('frame',gray)
    for rect in rects:
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
        for (x, y) in shape:
            
           
            #cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            #cv2.imwrite(''+"frame%d.jpg" % count,image)
            
            #end_time = time.time() #record end time of program
            #fps = countt / float(end_time - start)
            
            for k, d in enumerate(dets):
                shape = predictor(gray, d)
                countt+=1
                vec = np.empty([68, 2], dtype = int)
                for b in range(68):
                    p[b][0] = shape.part(b).x
                    p[b][1] = shape.part(b).y
                    q[b][0] = shape.part(b).x+a[b][0]
                    q[b][1] = shape.part(b).y+a[b][1]
                p=np.asarray(p)
                q=np.asarray(q)
        #print("Framerate count..")
        
        #print("FPS: ", 1.0 / fps) # FPS = 1 / time to process loop
		
        #cv2.imwrite("%d", countt)
                      
        cv2.imwrite(''+"framesave/framer%d.jpg" % countt,image)
        #print(count)       
		
                    
                #print(p.shape)
                #print(q.shape)
        #filename = './Users\Sony\Desktop\MSR\face morphing code\video frame extract\output' + "frame%d.jpg";count+=1
        #cv2.imwrite(filename, image)
        #cv2.imwrite(''+"framer%d.jpg" % countt,image)
        #cv2.imshow('frame',image)
        #count += 1
    #transformed_image = mls_rigid_deformation_inv(image, p, q, alpha=1, density=0.7)
       # cv2.imwrite("framesave/frame%d.jpg" % count, image)
        lstart=time.time()
        print('Preprocessing Time : ',(lstart-start))
        astart=time.time()
        transformed_image = mls_rigid_deformation_inv(image, p, q, alpha=1, density=1)
        end=time.time()
    # Display the resulting frame
        print('Frame processing Time : ',(end-astart))
        cv2.imwrite("frameM%d.jpg" % count, transformed_image)
        out.write(transformed_image)
		
	
        cv2.imshow('frame',transformed_image)
        print("success")
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break
  


# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
