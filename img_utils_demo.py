# Program To Read video 
# and Extract Frames 
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import json

# Function to extract frames 
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
def FrameCapture(path): 
	
	# Path to video file 
	vidObj = cv2.VideoCapture(0) 

	# Used as counter variable 
	count = 0

	# checks whether frames were extracted 
	success = 1

	'''img = io.imread("testimage.jpg")



#output face landmark points inside retangle
#shape is points datatype
#http://dlib.net/python/#dlib.point
for k, d in enumerate(dets):
    shape = predictor(img, d)

vec = np.empty([68, 2], dtype = int)
for b in range(68):
    vec[b][0] = shape.part(b).x
    vec[b][1] = shape.part(b).y
	
print(vec)
f=open("landmarks.txt","w+")
f.write(str(vec))'''

	while success: 

		# vidObj object calls read 
		# function extract frames 
		success, image = vidObj.read() 
		image = imutils.resize(image, width=900)
		dets = detector(image)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 0)
		# Saves the frames with frame-count 
		for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
			for (x, y) in shape:
				cv2.circle(image, (x, y), 3, (0, 255, 255), -1)
				for k, d in enumerate(dets):
					shape = predictor(image, d)
				vec = np.empty([68, 2], dtype = int)
			f=open("points.txt","w+")
			for b in range(68):
				vec[b][0] = shape.part(b).x
				vec[b][1] = shape.part(b).y
				f.write(str(vec[b][0])+" "+str(vec[b][1])+"\n")
	
			print(vec)

			f=open("landmarks.txt","w+")
			f.write(str(vec))
			#with open("output.txt", "w") as outfile:
			#	json.dump(vec.tolist(), outfile)
		cv2.imshow("Frame", image)
		cv2.imwrite("frame%d.jpg" % count, image) 

		count += 1
	dets = detector(image)
	
	
# Driver Code 
if __name__ == '__main__': 

	# Calling the function 
	FrameCapture("C:\\Users\\Admin\\PycharmProjects\\project_1\\openCV.mp4") 

	