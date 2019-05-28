import dlib
import numpy as np
from skimage import io



predictor_path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

img = io.imread("sample.jpg")

dets = detector(img)

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
f=open("landmarkpointsave.txt","w+")
f.write(str(vec))
f.close()

print("success")


