# Face-Feature-Reshaping-for-Face-Beautification
Face feature such as eyebrow, eye, nose, lip, nose, jaw and chin are reshaped as per user requirement in real time.
The code is availble in python. To successfully run this system we required opencv with python.
the device front camera is used to stream input live video so we required to take permission of camera and make it on.
In first step, it takes input video stream then detect the face from the video frame.
Then Face Landmark is mark and save 68 landmark point of face.
The moving least square method is used to deform the face feature in that they take the input as video frame and landmark point and they change the position of landmark point so the face feature are reshaped and generate the output video stream.
The paper realted this implememntation and research is develop that title " A Survey on Face beautification Techniques without Cosmetic surgery"
you have to run shortcut.py file for full code exxcute : command : python shortcut.py


