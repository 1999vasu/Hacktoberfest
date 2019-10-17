import numpy as np
import cv2

#loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#turns your webcam on
cap = cv2.VideoCapture(0)

while True:
	#capturing frame by frame
	ret,img = cap.read()

	#transforming the captured frame to gray
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	#detects objects of diffrent sizes in the img and is returned as a list of rectangles
	faces = face_cascade.detectMultiScale(gray,1.3,5) #(image,scale_factor,minNeighbors)
	for x,y,w,h in faces:
		cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2) #drawing rectangle around the face
		roi_gray = gray[y:y+h, x:x+w] #roi for the face
		roi_color = img[y:y+h, x:x+w] #roi for the face (colored image)

		#detecting eyes
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for ex,ey,ew,eh in eyes:
			cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)

	cv2.imshow('img',img) #show the image
	k = cv2.waitKey(30) & 0xff #exit if Esc is pressed
	if k == 27:
		break

cap.release() #release the webcam
cv2.destroyAllWindows() #destroy the window



