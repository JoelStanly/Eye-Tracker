import cv2
import numpy as np
import dlib


cap= cv2.VideoCapture(0)
detector =dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _,frame=cap.read()
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=detector(gray)
    for face in faces:
        landmarks=predictor(gray,face)
        left_point=(landmarks.part(36).x,landmarks.part(36).y)
        right_point=(landmarks.part(39).x,landmarks.part(36).y)
        
        hor_line=cv2.line(frame,left_point,right_point,(0,255,0),2)

    cv2.imshow("Frame",frame)

    key=cv2.waitKey(1)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()