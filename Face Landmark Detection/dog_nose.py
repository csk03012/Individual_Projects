import cv2
import dlib
import numpy as np
from math import hypot



print(dlib.__version__)
print(cv2.__version__)



cap = cv2.VideoCapture(0)
nose_image = cv2.imread("dog_nose.png")


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray_frame)

#faces_cordinates
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3) 

#nose_cordinates
        landmarks = predictor(gray_frame,face) 
        top_nose = (landmarks.part(29).x,landmarks.part(29).y)
        centre = (landmarks.part(30).x,landmarks.part(30).y)
        right_nose = (landmarks.part(31).x,landmarks.part(31).y)
        left_nose = (landmarks.part(35).x,landmarks.part(35).y)

#nose_dimension
        nose_width = int(hypot(left_nose[0]-right_nose[0],left_nose[1]-right_nose[1]))
        nose_height = int(nose_width * 1.5)

#nose_position
        top_left = (int(centre[0]-nose_width/2),int(centre[1]-nose_height/2))
        bottom_right = (int(centre[0]+nose_width/2),int(centre[1]+nose_height/2))


        # cv2.rectangle(frame,(int(centre[0]-nose_width/2),int(centre[1]-nose_height/2)),
        #     (int(centre[0]+nose_width/2),int(centre[1]+nose_height/2)),(0,255,0),2)

        # print((nose_width,nose_height))

#adding nose
        nose_dog = cv2.resize(nose_image,(nose_width, nose_height))
        nose_dog_gray = cv2.cvtColor(nose_dog, cv2.COLOR_BGR2GRAY)
        _,nose_mask = cv2.threshold(nose_dog_gray,25,255,cv2.THRESH_BINARY_INV)

        nose_area = frame[top_left[1]:top_left[1]+nose_height, top_left[0]:top_left[0]+nose_width]
        nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
        final_nose = cv2.add(nose_area_no_nose, nose_dog)

        frame[top_left[1]: top_left[1] + nose_height,
            top_left[0]: top_left[0] + nose_width] = final_nose


        # cv2.imshow("nose_area",nose_area)
        # cv2.imshow("nose_image",nose_dog)
        # cv2.imshow("nose_mask",nose_mask)
        # for n in range(68):
        #     x = landmarks.part(n).x
        #     y = landmarks.part(n).y
        #     cv2.circle(frame,(x,y),3,(0,255,0),-1)
    frame = cv2.Canny(frame,100,300)
    cv2.imshow("Frame",frame)
        
    

    key = cv2.waitKey(1)
    if key==27:
        break


cv2.destroyAllWindows()

