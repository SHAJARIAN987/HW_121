#Import Libraries
import cv2
import numpy as np
import time

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('out.avi', fourcc, 20.0, (640, 480))
onthemoon = cv2.resize(cv2.imread('OnTheMoon.jpg'), (640,480))

cam = cv2.VideoCapture(0)

while(cam.isOpened):
    frametf, frame = cam.read()
    if not frametf:
        break

    frame = np.flip(frame, axis = 1)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 100])
    upper_white = np.array([360, 255, 255])
    mask_1 = cv2.inRange(hsv_frame, lower_white, upper_white)

    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask_1 = cv2.morphologyEx(mask_1, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))

    mask_2 = cv2.bitwise_not(mask_1)

    alt_frame = cv2.bitwise_and(frame, frame, mask = mask_2)
    alt_moon = cv2.bitwise_and(onthemoon, onthemoon, mask = mask_1)


    output = cv2.addWeighted(alt_frame, 1, alt_moon, 1, 0)
    output_file.write(output)
    cv2.imshow('mask_1', output)
    cv2.waitKey(2)

cam.release()
cv2.destroyAllWindows()