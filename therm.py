#stable version. Just need to add the ambient temp offset and data correlation.

import time
import math
import busio
import board
import adafruit_amg88xx
import matplotlib
import numpy as np
import numpy as np
import cv2

def draw_label(img, text, pos, bg_color):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)
    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin
    start_x = pos[0] - margin
    start_y = pos[1] + margin
    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
i2c = busio.I2C(board.SCL, board.SDA)
amg = adafruit_amg88xx.AMG88XX(i2c)
frame = cv2.imread("static/img/therm_background.png")
cv2.namedWindow('therm', flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)
cv2.moveWindow('therm', 220, 30)

while(True):
    ret, img = cap.read()
    img  = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    x_offset = 40
    y_offset = 120
    crop_width = 300
    crop_height = 300
    img = img[y_offset:y_offset+crop_height, x_offset:x_offset+crop_width]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if type(faces) is tuple:
        draw_label(img, 'No Face Detected', (20,30), (255,255,255))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y+5), (x+w, y+h), (255, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if type(eyes) is tuple:
            label = "Please face the cameras."
            draw_label(img, label, (20, 30), (255, 255, 255))
        else:
            if eyes.shape[0] >= 2:
                if h*w < 20000:
                    label = "Please step closer."
                    draw_label(img, label, (20, 30), (255, 255, 255))
                elif h*w >= 40000:
                    label = "Please step back a bit."
                    draw_label(img, label, (20, 30), (255, 255, 255)) 
                else:
                    max_temp = np.amax(amg.pixels)
                    max_temp_f = (9/5)*max_temp + 32
                    label = "Surface temp: {0:.1f} F".format(max_temp_f)
                    draw_label(img, label, (20, 30), (255,255,255))
    x_offset = 75
    y_offset = 90
    frame[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
    cv2.imshow('therm', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
