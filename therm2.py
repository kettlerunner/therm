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
    scale = 1
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2
    txt_size = cv2.getTextSize(text, font_face, scale, thickness)
    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin
    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
i2c = busio.I2C(board.SCL, board.SDA)
amg = adafruit_amg88xx.AMG88XX(i2c)

while(True):
    ret, img = cap.read()
    img  = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
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
 
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
