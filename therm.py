#!/usr/bin/env python3

import os
import busio
import board
import adafruit_amg88xx
import numpy as np
import cv2
from twilio.rest import Client

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

account_sid = os.environ['ACCOUNT_SID']
auth_token = os.environ['AUTH_TOKEN']

#out = cv2.VideoWriter('therm.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (800,480))

face_in_frame = False
temp_readings = []
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('/home/pi/Scripts/therm/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/pi/Scripts/therm/haarcascade_eye.xml')
i2c = busio.I2C(board.SCL, board.SDA)
amg = adafruit_amg88xx.AMG88XX(i2c)
ambient_temp = [ 65 ]
temp_offset = 24.0
alpha = 1
corrected_temp = [ 98.6 ]
display_temp = 98.6
og_frame = cv2.imread("/home/pi/Scripts/therm/static/img/therm_background.png")
blank_screen = cv2.imread("/home/pi/Scripts/therm/static/img/default.png")
wait_ = cv2.imread("/home/pi/Scripts/therm/static/img/clock.png")
stop = cv2.imread("/home/pi/Scripts/therm/static/img/stop.png")
go = cv2.imread("/home/pi/Scripts/therm/static/img/go.png")
cv2.namedWindow('therm', cv2.WINDOW_FREERATIO)
cv2.setWindowProperty('therm', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('therm.avi', fourcc, 10.0, (800,480))

while(True):
    ret, img = cap.read()
    img  = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.flip(img, 1)
    frame = og_frame.copy()
    x_offset = 180
    y_offset = 120
    crop_width = 300
    crop_height = 300
    img = img[y_offset:y_offset+crop_height, x_offset:x_offset+crop_width]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    pixels = np.asarray(amg.pixels).flatten()
    label = "Room Temp: {0:.1f} F".format(np.average(ambient_temp))
    draw_label(frame, label, (490,210), (255,255,255))
    label = "Stdev: {0:.4f}".format(np.std(ambient_temp))
    draw_label(frame, label, (490, 230), (255,255,255))
    if type(faces) is tuple:
        if np.std(pixels) < 1.5:
            if len(ambient_temp) == 100:
                ambient_temp = ambient_temp[1:]
            ambient_temp.append( 9/5*np.average(pixels)+32 )
        draw_label(img, 'No Face Detected', (20,30), (255,255,255))
        if face_in_frame:
            corrected_temp = [ 98.6 ]
            display_temp = 98.6
            temp_readings = []
            face_in_frame = False
            if display_temp >= 100 and alpha <= 0.05:
                client = Client(account_sid, auth_token)
                client.messages.create(
                    body="A scan of {0:.1f} F was detected by Thermie.".format(display_temp),
                    from_="+19202602260",
                    to="+19206295560"
                )
            if display_temp < 100 and alpha <= 0.05:
                message_body = "A scan of {temp:.1f} F was detected by Thermie. \n\nRoom temp: {room_temp:.1} F \nalpha: {alpha:.4}"
                client = Client(account_sid, auth_token)
                client.messages.create(
                    body=message_body.format(temp=display_temp, room_temp = np.average(ambient_temp), alpha=alpha),
                    from_="+19202602260",
                    to="+19206295560"
                )
            
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y+5), (x+w, y+h), (255, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if type(eyes) is tuple:
            label = "Please face the cameras."
            draw_label(img, label, (20, 30), (255, 255, 255))
        else:
            if eyes.shape[0] >= 1:
                if h*w < 12000:
                    label = "Please step closer."
                    draw_label(img, label, (20, 30), (255, 255, 255))
                elif h*w >= 35000:
                    label = "Please step back a bit."
                    draw_label(img, label, (20, 30), (255, 255, 255)) 
                else:
                    temp_scan = np.asarray(amg.pixels).flatten()
                    temp_scan_f = (9/5)*temp_scan + 32
                    human_f = temp_scan_f[temp_scan_f > 70.0]
                    human_f = human_f[human_f < 95.0]
                    if face_in_frame:
                        temp_readings.append(np.average(human_f) + temp_offset)
                    else:
                        temp_readings = [np.average(human_f) + temp_offset]
                        face_in_frame = True                    
                    corrected_temps = temp_readings
                    if len(corrected_temp) > 10 or np.std(corrected_temp) > 0.10:
                        corrected_temp = corrected_temp[1:]
                    corrected_temp.append(np.average(corrected_temps))
                    display_temp = np.average(corrected_temp)
                    alpha = np.std(corrected_temp)
                    label = "alpha: {0:.4f}".format(np.std(corrected_temp))
                    draw_label(frame, label, (490, 270), (255,255,255))
                    label = "Temp: {0:.1f} F".format(display_temp)
                    draw_label(img, label, (40, 30), (255,255,255))
                    if alpha <= 0.05:
                        label = "Observed Temp: {0:.1f} F".format(display_temp)
                        draw_label(frame, label, (490, 250), (255,255,255))
                        if display_temp >= 101.0:
                            frame[300:400, 550:650] = stop
                        else:
                            frame[300:400, 550:650] = go
                    else:
                        label = "Reading Temp. Please Wait."
                        draw_label(frame, label, (490, 250), (255,255,255))
                        frame[300:400, 550:650] = wait_
                        
    x_offset = 75
    y_offset = 90
    if face_in_frame:
        frame[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
    else:
        frame[y_offset:y_offset+300, x_offset:x_offset+300] = blank_screen
    #out.write(frame)
    cv2.imshow('therm', frame)
    #out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#out.release()
cv2.destroyAllWindows()
