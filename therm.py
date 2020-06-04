#!/usr/bin/env python3

import os
import time
import math
import busio
import board
import cv2
import adafruit_amg88xx
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.interpolate import griddata
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

status = "reading"
face_in_frame = False
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
face_cascade = cv2.CascadeClassifier('/home/pi/Scripts/therm/haarcascade_frontalface_default.xml')
i2c = busio.I2C(board.SCL, board.SDA)
amg = adafruit_amg88xx.AMG88XX(i2c)
temp_offset = 25.0
display_temp = 98.6
ambient_temp = []
face_size = 0
heat_size = 0
room_temp = 0
og_frame = cv2.imread("/home/pi/Scripts/therm/static/img/therm_background.png")
blank_screen = cv2.imread("/home/pi/Scripts/therm/static/img/default2.png")
wait_ = cv2.imread("/home/pi/Scripts/therm/static/img/clock.png")
stop = cv2.imread("/home/pi/Scripts/therm/static/img/stop.png")
go = cv2.imread("/home/pi/Scripts/therm/static/img/go.png")
cv2.namedWindow('therm', cv2.WINDOW_FREERATIO)
cv2.setWindowProperty('therm', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
points = [(math.floor(ix / 8), (ix % 8)) for ix in range(0,64)]
grid_x, grid_y = np.mgrid[0:7:64j, 0:7:64j]
x_offset = 75
y_offset = 90

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('therm.avi', fourcc, 10.0, (800,480))

while(True):
    ret, img = cap.read()
    img  = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.flip(img, 1)
    frame = og_frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    face_sizes = []
    for (x, y, w, h) in faces:
        face_sizes.append(w*h)
        cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (255, 255, 255), 2)
        
    if len(face_sizes) > 0:
        (x, y, w, h) = faces[np.argmax(face_sizes)]
        tx = int(x+w/2-75)
        ty = int(y+h/2-75)
        if tx < 0: tx = 0
        if ty < 0: ty = 0
        bx = tx + 150
        by = ty + 150
        if bx > 240:
            tx = tx - (bx-240)
            bx = tx + 150 
        img = img[ty:ty+150, tx:bx]
        img = cv2.resize(img,(300,300))
        faces = faces[np.argmax(face_sizes):np.argmax(face_sizes)+1]
    else:
        tx = int(img.shape[1]/2 - 75)
        ty = int(img.shape[0]/2 - 75)
        img = img[ty:ty+150, tx:tx+150]
            
    pixels = np.fliplr(np.rot90(np.asarray(amg.pixels), k=3)).flatten()
    label = "Room Temp: {0:.1f} F".format(room_temp)
    draw_label(frame, label, (490,210), (255,255,255))
    label = "Stdev: {0:.4f}".format(np.std(ambient_temp))
    draw_label(frame, label, (490, 230), (255,255,255))
    if type(faces) is tuple:
        frame[y_offset:y_offset+300, x_offset:x_offset+300] = blank_screen
        if room_temp == 0:
            ambient_temp = []
        if len(ambient_temp) == 100:
            ambient_temp = ambient_temp[1:]
        temp_scan = np.asarray(amg.pixels).flatten()
        temp_scan_f = (9/5)*temp_scan + 32
        room_f = temp_scan_f[temp_scan_f > 50.0]
        room_f = room_f[room_f < 100.0]
        if len(room_f) >= 1:
            ambient_temp.append( np.average(room_f))
        room_temp = np.average(ambient_temp)
        if face_in_frame:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            output_data = { "Reading" : [display_temp], "Room_Temp" [room_temp], "Face_Size" [face_size], "Heat_Size" [heat_size], "TOD" [current_time]  }
            df = pd.DataFrame(output_data)
            if os.path.exists("data.csv"):
                df.to_csv('data.csv', mode='a', header=False)
            else:
                df.to_csv('data.csv', header=True)
            if display_temp >= 80:
                client = Client(account_sid, auth_token)
                client.messages.create(
                    body="A scan of {0:.1f} F was detected by Thermie.".format(display_temp),
                    from_="+19202602260",
                    to="+19206295560"
                )
        display_temp = 98.6
        face_in_frame = False
    else:
        max_face_index = 0
        max_face_size = 0
        mx = 0
        my = 0
        mw = 0
        mh = 0
        i = 0
        for (x, y, w, h) in faces:
            if max_face_size < w*h:
                max_face_index = i
                max_face_size = w*h
                mx = x
                my = y
                mw = w
                mh = h
            i += 1
        if face_in_frame == False:
            face_in_frame = True
            body_temp = []
        face_size = mh*mw
        if mh*mw < 1000:
            label = "Please step closer."
            draw_label(img, label, (20, 30), (255, 255, 255))
            frame[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
        elif mh*mw >= 6000:
            label = "Please step back a bit."
            draw_label(img, label, (20, 30), (255, 255, 255)) 
            frame[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
        else:
            temp_scan = np.fliplr(np.rot90(np.asarray(amg.pixels), k=3)).flatten()
            pixels_f = (9/5)*pixels+32
            grid_z = griddata(points, pixels_f, (grid_x, grid_y), method='cubic')
            flat_grid = grid_z.flatten()
            filtered_flat_grid = flat_grid[flat_grid >=70]
            flat_grid = filtered_flat_grid[filtered_flat_grid <=95]
            hist, bin_edges = np.histogram(flat_grid, bins=16)
            grid_z[grid_z < bin_edges[len(bin_edges) - 4]] = 0
            x_scatter_data = []
            y_scatter_data = []
            for y, row in enumerate(grid_z):
                for x, cell in enumerate(row):
                    if cell != 0:
                        x_scatter_data.append(x)
                        y_scatter_data.append(63 - y)
            found_groups = False
            group_count = 0
            data_grid = np.dstack((x_scatter_data, y_scatter_data))[0]
            j = 1
            while found_groups == False and j < 11:
                kmeans = KMeans(n_clusters=j, init='k-means++', max_iter=10, n_init=10, random_state=0)
                kmeans.fit(data_grid)
                j += 1
                if kmeans.inertia_ < 15000 and found_groups == False:
                    group_count = j
                    found_groups = True
            if group_count > 0:
                try:
                    kmeans = KMeans(n_clusters=group_count, init="k-means++", max_iter=10, n_init=10, random_state=0)
                    pred_y = kmeans.fit_predict(data_grid)
                except:
                    print("Lost track of object.")
            i = 0
            max_size = 0
            group_index = 0
            temp_reading = 0
            group_size = []
            group_dims = []
            while i < group_count:
                series = data_grid[pred_y == i]
                total = 0
                data_buffer = []
                for cell in series:
                    data_buffer.append(grid_z[63-cell[1]][cell[0]])
                    total += grid_z[63 - cell[1]][cell[0]]
                zone_average = total / len(series)
                if max_size < len(data_buffer):
                    max_size = len(data_buffer)
                    heat_size = max_size
                    group_index = i
                    temp_reading = zone_average
                i += 1
            if max_size < 20:
                label = "Please step closer."
                draw_label(img, label, (20, 30), (255, 255, 255))
                frame[300:400, 550:650] = wait_
                face_in_frame = False
            elif max_size > 150:
                label = "Please step back a bit."
                draw_label(img, label, (20, 30), (255, 255, 255))
                frame[300:400, 550:650] = wait_
                face_in_frame = False
            else:
                if room_temp > 73:
                    correction_factor = 16
                elif room_temp > 72:
                    correction_factor = 17
                elif room_temp > 70:
                    correction_factor = 18
                elif room_temp > 65:
                    correction_factor = 19
                else:
                    correction_factor = 20
                if len(body_temp) >= 10:
                    body_temp = body_temp[1:]
                body_temp.append(temp_reading + correction_factor)
                display_temp = np.average(body_temp)
                label = "Observed Temp: {0:.2f} F".format(display_temp)
                draw_label(frame, label, (490, 250), (255,255,255))
                if display_temp >= 100.0:
                    frame[300:400, 550:650] = stop
                    status = "high"
                else:
                    frame[300:400, 550:650] = go
                    status = "normal"  
            frame[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
    #out.write(frame)
    cv2.imshow('therm', frame)
    #out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#out.release()
cv2.destroyAllWindows()
