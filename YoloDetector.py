import cv2
from ultralytics import YOLO
from sort import *
import numpy as np
import pandas as pd
import time
import calendar
col=['time','date','day of the week','carcount','bikecount','buscount','truckcount','total']
df = pd.DataFrame(columns=col)
yolo = YOLO('yolov8n.pt')
videoCap = cv2.VideoCapture("project1.mp4")
tracker = Sort(max_age=20,min_hits=2,iou_threshold=0.3)
enter = Sort(max_age=20,min_hits=2,iou_threshold=0.3)
bus_tk = Sort(max_age=20,min_hits=2,iou_threshold=0.3)
bike_tk = Sort(max_age=20,min_hits=2,iou_threshold=0.3)
'''#[[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]'''
line = [100,200,500,200]
input_lis = []
seconds = 0
zero=[]
one = []
bus_lis =[]
bike_lis =[]
while True:
    from datetime import datetime
    now = datetime.now()
    time.sleep(1)
    seconds = seconds+1
    ret, frame = videoCap.read()
    if not ret:
        break
    results = yolo.track(frame)
    detections = np.empty((0,5))
    truck = np.empty((0,5))
    bus = np.empty((0,5))
    bike = np.empty((0,5))
    for result in results:
        name = result.names
    for box in result.boxes:   #retrive information
        [x1,y1,x2,y2] = box.xyxy[0]       # retrive info
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        cls = int(box.cls[0])
        class_name = name[cls]
        colour =[255,0,0]
        if class_name == "car": #To Detect the cars
            cv2.rectangle(frame, (x1,y1),(x2,y2),colour,2)    #display
            cv2.putText(frame,f'{name[int(box.cls[0])]}{box.conf[0]:.2f}',(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,colour,2)
            currentarray = np.array([x1,y1,x2,y2,box.conf[0]])
            detections = np.vstack((detections,currentarray))
        elif class_name == "truck": # To detect the trucks
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)  # display
            cv2.putText(frame, f'{name[int(box.cls[0])]}{box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        colour, 2)
            currenttr = np.array([x1, y1, x2, y2, box.conf[0]])
            truck = np.vstack((truck, currenttr))
        elif class_name == "bike": # To detect the bike
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)  # display
            cv2.putText(frame, f'{name[int(box.cls[0])]}{box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        colour, 2)
            currentbike = np.array([x1, y1, x2, y2, box.conf[0]])
            bike = np.vstack((bike, currentbike))
        elif class_name == "bus": # To detect the bus
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)  # display
            cv2.putText(frame, f'{name[int(box.cls[0])]}{box.conf[0]:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        colour, 2)
            currentbus = np.array([x1, y1, x2, y2, box.conf[0]])
            bus = np.vstack((bus, currentbus))


    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
    resultstrackers = tracker.update(detections) # tracker for cars
    trucktracker = enter.update(truck) # tracker for trucks
    bustracker = bus_tk.update(bus) # tracker for bus
    biketracker = bike_tk.update(bike) # tracker for bike

    cv2.line(frame,(line[0],line[1]),(line[2],line[3]),(0,0,255),5)
    #
    for delta in resultstrackers:
        x1,y1,x2,y2,Id = delta
        x1,y1,x2,y2 = int(x1),int(y2),int(x2),int(y2)
        w,h = x2-x1, y2-y1
        print(delta)
        cx,cy = x1+w//2,y1+h//2
        cv2.circle(frame,(cx,cy),5,(255,0,255),cv2.FILLED)
        if line[0]<cx<line[2] and line[1]-5<cy<line[1]+5:
            if zero.count(Id) ==0:
                zero.append(Id)
    for r in trucktracker:
        X1,Y1,X2,Y2,ID = r
        X1,Y1,X2,Y2 = int(X1),int(Y2),int(X2),int(Y2)
        W,H = X2-X1, Y2-Y1
        print(r)
        CX,CY = X1+W//2,Y1+H//2
        cv2.circle(frame,(CX,CY),5,(255,0,255),cv2.FILLED)
        if line[0]<CX<line[2] and line[1]-5<CY<line[1]+5:
            if one.count(ID) ==0:
                one.append(ID)
    for a in bustracker:
        a1, b1, a2, b2, ed = a
        a1, b1, a2, b2 = int(a1), int(b2), int(a2), int(b2)
        e, d = a2 - a1, b2 - b1
        #print(r)
        c1, c2 = a1 + e // 2, b1 + d // 2
        cv2.circle(frame, (c1, c2), 5, (255, 0, 255), cv2.FILLED)
        if line[0] < c1 < line[2] and line[1] - 5 < c2 < line[1] + 5:
            if bus_lis.count(ed) == 0:
                bus_lis.append(ed)

    for b in biketracker:
        A1, B1, A2, B2, ED = b
        A1, B1, A2, B2 = int(A1), int(B2), int(A2), int(B2)
        E, D = A2 - A1, B2 - B1
        # print(r)
        C1, C2 = A1 + E // 2, B1 + D // 2
        cv2.circle(frame, (C1, C2), 5, (255, 0, 255), cv2.FILLED)
        if line[0] < C1 < line[2] and line[1] - 5 < C2 < line[1] + 5:
            if bike_lis.count(ED) == 0:
                bike_lis.append(ED)


    if seconds == 5:
        input_lis.append(f"{now.hour:02}:{now.minute:02}:{now.second:02}")
        input_lis.append(now.day)
        input_lis.append(calendar.weekday(now.year, now.month, now.day))
        input_lis.append(len(zero))
        input_lis.append(len(one))
        input_lis.append(len(bike_lis))
        input_lis.append(len(bus_lis))
        input_lis.append(len(zero) + len(one) + len(bike_lis) + len(bus_lis))
        df.loc[len(df)] = input_lis
        zero.clear()
        one.clear()
        bike_lis.clear()
        bus_lis.clear()
        seconds = 0
        input_lis.clear()
    #print(len(zero),len(one))
    cv2.putText(frame,f'Car:{len(zero)}',(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,colour,2)
    cv2.putText(frame, f'truck:{len(one)}', (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
    cv2.putText(frame,f'Bike:{len(bike_lis)}',(200,200),cv2.FONT_HERSHEY_SIMPLEX,1,colour,2)
    cv2.putText(frame,f'Bus:{len(bus_lis)}',(250,250),cv2.FONT_HERSHEY_SIMPLEX,1,colour,2)
    cv2.imshow("frame", frame)
    cv2.waitKey(0)
    print(df)
videoCap.release()
cv2.destroyALLWindows()

