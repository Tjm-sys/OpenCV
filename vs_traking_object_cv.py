import os
import cv2 
import sys 
import numpy as np
import matplotlib.pyplot as plt

# Set up tracker 
tracker_types = [ 
   "BOOSTING", 
   "MIL", 
   "KCF", 
   "CSRT", 
   "TLD", 
   "MEDIANFLOW", 
   "GOTURN", 
   "MOSSE", 
] 
 
# Change the index to change the tracker type 
tracker_type = tracker_types[2] 
 
if tracker_type == "BOOSTING": 
   tracker = cv2.legacy.TrackerBoosting.create() 
elif tracker_type == "MIL": 
   tracker = cv2.legacy.TrackerMIL.create() 
elif tracker_type == "KCF": 
   tracker = cv2.legacy.TrackerKCF.create() 
elif tracker_type == "CSRT": 
   tracker = cv2.TrackerCSRT.create() 
elif tracker_type == "TLD": 
   tracker = cv2.legacy.TrackerTLD.create() 
elif tracker_type == "MEDIANFLOW": 
   tracker = cv2.legacy.TrackerMedianFlow.create() 
elif tracker_type == "GOTURN": 
   tracker = cv2.TrackerGOTURN.create() 
else: 
   tracker = cv2.legacy.TrackerMOSSE.create()

def drawRectangle(frame, bbox): 
   p1 = (int(bbox[0]), int(bbox[1])) 
   p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])) 
   cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1) 
 
def displayRectangle(frame, bbox): 
   plt.figure(figsize=(20, 10)) 
   frameCopy = frame.copy() 
   drawRectangle(frameCopy, bbox) 
   frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR) 
   plt.imshow(frameCopy) 
   plt.axis("off") 
 
def drawText(frame, txt, location, color=(50, 170, 50)): 
   cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

 
source = cv2.VideoCapture(0) 
win_name = 'Obect Tracking' 
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL) 

# Define a bounding box 
bbox = (1300, 405, 160, 120) 
ok, frame = source.read() 
displayRectangle(frame, bbox) 

# Initialize tracker with first frame and bounding box
tracker.init(frame, bbox)


while cv2.waitKey(1) != 27: # Escape 
    ok, frame = source.read() 
    if not ok: 
       break 
    cv2.imshow(win_name, frame)

     # Start timer 
    timer = cv2.getTickCount() 
 
    # Update tracker 
    ok, bbox = tracker.update(frame) 
 
    # Calculate Frames per second (FPS) 
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer) 
 
    # Draw bounding box 
    if ok: 
       drawRectangle(frame, bbox) 
    else: 
       drawText(frame, "Tracking failure detected", (80, 140), (0, 0, 255)) 
 
    # Display Info 
    drawText(frame, tracker_type + " Tracker", (80, 60)) 
    drawText(frame, "FPS : " + str(int(fps)), (80, 100)) 
 
source.release() 
cv2.destroyWindow(win_name) 