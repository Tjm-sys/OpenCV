import cv2
import numpy as np

videoCapture = cv2.VideoCapture(0)
prevCircle = None
dist = lambda x1, y1, x2, y2: (x1 - x2)**2 + (y1 - y2)**2

silver_low = np.array([0, 0, 170])
silver_high = np.array([180, 40, 255])

while True:
    ret, frame = videoCapture.read()
    if not ret: break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #filtro
    silver_mask = cv2.inRange(hsv, silver_low, silver_high) #cria mask/filtro prata
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #mask/filtro para regular brilo (-falsos positivos)
    bright_mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -5)
    silver_mask = cv2.bitwise_and(silver_mask, bright_mask) #combina masks/filtros
    kernel = np.ones((3,3), np.uint16)
    silver_mask = cv2.morphologyEx(silver_mask, cv2.MORPH_OPEN, kernel)
    silver_mask = cv2.morphologyEx(silver_mask, cv2.MORPH_CLOSE, kernel)
    blur = cv2.GaussianBlur(silver_mask, (9, 9), 2)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=100,
        param2=18,
        minRadius=20,
        maxRadius=60)
    
    if circles is  None:
        prevcircle = None
    elif circles is not None:
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None: chosen = i
            if prevcircle is not None:
                if dist(chosen[0], chosen[1], prevcircle[0], prevcircle[1]) > dist(i[0], i[1], prevCircle[0], prevCircle[1]):
                    chosen = i

        cv2.circle(frame, (chosen[0], chosen[1]), 1, (0, 100, 100), 3)
        cv2.circle(frame, (chosen[0], chosen[1]), chosen[2], (0, 255, 0), 3)
        prevcircle = chosen

    cv2.imshow("Frame", frame)
    cv2.imshow("Prata", silver_mask)

    if cv2.waitKey(1) == 27: 
        break

videoCapture.release()
cv2.destroyAllWindows()
