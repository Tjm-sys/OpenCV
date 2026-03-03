import cv2
import numpy as np

videoCapture = cv2.VideoCapture(0)
circle_ant = None
center_cal = None
dist = lambda x1, y1, x2, y2: (x1 - x2)**2 + (y1 - y2)**2

silver_low = np.array([0, 0, 170])
silver_high = np.array([180, 40, 255])

while True:
    ret, frame = videoCapture.read()
    if not ret: break

    #config da mask/filtro prata
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #filtro
    silver_mask = cv2.inRange(hsv, silver_low, silver_high) #cria mask/filtro prata
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #mask/filtro para regular brilo (-falsos positivos)
    bright_mask = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,-5)
    silver_mask = cv2.bitwise_and(silver_mask, bright_mask) #combina masks/filtros
    kernel = np.ones((3,3), np.uint8)
    silver_mask = cv2.morphologyEx(silver_mask, cv2.MORPH_OPEN, kernel)
    silver_mask = cv2.morphologyEx(silver_mask, cv2.MORPH_CLOSE, kernel)
    silver_blur = cv2.GaussianBlur(silver_mask, (9, 9), 2)

    #config da mask/filtro preta
    black_low = np.array([0, 0, 0])
    black_high = np.array([180, 255, 60])
    black_mask = cv2.inRange(hsv, black_low, black_high)
    kernel = np.ones((3,3), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    black_blur = cv2.GaussianBlur(black_mask, (9, 9), 2)

    masks = [black_blur, silver_blur]

    #parametros dos circulos
    black_circles = cv2.HoughCircles(black_blur,cv2.HOUGH_GRADIENT,dp=1.2,
        minDist=80,
        param1=100,
        param2=18,
        minRadius=20,
        maxRadius=60)
    silver_circles = cv2.HoughCircles(silver_blur,cv2.HOUGH_GRADIENT,dp=1.2,
        minDist=80,
        param1=100,
        param2=18,
        minRadius=20,
        maxRadius=60)

    detections = [(black_circles, (255, 0, 0)), 
                (silver_circles, (0, 255, 0))]

    for circles, color in detections:
        if circles is None:
            continue
        circles = np.uint16(np.around(circles))
        chosen = None
        for i in circles[0, :]:
            if chosen is None:
                chosen = i
            if circle_ant is not None:
                if dist(chosen[0], chosen[1], circle_ant[0], circle_ant[1]) > dist(i[0], i[1], circle_ant[0], circle_ant[1]):
                    chosen = i

        if center_cal is None:
            center_cal = [chosen[0], chosen[1]]
        else:
            alpha = 0.7  #AJUSTAR, suavização do movimento
            center_cal[0] = int(alpha * center_cal[0] + (1-alpha) * chosen[0])
            center_cal[1] = int(alpha * center_cal[1] + (1-alpha) * chosen[1])

        circle_ant = chosen

        cv2.circle(frame, (center_cal[0], center_cal[1]), 1, color, 3)
        cv2.circle(frame, (center_cal[0], center_cal[1]), chosen[2], color, 3)
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Prata", silver_mask)
    cv2.imshow("Preto", black_mask)

    if cv2.waitKey(1) == 27: 
        break

videoCapture.release()
cv2.destroyAllWindows()