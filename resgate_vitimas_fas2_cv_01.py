import cv2 # type: ignore
import numpy as np # type: ignore

cap = cv2.VideoCapture(0)

#parametros de filtro hsv
green_low = np.array([40, 70, 70])
green_high = np.array([80, 255, 255])

black_low = np.array([0, 0, 0])
black_high = np.array([180, 255, 60])

kernel = np.ones((5,5), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, green_low, green_high)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black_mask = cv2.inRange(hsv, black_low, black_high)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in green_contours:
        area = cv2.contourArea(cnt)

        if area < 1000:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        x, y, r = int(x), int(y), int(radius)

        victim_type = "ALIVE"
        color = (0,200,0)

        cv2.circle(frame, (x,y), r, color, 3)
        cv2.putText(frame, victim_type, (x-40,y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    for cnt in black_contours:
        area = cv2.contourArea(cnt)

        if area < 1000:
            continue

        (xb, yb), radius = cv2.minEnclosingCircle(cnt)
        xb, yb, rb = int(xb), int(yb), int(radius)

        victim_typeb = "DEAD"
        colorb = (0,0,0)

        cv2.circle(frame, (xb,yb), rb, colorb, 3)
        cv2.putText(frame, victim_typeb, (xb-40,yb-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colorb, 2)

    cv2.imshow("Resgate", frame)
    cv2.imshow("Filtro Verde", green_mask)
    cv2.imshow("Filtro Preto", black_mask)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()