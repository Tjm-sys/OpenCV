import cv2 # type: ignore
import numpy as np # type: ignore

cap = cv2.VideoCapture(0)

#parametros de cores hsv, ajustar se nesessário depois de testes
green_low = np.array([40, 70, 70]) #verde é temporário, somente para testes
green_high = np.array([80, 255, 255])
black_low = np.array([0, 0, 0])
black_high = np.array([180, 255, 60])

kernel = np.ones((5,5), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    def detect_victims(mask, frame, label, color):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours: #filtra contornos para verificar circulos (-falsos positivos)

            area = cv2.contourArea(cnt)
            if area < 600: #evita ruido do mapa hsv
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter) 
            if circularity < 0.7: #verifica forma
                continue

            (x, y), r = cv2.minEnclosingCircle(cnt)
            x, y, r = int(x), int(y), int(r)

            cv2.circle(frame, (x,y), r, color, 3) #escreve/desenha os circulos e legendas
            cv2.putText(frame, label, (x-40,y-20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #mascaras/filtroa para cada cor
    green_mask = cv2.inRange(hsv, green_low, green_high) #verde é temporário, somente para testes
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    black_mask = cv2.inRange(hsv, black_low, black_high)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)

    detect_victims(green_mask, frame, "viva", (0,255,0))
    detect_victims(black_mask, frame, "morta", (0,0,255))

    cv2.imshow("Resgate", frame)
    cv2.imshow("Filtro Verde", green_mask)
    cv2.imshow("Filtro Preto", black_mask)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()