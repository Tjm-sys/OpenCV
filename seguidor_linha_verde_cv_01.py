import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #cofig do filtro/mask, preto
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    #cofig do filtro/mask, verde
    lower_green = np.array([40, 70, 70])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    #Linha preta
    contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #função da biblioteca CV para contornos após o filtro da linha

    if contours_black:
        c = max(contours_black, key=cv2.contourArea)

        if cv2.contourArea(c) > 500:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2) #RGB -> BGR
            cv2.putText(frame, "Linha", (x,y-10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)

    #Quadrado verde
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #função da biblioteca CV para contornos após o filtro do verde

    if contours_green:
        for c in contours_green:
            area = cv2.contourArea(c)

            if area > 1000:
                approx = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)

                #Testa os lados para verificar se é um quadrado/retângulo
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
                    cv2.putText(frame, "Verde", (x,y-10),
                                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask Black", mask_black)
    cv2.imshow("Mask Green", mask_green)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()