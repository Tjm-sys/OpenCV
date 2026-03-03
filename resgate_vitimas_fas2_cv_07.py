import cv2 # type: ignore
import numpy as np # type: ignore

videoCapture = cv2.VideoCapture(0)

circle_ant = None #cria variaveis para prioridade da esfera
center_cal = None

stable_count = 0
stable_target = None
tolerance = 15 #AJUSTAR, 'sensibilidade' para verificar as vitimas
stable_frames = 50 #AJUSTAR 'tempo' para verificar as vitimas
dist = lambda x1, y1, x2, y2: (x1 - x2)**2 + (y1 - y2)**2

silver_low = np.array([0, 0, 170]) #paranetros da mask prata
silver_high = np.array([180, 40, 255])
black_low = np.array([0, 0, 0]) #paranetros da mask preta
black_high = np.array([180, 255, 60])

while True:
    ret, frame = videoCapture.read() #verifica se há captura de imagem
    if not ret: break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #filtro
    
    #config da mask/filtro prata
    silver_mask = cv2.inRange(hsv, silver_low, silver_high) #cria mask/filtro prata
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #mask/filtro para regular brilo (-falsos positivos)
    bright_mask = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,-5)
    silver_mask = cv2.bitwise_and(silver_mask, bright_mask) #combina masks/filtros
    kernel = np.ones((3,3), np.uint8)
    silver_mask = cv2.morphologyEx(silver_mask, cv2.MORPH_OPEN, kernel)
    silver_mask = cv2.morphologyEx(silver_mask, cv2.MORPH_CLOSE, kernel)
    silver_blur = cv2.GaussianBlur(silver_mask, (9, 9), 2)

    #config da mask/filtro preto
    black_mask = cv2.inRange(hsv, black_low, black_high)
    kernel = np.ones((3,3), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    black_blur = cv2.GaussianBlur(black_mask, (9, 9), 2)

    masks = [black_blur, silver_blur] #lista de masks

    #parametros dos circulos
    black_circles = cv2.HoughCircles(black_blur,cv2.HOUGH_GRADIENT,dp=1.2, #morta
        minDist=80,
        param1=100,param2=18,minRadius=20,maxRadius=60)
    silver_circles = cv2.HoughCircles(silver_blur,cv2.HOUGH_GRADIENT,dp=1.2, #viva
        minDist=80,
        param1=100,param2=18,minRadius=20,maxRadius=60)

    detections = [(black_circles, (255, 0, 0)), (silver_circles, (0, 255, 0))]

    for circles, color in detections:
        if circles is None:
            continue

        circles = np.uint16(np.around(circles))

        # Escolhe o maior círculo
        chosen = max(circles[0, :], key=lambda c: c[2])
        current_center = (chosen[0], chosen[1])

        if stable_target is None:
            stable_target = current_center
            stable_count = 1
        else:
            d = np.sqrt(dist(current_center[0], current_center[1],stable_target[0], stable_target[1]))

            if d < tolerance:
                stable_count += 1
            else:
                stable_target = current_center
                stable_count = 1

        if center_cal is None:
            center_cal = [chosen[0], chosen[1]]
        else:
            alpha = 0.7
            center_cal[0] = int(alpha * center_cal[0] + (1 - alpha) * chosen[0])
            center_cal[1] = int(alpha * center_cal[1] + (1 - alpha) * chosen[1])

        circle_ant = chosen

        cv2.circle(frame, (center_cal[0], center_cal[1]), 1, color, 3)
        cv2.circle(frame, (center_cal[0], center_cal[1]), chosen[2], color, 3)

        if stable_count >= stable_frames:
            cv2.putText(frame,"Vitima confirmada/estabilizada",(50, 50),cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 0),2)

    cv2.imshow("Frame", frame) #display do frame
    cv2.imshow("Prata", silver_mask) 
    cv2.imshow("Preto", black_mask)

    if cv2.waitKey(1) == 27: #tecla para fechar o programa (quebrar o loop)
        break

videoCapture.release()
cv2.destroyAllWindows()