import cv2 # type: ignore
import numpy as np # type: ignore
import time

cap = cv2.VideoCapture(0) #indica a porta da cam
motor_dir = 0 #desconectado temporariamente
motor_esq = 0 #desconectado temporariamente

lower_black = np.array([0, 0, 0]) #cofig do filtro/mask, verde/preto
upper_black = np.array([180, 255, 60])
lower_green = np.array([40, 70, 70])
upper_green = np.array([80, 255, 255])

tempo_ant = time.time() #para o dt
erro_pos_ant = 0 #variavel para o calculo do derivativo do erro de posição

#K = constante que define o quanto que reajimos ao erro #P=presente, D=futuro, I=passado
Kp_pos = 0.5 #AJUSTAR+: -se tremer, +se demorar para reajir: regula o centro. #Kp = Proporcional
Kp_ang = 2.5 #AJUSTAR: +se sair da linha em curvas: regula o angulo. #Kp = Proporcional
Kd = 0.02 #AJUSTAR: -se demorar para reajir Kd = Derivativo         

erro_pos_flit = 0 #variaveis para o calculo do filtro do Kp
erro_ang_flit = 0
alpha = 0.2   #AJUSTAR, quanto menor mais 'suave'

derivativo_filt = 0 #filtro do Kd
alpha_d = 0.2 #AJUSTAR, quanto menor mais

#ROI
#altura, largura, _ = frame.shape
#roi = frame[int(altura*0.6):altura, 0:largura]
#ROI #hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

while True:
    ret, frame = cap.read() #verifica a leitura da cam
    if not ret:
        break

    #cofig do mapa visual e do filtro/mask preto/verde
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #converte imagem para HSV (tonalidade, saturação, valor)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    #mask_green = cv2.medianBlur(mask_green, 5) #mask_green = cv2.erode(mask_green, None, iterations=1) #mask_green = cv2.dilate(mask_green, None, iterations=2)
    
    #calculos do Dt
    tempo_atual = time.time()
    dt = tempo_atual - tempo_ant
    tempo_ant = tempo_atual
    if dt <= 0:
        dt = 0.0001

    #LINHA
    contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #função da biblioteca CV para contornos após o filtro da linha
    if contours_black:
        c = max(contours_black, key=cv2.contourArea)

        if cv2.contourArea(c) > 500:
            rect = cv2.minAreaRect(c) #função cv, retorna centro X, Y, largura, altura e ângulo
            angle = rect[2] #ângulo da linha

            if angle < -45: #reajusta o ângulo
                angle = 90 + angle

            #CONTROLE
            centro_frame = frame.shape[1] // 2 #centro do frame
            cl = int(rect[0][0]) #centro da linha

            erro_pos = cl - centro_frame

            derivativo = (erro_pos - erro_pos_ant) / dt
            derivativo_filt = (1 - alpha_d) * derivativo_filt + alpha_d * derivativo

            erro_pos_flit = (1 - alpha) * erro_pos_flit + alpha * erro_pos #'filtro' deixa o movimento mais continuo
            erro_ang_flit = (1 - alpha) * erro_ang_flit + alpha * angle 

            #calculo da correção com PD e filtro
            correcao = (
                Kp_pos * erro_pos_flit +
                Kp_ang * erro_ang_flit +
                Kd * derivativo_filt
            )

            correcao = max(min(correcao, 100), -100) #**limita a correção
            erro_pos_ant = erro_pos #salva o erro anterior
            
            #MOTORES: esperar para poder testar
            #Vmax = 140 #limite máximo e mínimo da vel
            #Vmin = 70
            #Kvel = 1.2
            #vel_base = Vmax - Kvel * abs(erro_ang_flit) #calculo da velocidade
            #vel_base = max(Vmin, min(Vmax, vel_base)) #aplica os limites de velocidade

            vel_base = 120 #temporário
            motor_esq_mov = vel_base - correcao #aplica a correção
            motor_dir_mov = vel_base + correcao

            #DISPLAY DE DADOS
            centro_frame = frame.shape[1] // 2
            altura = frame.shape[0]
            cv2.line(frame, (centro_frame, 0), (centro_frame, altura), (0,0,0), 1)

            cv2.putText(frame, f"Erro Pos: {erro_pos}", (20,30),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0), 1)  #visualizar o angulo e a diferença entre a linha e o centro
            cv2.putText(frame, f"Erro Ang: {round(angle,1)}", (20,60),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0), 1)
            #cv2.putText(frame, f"Correção: {correcao}", (20,120),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0), 1)

            box = cv2.boxPoints(rect) #define propriedades da "box" (borda da linha)
            box = np.int32(box)

            cx = int(rect[0][0]) #define propriedades do circulo (centro da linha)
            cy = int(rect[0][1])

            cv2.circle(frame, (cx, cy), 2, (0,0,255), -1)
            cv2.drawContours(frame, [box], 0, (0,0,255), 2) #ilustra o contortno da linha e escreve a legenda
            cv2.putText(frame, "Linha", (box[0][0], box[0][1]-10),cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)

    #VERDE
    centros_verdes = [] #lista para identificar 2 verdes
    dir = ""
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #função da biblioteca CV para contornos após o filtro do verde
    if contours_green:
        for c in contours_green:
            area = cv2.contourArea(c)

            if area > 200: #AJUSTAR a área/tamanho min do verde
                approx = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)

                #aspect = w / float(h) #caso de falso positivo no verde, testa proporções (improvavel)
                #if 0.7 < aspect < 1.3:

                rectv = cv2.minAreaRect(c)
                (cxv, cyv) = rectv[0]
                cxv, cyv = int(cxv), int(cyv)
                centros_verdes.append((cxv, cyv))

                #if min(w, h) > 30:   # evita ruído #caso haja ruido na imagem (improvável)
                #    centros_verdes.append((int(cxv), int(cyv)))

                boxv = cv2.boxPoints(rectv)
                boxv = np.int32(boxv)

                cv2.drawContours(frame, [boxv], 0, (0,255,0), 2)
                cv2.circle(frame, (cxv, cyv), 3, (0,255,0), -1)

        if len(centros_verdes) == 1:
            cx_unico = centros_verdes[0][0]
            if cx_unico > cl: #testa em qual lado o verde está em relação à linha
                dir = "Esquerda"
            elif cx_unico < cl: 
                dir = "Direita"
        if len(centros_verdes) == 2:
            dir = "Dois verdes"

        #cv2.putText(frame, f"Verdes: {len(centros_verdes)}", (20,90),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0), 1)
        cv2.putText(frame, f"Verde: {dir}", (20,90),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0), 1)
    
    #DISPLAY/ESC
    cv2.imshow("Frame", frame) #Abre a visualização de 3 janelas (normal, flitro preto, filtro verde))
    cv2.imshow("Mask Black", mask_black)
    cv2.imshow("Mask Green", mask_green)
    if cv2.waitKey(1) == 27: #esc para fechar as janelas de visualização
        break

cap.release()
cv2.destroyAllWindows()