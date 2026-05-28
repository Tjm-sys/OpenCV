import cv2 # type: ignore
import numpy as np # type: ignore
from ev3dev2.motor import LargeMotor, MediumMotor, SpeedDPS, OUTPUT_A, OUTPUT_B, OUTPUT_C, OUTPUT_D # type: ignore
import time

cap = cv2.VideoCapture(0) #indica a porta da cam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #regula a resolucao (qualidade da imagem)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #codigo mais rapido assim
kernel = np.ones((4,4), np.uint8)

tempo_ant = time.time() #para o dt
erro_pos_ant = 0 #variavel para o calculo do derivativo do erro de posicao

#variaveis motores --
motor_dir = LargeMotor(OUTPUT_A)  # type: ignore
motor_esq = LargeMotor(OUTPUT_B)  # type: ignore
motor_fr_br = MediumMotor(OUTPUT_C)  # type: ignore
motor_fr_gr = MediumMotor(OUTPUT_D)  # type: ignore
motor_dir_mov = 0
motor_esq_mov = 0
motor_tr = 0
vel_base = 70 #AJUSTAR

#variaveis circles --
circle_ant = None #cria variaveis para prioridade da esfera
center_cal = None
stable_count = 100
stable_target = None
stable_frames = 50 #AJUSTAR 'tempo' para verificar as vitimas
tolerance = 15 #AJUSTAR, 'sensibilidade' para verificar as vitimas
dist = lambda x1, y1, x2, y2: (x1 - x2)**2 + (y1 - y2)**2
init = time.perf_counter()
frames_sem_linha = 0 #variavel para a interssecao
vitimas = 0

#masks vitimas --
silver_low = np.array([0, 0, 170]) #paranetros da mask prata
silver_high = np.array([180, 40, 255])
black_low_rsg = np.array([0, 0, 0]) #paranetros da mask preta
black_high_rsg = np.array([180, 255, 60])

#masks linha --
black_low_line = np.array([0, 0, 0]) #cofig do filtro/mask, verde/preto 
black_high_line = np.array([180, 255, 60])
green_low = np.array([40, 70, 70])
green_high = np.array([80, 255, 255])

#K´s = constante que define o quanto que reajimos ao erro #P=presente, D=futuro, I=passado
Kp_pos = 0.5 #AJUSTAR+: -se tremer, +se demorar para reajir: regula o centro. #Kp = Proporcional
Kp_ang = 2.5 #AJUSTAR: +se sair da linha em curvas: regula o angulo. #Kp = Proporcional
Kd = 0.02 #AJUSTAR: -se demorar para reajir Kd = Derivativo         
erro_pos_flit = 0 #variaveis para o calculo do filtro do Kp
erro_ang_flit = 0
alpha_pid = 0.2   #AJUSTAR, quanto menor mais 'suave'
alpha_cir = 0.7
alpha_derivativo = 0.2 #AJUSTAR, quanto menor mais
derivativo_filt = 0 #filtro do Kd
correcao_ant = 0 #variavel para guardar a ultima correcao aplicada, caso a linha nao seja detectada
cruzamento = False
rg = 0

def motor_fr_pg(motor_fr_br, motor_fr_gr):
    motor_fr_br.on_for_seconds(SpeedDPS(200), 1.5) # type: ignore
    motor_fr_gr.on_for_seconds(SpeedDPS(90), 1.5) # type: ignore
    motor_fr_br.on_for_seconds(SpeedDPS(-200), 1) # type: 
    
def motors_stop():
    motor_dir.stop()
    motor_esq.stop()


def roi(frame, linha=True): #ROI
    if linha:
        altura, largura = frame.shape[:2]
        return frame[int(altura*0.6):altura, 0:largura] #metade inferior
    else:
        return frame #frame completo
    
while True:
    ret, frame = cap.read() #verifica a leitura da cam
    if not ret:
        break

    if rg == 0:
        frame_process, offset_y = roi(frame, modo_linha=True)
    else:
        frame_process, offset_y = roi(frame, modo_linha=False)
    
    #calculos do Dt / timer --
    timer = int(time.perf_counter() - init) #timer cal
    tempo_atual = time.time()
    dt = tempo_atual - tempo_ant
    tempo_ant = tempo_atual
    if dt <= 0:
        dt = 0.0001

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #converte imagem para HSV (tonalidade, saturacao, valor)
    #cofig do filtro/mask preto/verde linha --
    mask_black_line = cv2.inRange(hsv, black_low_line, black_high_line)
    mask_green = cv2.inRange(hsv, green_low, green_high)
    #kernel = np.ones((4,4), np.uint8) #para o mapa visual
    #mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
    #mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel)    

    #config da mask/filtro prata --
    silver_mask = cv2.inRange(hsv, silver_low, silver_high) #cria mask/filtro prata
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #mask/filtro para regular brilo (-falsos positivos)
    bright_mask = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,-5)
    silver_mask = cv2.bitwise_and(silver_mask, bright_mask) #combina masks/filtros
    silver_mask = cv2.morphologyEx(silver_mask, cv2.MORPH_OPEN, kernel)
    silver_mask = cv2.morphologyEx(silver_mask, cv2.MORPH_CLOSE, kernel)
    silver_blur = cv2.GaussianBlur(silver_mask, (9, 9), 2)

    #config da mask/filtro preto --
    black_mask = cv2.inRange(hsv, black_low_rsg, black_high_rsg)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    black_blur = cv2.GaussianBlur(black_mask, (9, 9), 2)

    #parametros dos circulos -- #AJUSTAR
    black_circles = cv2.HoughCircles(black_blur,cv2.HOUGH_GRADIENT,dp=1.2, #morta
        minDist=80,
        param1=100,param2=18,minRadius=20,maxRadius=60)
    silver_circles = cv2.HoughCircles(silver_blur,cv2.HOUGH_GRADIENT,dp=1.2, #viva
        minDist=80,
        param1=100,param2=18,minRadius=20,maxRadius=60)
    detections = [(black_circles, (255, 0, 0)), (silver_circles, (0, 255, 0))]

    if rg == 1: #se o robo ja chegou na area de resgate
        if detections is None:
            motor_esq_mov = vel_base * 0.4
            motor_dir_mov = -vel_base * 0.4
            motor_esq.on(SpeedDPS(vel_base * 0.4))
            motor_dir.on(SpeedDPS(-vel_base * 0.4))
            continue

        for circles, color in detections:
            if circles is None:
                motor_esq_mov = vel_base * 0.4
                motor_dir_mov = -vel_base * 0.4
                motor_esq.on(SpeedDPS(vel_base * 0.4))
                motor_dir.on(SpeedDPS(-vel_base * 0.4))
                continue

            circles = np.uint16(np.around(circles))

            #escolhe o maior circulo
            chosen = max(circles[0, :], key=lambda c: c[2])
            current_center = (chosen[0], chosen[1])

            #contador de frames
            if stable_target is None:
                stable_target = current_center
                stable_count = 0
            else:
                d = np.sqrt(dist(current_center[0], current_center[1],stable_target[0], stable_target[1]))

                if d < tolerance:
                    stable_count += 1
                else:
                    stable_target = current_center
                    stable_count = 0

            if center_cal is None:
                center_cal = [chosen[0], chosen[1]]
            else:
                center_cal[0] = int(alpha_cir * center_cal[0] + (1 - alpha_cir) * chosen[0])
                center_cal[1] = int(alpha_cir * center_cal[1] + (1 - alpha_cir) * chosen[1])
            circle_ant = chosen

            cv2.circle(frame, (center_cal[0], center_cal[1]), 1, color, 3)
            cv2.circle(frame, (center_cal[0], center_cal[1]), chosen[2], color, 3)

            if stable_count >= stable_frames:
                cv2.putText(frame,"Vitima estabilizada",(50, 50),cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 0),2)
                #calculo movimento até a vitima
                center_frame = frame.shape[1] // 2 #centro do frame

                if chosen[2] <= 50: #AJUSTAR
                    erro_cen = center_cal[0] - center_frame #erro
                    kvp = 0.3 #AJUSTAR
                    erro_vit = erro_cen * kvp

                    motor_esq_mov = vel_base - erro_vit
                    motor_dir_mov = vel_base + erro_vit
                    motor_esq_mov = max(min(motor_esq_mov, 100), -100)
                    motor_dir_mov = max(min(motor_dir_mov, 100), -100)
                    motor_esq.on(SpeedDPS(motor_esq_mov)) # type: ignore
                    motor_dir.on(SpeedDPS(motor_dir_mov)) # type: ignore

                else: 
                    motor_fr_pg(motor_fr_br, motor_fr_gr) #pega vitima
                    time.sleep(1.0) #em segs
                    vitimas += 1 #adiciona ao contador
                    stable_count = 0
                    stable_target = None
                    center_cal = None  

        if vitimas >= 5 or timer >= 100: #numero de vitimas antes de sair do resgate
            rg=0
            #sair da area de resgate
        
    elif rg == 0: #robo está fora do resgate
        #LINHA
        cl = frame.shape[1] // 2  # padrão = centro do frame
        contours_black, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #funcao da biblioteca CV para contornos apos o filtro da linha
        if contours_black:
            c = max(contours_black, key=cv2.contourArea)

            if cv2.contourArea(c) > 500:
                rect = cv2.minAreaRect(c) #funcao cv, retorna centro X, Y, largura, altura e angulo
                angle = rect[2] #angulo da linha

                if angle < -45: #reajusta o angulo
                    angle = 90 + angle

                #CONTROLE
                centro_frame = frame.shape[1] // 2 #centro do frame
                cl = int(rect[0][0]) #centro da linha

                erro_pos = cl - centro_frame

                derivativo = (erro_pos - erro_pos_ant) / dt
                derivativo_filt = (1 - alpha_derivativo) * derivativo_filt + alpha_derivativo * derivativo

                erro_pos_flit = (1 - alpha_pid) * erro_pos_flit + alpha_pid * erro_pos #'filtro' deixa o movimento mais continuo
                erro_ang_flit = (1 - alpha_pid) * erro_ang_flit + alpha_pid * angle 

                #calculo da correcao com PD e filtro
                correcao = (
                    Kp_pos * erro_pos_flit +
                    Kp_ang * erro_ang_flit +
                    Kd * derivativo_filt)

                correcao = max(min(correcao, 100), -100) #**limita a correcao
                erro_pos_ant = erro_pos #salva o erro anterior
                correcao_ant = correcao #salva a correcao anterior
                frames_sem_linha = 0 #reseta os frames com gap
                
                #MOTORES: esperar para poder testar
                #Vmax = 140 #limite maximo e minimo da vel
                #Vmin = 70
                #Kvel = 1.2
                #vel_base = Vmax - Kvel * abs(erro_ang_flit) #calculo da velocidade
                #vel_base = max(Vmin, min(Vmax, vel_base)) #aplica os limites de velocidade

                vel_base = 120 #temporario
                motor_esq_mov = vel_base - correcao #aplica a correcao
                motor_dir_mov = vel_base + correcao

                #CRUZAMENTO: logica para detectar
                (w, h) = rect[1]

                if w > 0 and h > 0:
                    radio = max(w, h) / min(w, h)

                    if radio < 2:
                        cruzamento = True
                    else:
                        cruzamento = False

                #DISPLAY DE DADOS
                centro_frame = frame.shape[1] // 2
                altura = frame.shape[0]
                cv2.line(frame, (centro_frame, 0), (centro_frame, altura), (0,0,0), 1)

                cv2.putText(frame, f"Erro Pos: {erro_pos}", (20,30),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0), 1)  #visualizar o angulo e a diferenca entre a linha e o centro
                cv2.putText(frame, f"Erro Ang: {round(angle,1)}", (20,60),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0), 1)
                cv2.putText(frame, f"Cruzamento: {cruzamento}", (20,90),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0), 1)
                cv2.putText(frame, f"Correcao: {correcao:.2f}", (20,180),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0), 1)
                #cv2.putText(frame, f"Correcao: {correcao}", (20,120),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0), 1)

                box = cv2.boxPoints(rect) #define propriedades da "box" (borda da linha)
                box = np.int32(box)

                cx = int(rect[0][0]) #define propriedades do circulo (centro da linha)
                cy = int(rect[0][1])

                cv2.circle(frame, (cx, cy), 2, (0,0,255), -1)
                cv2.drawContours(frame, [box], 0, (0,0,255), 2) #ilustra o contortno da linha e escreve a legenda
                cv2.putText(frame, "Linha", (box[0][0], box[0][1]-10),cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 2)
            else:
                correcao = correcao_ant #se a linha nao for detectada, mantem a ultima correcao aplicada
                cv2.putText(frame, "Linha: Nao detectada", (20,30),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0), 1)

                vel_base = 120 #temporario
                motor_esq_mov = vel_base - correcao #aplica a correcao
                motor_dir_mov = vel_base + correcao
        #VERDE
        centros_verdes = [] #lista para identificar 2 verdes
        dir = ""
        anglev = 0 #angulo do verde
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #funcao da biblioteca CV para contornos apos o filtro do verde
        if contours_green:
            for c in contours_green:
                area = cv2.contourArea(c)

                if area > 500: #AJUSTAR a area/tamanho min do verde
                    approx = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)

                    #aspect = w / float(h) #caso de falso positivo no verde, testa proporcoes (improvavel)
                    #if 0.7 < aspect < 1.3:

                    rectv = cv2.minAreaRect(c)
                    (cxv, cyv) = rectv[0]
                    cxv, cyv = int(cxv), int(cyv)
                    centros_verdes.append((cxv, cyv))
                    anglev = rectv[2] #angulo do verde

                    boxv = cv2.boxPoints(rectv)
                    boxv = np.int32(boxv)

                    cv2.drawContours(frame, [boxv], 0, (0,255,0), 2)
                    cv2.circle(frame, (cxv, cyv), 3, (0,255,0), -1)

            if len(centros_verdes) == 1:
                cx_unico = centros_verdes[0][0]
                if cx_unico > cl: #testa em qual lado o verde esta em relacao a linha
                    if cruzamento == True: #logica para anular verdes caso haja intersecao
                        dir = "Esquerda anulado"
                    elif cruzamento == False: 
                        dir = "Esquerda normal"
                        motor_dir.on_for_seconds(SpeedDPS(-200), 1) # type: ignore
                        motor_esq.on_for_seconds(SpeedDPS(0), 1) # type: ignore

                elif cx_unico < cl: 
                    if cruzamento == True: #logica para anular verdes caso haja intersecao
                        dir = "Direita anulado"
                    elif cruzamento == False: 
                        dir = "Direita normal"
                        motor_esq.on_for_seconds(SpeedDPS(-200), 1) # type: ignore
                        motor_dir.on_for_seconds(SpeedDPS(0), 1) # type: ignore

            if len(centros_verdes) == 2:
                if cruzamento == True: #logica para anular dois verdes caso haja intersecao
                    dir = "Dois anulado"
                elif cruzamento == False: 
                    dir = "Dois verdes"

            #cv2.putText(frame, f"Verdes: {len(centros_verdes)}", (20,90),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0), 1)
            cv2.putText(frame, f"Verde: {dir}", (20,120),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0), 1)
            cv2.putText(frame, f"Angulo Verde: {anglev:.1f}", (20,150),cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0), 1)

    cv2.imshow("Frame", frame) #display do frame
    cv2.imshow("Prata", silver_mask) 
    cv2.imshow("Preto", black_mask)

    if cv2.waitKey(1) == 27: break #esc para fechar as janelas de visualizacao
cap.release()
cv2.destroyAllWindows()
