import cv2 # type: ignore
import numpy as np # type: ignore
from ev3dev2.motor import LargeMotor, MediumMotor, SpeedDPS, OUTPUT_A, OUTPUT_B, OUTPUT_C, OUTPUT_D # type: ignore
import time
import traceback

# ============================================
# CONFIGURAÇÕES INICIAIS
# ============================================

# Câmera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Constantes do frame
CENTRO_FRAME = 320  # 640/2
ALTURA_FRAME = 480
ROI_LINHA = (240, 480, 0, 640)  # y1, y2, x1, x2

# Kernels (pré-alocados)
kernel_3x3 = np.ones((3,3), np.uint8)
kernel_4x4 = np.ones((4,4), np.uint8)

# ============================================
# CLASSES
# ============================================

class EstadoRobo:
    SEGUINDO_LINHA = 0
    AREA_RESGATE = 1
    RETORNANDO = 2

class PIDController:
    def __init__(self, kp, ki, kd, alpha=0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.alpha = alpha
        self.erro_anterior = 0
        self.integral = 0
        self.derivativo_filtrado = 0
        
    def calcular(self, erro, dt):
        # Proporcional
        p = self.kp * erro
        
        # Integral
        self.integral += erro * dt
        self.integral = max(-100, min(100, self.integral))  # Anti-windup
        i = self.ki * self.integral
        
        # Derivativo com filtro
        if dt > 0:
            derivativo = (erro - self.erro_anterior) / dt
            self.derivativo_filtrado = (1 - self.alpha) * self.derivativo_filtrado + \
                                       self.alpha * derivativo
        d = self.kd * self.derivativo_filtrado
        
        self.erro_anterior = erro
        return p + i + d
    
    def reset(self):
        self.erro_anterior = 0
        self.integral = 0
        self.derivativo_filtrado = 0

# ============================================
# MOTORES
# ============================================

motor_dir = LargeMotor(OUTPUT_A)
motor_esq = LargeMotor(OUTPUT_B)
motor_fr_br = MediumMotor(OUTPUT_C)
motor_fr_gr = MediumMotor(OUTPUT_D)

def parar_motores():
    """Para todos os motores de movimento"""
    motor_esq.stop()
    motor_dir.stop()

def mover_robo(vel_esq, vel_dir, duracao=None, block=False):
    """Movimenta o robô com velocidades específicas"""
    parar_motores()
    
    # Limita velocidades
    vel_esq = max(-100, min(100, vel_esq))
    vel_dir = max(-100, min(100, vel_dir))
    
    if duracao is not None:
        motor_esq.on_for_seconds(SpeedDPS(vel_esq), duracao, block=block)
        motor_dir.on_for_seconds(SpeedDPS(vel_dir), duracao, block=block)
    else:
        motor_esq.on(SpeedDPS(vel_esq))
        motor_dir.on(SpeedDPS(vel_dir))

def motor_fr_pg():
    """Aciona o motor frontal para pegar vítima"""
    motor_fr_br.on_for_seconds(SpeedDPS(200), 1.5, block=False)
    motor_fr_gr.on_for_seconds(SpeedDPS(90), 1.5, block=False)
    time.sleep(0.5)
    motor_fr_br.on_for_seconds(SpeedDPS(-200), 1.0)

# ============================================
# MÁSCARAS HSV
# ============================================

# Vítimas
silver_low = np.array([0, 0, 170])
silver_high = np.array([180, 40, 255])
black_low_rsg = np.array([0, 0, 0])
black_high_rsg = np.array([180, 255, 60])

# Linha
black_low_line = np.array([0, 0, 0])
black_high_line = np.array([180, 255, 60])
green_low = np.array([40, 70, 70])
green_high = np.array([80, 255, 255])

# ============================================
# CONTROLADORES PID
# ============================================

pid_pos = PIDController(kp=0.5, ki=0.0, kd=0.02, alpha=0.2)
pid_ang = PIDController(kp=2.5, ki=0.0, kd=0.0, alpha=0.2)

# ============================================
# VARIÁVEIS DE ESTADO
# ============================================

estado = EstadoRobo.SEGUINDO_LINHA

# Resgate
circle_ant = None
center_cal = None
stable_count = 0
stable_target = None
vitimas = 0
ultima_acao_verde = 0

# Linha
correcao_ant = 0
cruzamento = False
frames_sem_linha = 0

# Tempo
tempo_ant = time.time()
init = time.perf_counter()

# Constantes ajustáveis
STABLE_FRAMES = 50
TOLERANCE_VITIMA = 15
VEL_BASE = 70
VEL_BASE_RESGATE = 40
TIMEOUT_RESGATE = 100  # segundos
MAX_VITIMAS = 5
TIMEOUT_VERDE = 2.0  # segundos entre ações de verde

# ============================================
# FUNÇÕES AUXILIARES
# ============================================

dist = lambda x1, y1, x2, y2: (x1 - x2)**2 + (y1 - y2)**2

def aplicar_roi(frame, para_linha=True):
    """Aplica ROI para processamento mais rápido"""
    if para_linha:
        y1, y2, x1, x2 = ROI_LINHA
        return frame[y1:y2, x1:x2]
    return frame

def processar_mascaras(frame):
    """Processa todas as máscaras necessárias"""
    # Única conversão HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Máscaras de linha
    mask_black_line = cv2.inRange(hsv, black_low_line, black_high_line)
    mask_green = cv2.inRange(hsv, green_low, green_high)
    
    # Máscara prata (vítimas vivas)
    silver_mask = cv2.inRange(hsv, silver_low, silver_high)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, -5)
    silver_mask = cv2.bitwise_and(silver_mask, bright_mask)
    silver_mask = cv2.morphologyEx(silver_mask, cv2.MORPH_OPEN, kernel_3x3)
    silver_mask = cv2.morphologyEx(silver_mask, cv2.MORPH_CLOSE, kernel_3x3)
    silver_blur = cv2.GaussianBlur(silver_mask, (9, 9), 2)
    
    # Máscara preta (vítimas mortas)
    black_mask = cv2.inRange(hsv, black_low_rsg, black_high_rsg)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel_3x3)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_3x3)
    black_blur = cv2.GaussianBlur(black_mask, (9, 9), 2)
    
    return {
        'mask_black_line': mask_black_line,
        'mask_green': mask_green,
        'silver_blur': silver_blur,
        'black_blur': black_blur,
        'silver_mask': silver_mask,
        'black_mask': black_mask
    }

def detectar_circulos(black_blur, silver_blur):
    """Detecta círculos nas máscaras de vítimas"""
    black_circles = cv2.HoughCircles(black_blur, cv2.HOUGH_GRADIENT, dp=1.2,
                                     minDist=80, param1=100, param2=18,
                                     minRadius=20, maxRadius=60)
    silver_circles = cv2.HoughCircles(silver_blur, cv2.HOUGH_GRADIENT, dp=1.2,
                                      minDist=80, param1=100, param2=18,
                                      minRadius=20, maxRadius=60)
    
    detections = []
    if black_circles is not None:
        detections.append((black_circles, (255, 0, 0)))  # Azul para mortas
    if silver_circles is not None:
        detections.append((silver_circles, (0, 255, 0)))  # Verde para vivas
    
    return detections

def processar_resgate(frame, detections):
    """Processa o modo de resgate"""
    global center_cal, stable_target, stable_count, circle_ant
    global vitimas, estado, ultima_acao_verde
    
    if not detections:
        # Gira para procurar vítimas
        mover_robo(VEL_BASE_RESGATE * 0.3, -VEL_BASE_RESGATE * 0.3)
        return
    
    for circles, color in detections:
        circles = np.uint16(np.around(circles))
        
        if len(circles[0, :]) == 0:
            continue
            
        # Escolhe o maior círculo
        chosen = max(circles[0, :], key=lambda c: c[2])
        current_center = (chosen[0], chosen[1])
        
        # Contador de estabilidade
        if stable_target is None:
            stable_target = current_center
            stable_count = 0
        else:
            d = np.sqrt(dist(current_center[0], current_center[1],
                           stable_target[0], stable_target[1]))
            if d < TOLERANCE_VITIMA:
                stable_count += 1
            else:
                stable_target = current_center
                stable_count = 0
        
        # Filtro de posição
        if center_cal is None:
            center_cal = [chosen[0], chosen[1]]
        else:
            center_cal[0] = int(0.7 * center_cal[0] + 0.3 * chosen[0])
            center_cal[1] = int(0.7 * center_cal[1] + 0.3 * chosen[1])
        
        circle_ant = chosen
        
        # Desenha na tela
        cv2.circle(frame, (center_cal[0], center_cal[1]), 1, color, 3)
        cv2.circle(frame, (center_cal[0], center_cal[1]), chosen[2], color, 3)
        
        # Verifica se estabilizou
        if stable_count >= STABLE_FRAMES:
            cv2.putText(frame, "Vitima estabilizada", (50, 50),
                       cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            
            # Aproximação ou coleta
            if chosen[2] <= 50:  # Se está longe, aproxima
                erro_cen = center_cal[0] - CENTRO_FRAME
                ganho = 0.3
                vel_esq = VEL_BASE_RESGATE - erro_cen * ganho
                vel_dir = VEL_BASE_RESGATE + erro_cen * ganho
                mover_robo(vel_esq, vel_dir)
            else:  # Se está perto, coleta
                motor_fr_pg()
                time.sleep(0.5)
                vitimas += 1
                
                # Reset após coleta
                stable_count = 0
                stable_target = None
                center_cal = None
                
                # Verifica se deve sair do resgate
                if vitimas >= MAX_VITIMAS:
                    estado = EstadoRobo.RETORNANDO
        
        break  # Processa apenas o primeiro círculo válido

def processar_linha(frame, masks):
    """Processa o modo de seguir linha"""
    global correcao_ant, cruzamento, frames_sem_linha, ultima_acao_verde
    
    cl = CENTRO_FRAME
    altura = frame.shape[0]
    
    # Contornos da linha preta
    contours_black, _ = cv2.findContours(masks['mask_black_line'],
                                         cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    
    if contours_black:
        c = max(contours_black, key=cv2.contourArea)
        
        if cv2.contourArea(c) > 500:
            # Análise do contorno
            rect = cv2.minAreaRect(c)
            angle = rect[2]
            
            if angle < -45:
                angle = 90 + angle
            
            cl = int(rect[0][0])
            erro_pos = cl - CENTRO_FRAME
            
            # Calcula dt
            global tempo_ant
            tempo_atual = time.time()
            dt = tempo_atual - tempo_ant
            tempo_ant = tempo_atual
            if dt <= 0:
                dt = 0.0001
            
            # PID
            correcao_pos = pid_pos.calcular(erro_pos, dt)
            correcao_ang = pid_ang.calcular(angle, dt)
            correcao = correcao_pos + correcao_ang
            
            correcao = max(-100, min(100, correcao))
            correcao_ant = correcao
            frames_sem_linha = 0
            
            # Velocidade base adaptativa
            vel_base_adaptativa = VEL_BASE
            
            # Aplica correção
            vel_esq = vel_base_adaptativa - correcao
            vel_dir = vel_base_adaptativa + correcao
            mover_robo(vel_esq, vel_dir)
            
            # Detecção de cruzamento
            (w, h) = rect[1]
            if w > 0 and h > 0:
                razao = max(w, h) / min(w, h)
                cruzamento = razao < 2
            
            # Visualização
            cv2.line(frame, (CENTRO_FRAME, 0), (CENTRO_FRAME, altura), (0, 0, 0), 1)
            
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cx = int(rect[0][0])
            cy = int(rect[0][1])
            
            cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
            cv2.putText(frame, "Linha", (box[0][0], box[0][1] - 10),
                       cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
            
            # Exibe informações
            cv2.putText(frame, f"Erro Pos: {erro_pos}", (20, 30),
                       cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)
            cv2.putText(frame, f"Erro Ang: {round(angle, 1)}", (20, 60),
                       cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)
            cv2.putText(frame, f"Cruzamento: {cruzamento}", (20, 90),
                       cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)
            cv2.putText(frame, f"Correcao: {correcao:.2f}", (20, 180),
                       cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)
        else:
            # Contorno muito pequeno
            usar_correcao_anterior()
    else:
        # Nenhum contorno encontrado
        usar_correcao_anterior()
    
    # Processamento de verdes
    processar_verdes(frame, masks['mask_green'], cl)

def usar_correcao_anterior():
    """Usa a última correção quando a linha não é detectada"""
    global frames_sem_linha, correcao_ant
    
    frames_sem_linha += 1
    
    if frames_sem_linha < 30:  # Mantém correção por 30 frames
        vel_esq = VEL_BASE - correcao_ant
        vel_dir = VEL_BASE + correcao_ant
        mover_robo(vel_esq, vel_dir)
    else:
        # Gira para procurar linha
        mover_robo(VEL_BASE * 0.3, -VEL_BASE * 0.3)

def processar_verdes(frame, mask_green, cl):
    """Processa marcações verdes"""
    global ultima_acao_verde, cruzamento
    
    centros_verdes = []
    anglev = 0
    
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
    
    if contours_green:
        for c in contours_green:
            area = cv2.contourArea(c)
            
            if area > 500:
                rectv = cv2.minAreaRect(c)
                (cxv, cyv) = rectv[0]
                cxv, cyv = int(cxv), int(cyv)
                centros_verdes.append((cxv, cyv))
                anglev = rectv[2]
                
                boxv = cv2.boxPoints(rectv)
                boxv = np.int32(boxv)
                
                cv2.drawContours(frame, [boxv], 0, (0, 255, 0), 2)
                cv2.circle(frame, (cxv, cyv), 3, (0, 255, 0), -1)
        
        direcao = ""
        if len(centros_verdes) == 1:
            cx_unico = centros_verdes[0][0]
            
            if not cruzamento:  # Só age se não estiver em cruzamento
                tempo_atual = time.time()
                if tempo_atual - ultima_acao_verde > TIMEOUT_VERDE:
                    ultima_acao_verde = tempo_atual
                    
                    if cx_unico > cl:
                        direcao = "Esquerda"
                        parar_motores()
                        motor_dir.on_for_seconds(SpeedDPS(-200), 0.5, block=False)
                        motor_esq.on_for_seconds(SpeedDPS(0), 0.5)
                    else:
                        direcao = "Direita"
                        parar_motores()
                        motor_esq.on_for_seconds(SpeedDPS(-200), 0.5, block=False)
                        motor_dir.on_for_seconds(SpeedDPS(0), 0.5)
            else:
                direcao = "Anulado (cruzamento)"
        
        elif len(centros_verdes) == 2:
            direcao = "Dois verdes" if not cruzamento else "Dois anulado"
        
        if direcao:
            cv2.putText(frame, f"Verde: {direcao}", (20, 120),
                       cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(frame, f"Angulo Verde: {anglev:.1f}", (20, 150),
                   cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)

def exibir_status(frame, timer):
    """Exibe informações gerais de status"""
    dados = [
        f"Estado: {['Linha', 'Resgate', 'Retornando'][estado]}",
        f"Timer: {timer}s",
        f"Vitimas: {vitimas}/{MAX_VITIMAS}",
        f"Vel Base: {VEL_BASE}"
    ]
    
    y = frame.shape[0] - 20
    for dado in reversed(dados):
        cv2.putText(frame, dado, (10, y), cv2.FONT_HERSHEY_COMPLEX,
                   0.5, (255, 255, 255), 1)
        y -= 20

# ============================================
# LOOP PRINCIPAL
# ============================================

print("Iniciando robô...")
print("Pressione ESC para sair")

try:
    while True:
        # Leitura da câmera
        ret, frame = cap.read()
        if not ret:
            print("Erro na câmera, tentando reconectar...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            continue
        
        # Timer
        timer = int(time.perf_counter() - init)
        
        # Processamento de máscaras
        masks = processar_mascaras(frame)
        
        # Detecção de círculos (para resgate)
        detections = detectar_circulos(masks['black_blur'], masks['silver_blur'])
        
        # Máquina de estados
        if estado == EstadoRobo.AREA_RESGATE:
            processar_resgate(frame, detections)
            cv2.putText(frame, "MODO: RESGATE", (20, frame.shape[0] - 50),
                       cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            
            # Verifica timeout do resgate
            if timer >= TIMEOUT_RESGATE or vitimas >= MAX_VITIMAS:
                estado = EstadoRobo.RETORNANDO
                parar_motores()
                time.sleep(0.5)
                
        elif estado == EstadoRobo.SEGUINDO_LINHA:
            processar_linha(frame, masks)
            cv2.putText(frame, "MODO: LINHA", (20, frame.shape[0] - 50),
                       cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            
            # TODO: Adicionar detecção de área de resgate para mudar estado
            # if detectar_area_resgate(frame):
            #     estado = EstadoRobo.AREA_RESGATE
                
        elif estado == EstadoRobo.RETORNANDO:
            processar_linha(frame, masks)
            cv2.putText(frame, "MODO: RETORNANDO", (20, frame.shape[0] - 50),
                       cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
        
        # Status geral
        exibir_status(frame, timer)
        
        # Display
        cv2.imshow("Frame", frame)
        cv2.imshow("Prata", masks['silver_mask'])
        cv2.imshow("Preto", masks['black_mask'])
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):  # Força modo resgate
            estado = EstadoRobo.AREA_RESGATE
            pid_pos.reset()
            pid_ang.reset()
            print("Modo resgate ativado")
        elif key == ord('l'):  # Força modo linha
            estado = EstadoRobo.SEGUINDO_LINHA
            pid_pos.reset()
            pid_ang.reset()
            print("Modo linha ativado")

except KeyboardInterrupt:
    print("\nInterrompido pelo usuário")
except Exception as e:
    print(f"Erro inesperado: {e}")
    traceback.print_exc()
finally:
    # Limpeza
    print("Finalizando...")
    parar_motores()
    cap.release()
    cv2.destroyAllWindows()
    print("Pronto!")