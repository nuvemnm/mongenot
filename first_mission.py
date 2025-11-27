import cv2
import numpy as np

# --- CONFIGURAÇÕES ---
VIDEO_PATH = "video.mp4"
OUTPUT_PATH = "first_output.mp4"
LARGURA_FINAL = 1280
ALTURA_FINAL = 720

def ordenar_pontos(pts):
    pts = pts.reshape((4, 2))
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def calibrar_azul(frame):
    """
    Calibra H, S e V automaticamente baseado no que mais aparece na imagem.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 1. Pré-máscara para ignorar bordas pretas (V=0) e ruído muito cinza
    # Isso evita que a calibração pegue a borda preta como "cor da água"
    mask_valida = cv2.inRange(hsv, np.array([0, 20, 20]), np.array([180, 255, 255]))
    
    # --- CANAL H (Matiz/Cor) ---
    hist_h = cv2.calcHist([hsv], [0], mask_valida, [180], [0, 180])
    pico_h = np.argmax(hist_h)
    
    # --- CANAL S (Saturação) ---
    hist_s = cv2.calcHist([hsv], [1], mask_valida, [256], [0, 256])
    pico_s = np.argmax(hist_s)
    
    # --- CANAL V (Brilho) ---
    hist_v = cv2.calcHist([hsv], [2], mask_valida, [256], [0, 256])
    pico_v = np.argmax(hist_v)
    
    # Definição de Tolerâncias (Você pode ajustar isso)
    tol_h = 15  # Margem de cor
    tol_s = 40  # Margem de saturação (água pode variar bastante)
    tol_v = 40  # Margem de brilho (sombras vs luz direta)
    
    # Criação dos arrays de limite (com proteção para não sair de 0-255)
    lower = np.array([
        max(0, pico_h - tol_h),
        max(20, pico_s - tol_s), # Mínimo 20 para não pegar cinza total
        max(20, pico_v - tol_v)  # Mínimo 20 para não pegar preto total
    ])
    
    upper = np.array([
        min(179, pico_h + tol_h),
        min(255, pico_s + tol_s),
        min(255, pico_v + tol_v)
    ])
    
    return lower, upper

def encontrar_retangulo_tanque(frame, lower_blue, upper_blue):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    perimetro = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * perimetro, True)
    if len(approx) == 4:
        return ordenar_pontos(approx)
    else:
        rect = cv2.minAreaRect(c)
        return ordenar_pontos(cv2.boxPoints(rect))

# --- NOVA FUNÇÃO INTELIGENTE ---
def calcular_limite_dinamico(v_agua_medio):
    
    #xp = [210, 220, 225, 230, 240]  # Pontos de entrada (Brilho da Água - V)
    #fp = [0.9, 0.8, 0.62, 0.3, 0.0]  # Pontos de saída (Fator Multiplicador)

    fator = np.interp(v_agua_medio, [225, 235], [0.6, 0.35])
        
    limite = v_agua_medio * fator
    return int(limite), fator

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        return

    ret, frame_inicial = cap.read()
    if not ret: return

    lower_blue, upper_blue = calibrar_azul(frame_inicial)
    pontos_tanque = encontrar_retangulo_tanque(frame_inicial, lower_blue, upper_blue)

    if pontos_tanque is None:
        return

    pts_destino = np.array([[0, 0], [LARGURA_FINAL - 1, 0], [LARGURA_FINAL - 1, ALTURA_FINAL - 1], [0, ALTURA_FINAL - 1]], dtype="float32")
    matriz_warp = cv2.getPerspectiveTransform(pontos_tanque, pts_destino)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30, (LARGURA_FINAL, ALTURA_FINAL))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        warp = cv2.warpPerspective(frame, matriz_warp, (LARGURA_FINAL, ALTURA_FINAL))
        hsv_warp = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)

        # 1. Analisar Água
        mask_agua_ref = cv2.inRange(hsv_warp, lower_blue, upper_blue)
        pixels_agua_v = hsv_warp[:, :, 2][mask_agua_ref > 0]

        if len(pixels_agua_v) > 0:
            v_medio = np.median(pixels_agua_v)
            
            # --- APLICAÇÃO DA CORREÇÃO ---
            limite_v, fator_atual = calcular_limite_dinamico(v_medio)
            # -----------------------------
        else:
            v_medio = 0
            limite_v = 50 # Fallback seguro

        # 2. Detectar Óleo com o Limite Calculado
        # Note que o limite agora muda frame a frame dependendo da luz
        mask_oleo = cv2.inRange(hsv_warp, np.array([0, 0, 0]), np.array([179, 255, limite_v]))
        
        kernel = np.ones((5, 5), np.uint8)
        mask_oleo = cv2.morphologyEx(mask_oleo, cv2.MORPH_OPEN, kernel)
        # Mostrar máscara para debug
        cv2.imshow("Mascara Inteligente (Debug)", mask_oleo)
        # 3. Desenhar
        cnts, _ = cv2.findContours(mask_oleo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_total = 0
        for i, c in enumerate(cnts):
            area = cv2.contourArea(c)
            if area > 20:
                area_total += area
                cv2.drawContours(warp, [c], -1, (0, 0, 255), 2)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    cv2.putText(warp, f"#{i+1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Infos de Debug na tela (Crucial para você entender o que está acontecendo)
        texto_debug = f"Agua(V): {int(v_medio)} | Fator: {fator_atual:.2f} | Limite Cut: {limite_v}"
        cv2.putText(warp, texto_debug, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(warp, f"AREA TOTAL: {int(area_total)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        out.write(warp)
        cv2.imshow("Resultado Dinamico", warp)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(area_total)

if __name__ == "__main__":
    main()