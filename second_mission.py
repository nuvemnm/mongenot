import cv2
import csv
import sys
import numpy as np

# --- CONFIGURAÇÕES ---
VIDEO_PATH = "video.mp4"
OUTPUT_PATH = "second_output.mp4"
CSV_FILENAME = "leituras.csv" 
LARGURA_FINAL = 1280
ALTURA_FINAL = 720

# Configurações do Rastreador de Barco
RING_INNER = 5   
RING_OUTER = 25  

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

def converter_posicao(cx, cy):
    # Ajuste aqui conforme seu cenário real
    lat = np.interp(cx, [0, 1280], [-180, 180])
    long = np.interp(cy, [0, 720], [90, -90])
    return lat, long

def calibrar_azul_predominante(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_sat = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([180, 255, 255]))
    hist = cv2.calcHist([hsv], [0], mask_sat, [180], [0, 180])
    pico_h = np.argmax(hist)
    lower = np.array([max(0, pico_h - 20), 60, 60])
    upper = np.array([min(179, pico_h + 20), 255, 255])
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

def calcular_limite_dinamico(v_agua_medio):
    #xp = [210, 220, 225, 230, 240]  
    #fp = [0.9, 0.8, 0.62, 0.3, 0.0] 
    fator = np.interp(v_agua_medio, [225, 235], [0.6, 0.35])
        
    limite = v_agua_medio * fator
    return int(limite), fator

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        return
    
    video_start_time = float(sys.argv[1])
    frames = 0

    arquivo_csv = open(CSV_FILENAME, mode='w', newline='')
    writer = csv.writer(arquivo_csv, delimiter=';') 
    writer.writerow(['frame', 'lat', 'long'])

    ret, frame_inicial = cap.read()
    if not ret: return

    lower_blue, upper_blue = calibrar_azul_predominante(frame_inicial)
    pontos_tanque = encontrar_retangulo_tanque(frame_inicial, lower_blue, upper_blue)

    if pontos_tanque is None:
        return

    pts_destino = np.array([[0, 0], [LARGURA_FINAL - 1, 0], [LARGURA_FINAL - 1, ALTURA_FINAL - 1], [0, ALTURA_FINAL - 1]], dtype="float32")
    matriz_warp = cv2.getPerspectiveTransform(pontos_tanque, pts_destino)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30, (LARGURA_FINAL, ALTURA_FINAL))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- VARIÁVEIS DE ESTADO ---
    barco_travado = False
    lower_barco = np.array([0, 0, 0])
    upper_barco = np.array([0, 0, 0])
    posicao_barco = None 

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. WARP
        warp = cv2.warpPerspective(frame, matriz_warp, (LARGURA_FINAL, ALTURA_FINAL))
        hsv_warp = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)

        # Inicializa máscaras de debug vazias (Pretas) para este frame
        mask_anel_debug = np.zeros((ALTURA_FINAL, LARGURA_FINAL), dtype=np.uint8)
        mask_barco_debug = np.zeros((ALTURA_FINAL, LARGURA_FINAL), dtype=np.uint8)

        # 2. INTELIGÊNCIA DA ÁGUA & ÓLEO
        mask_agua_ref = cv2.inRange(hsv_warp, lower_blue, upper_blue)
        pixels_agua_v = hsv_warp[:, :, 2][mask_agua_ref > 0]

        if len(pixels_agua_v) > 0:
            v_medio = np.median(pixels_agua_v)
            limite_v, fator_atual = calcular_limite_dinamico(v_medio)
        else:
            v_medio = 0; limite_v = 50

        mask_oleo = cv2.inRange(hsv_warp, np.array([0, 0, 0]), np.array([180, 255, limite_v]))
        kernel = np.ones((5, 5), np.uint8)
        mask_oleo = cv2.morphologyEx(mask_oleo, cv2.MORPH_OPEN, kernel)

        cnts_oleo, _ = cv2.findContours(mask_oleo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maior_mancha = None
        if cnts_oleo:
            maior_mancha = max(cnts_oleo, key=cv2.contourArea)
            if cv2.contourArea(maior_mancha) < 50:
                maior_mancha = None

        # --- LÓGICA DO BARCO ---
        
        # FASE A: DESCOBERTA
        if not barco_travado and maior_mancha is not None:
            mask_ancora = np.zeros_like(mask_oleo)
            cv2.drawContours(mask_ancora, [maior_mancha], -1, 255, -1)
            
            dilatada = cv2.dilate(mask_ancora, np.ones((RING_OUTER, RING_OUTER), np.uint8))
            miolo = cv2.dilate(mask_ancora, np.ones((RING_INNER, RING_INNER), np.uint8))
            anel_busca = cv2.subtract(dilatada, miolo)
            
            # Salva o anel para o Debug
            mask_anel_debug = anel_busca.copy() 

            anel_sem_agua = cv2.bitwise_and(anel_busca, anel_busca, mask=cv2.bitwise_not(mask_agua_ref))

            h_vals = hsv_warp[:,:,0][anel_sem_agua > 0]
            s_vals = hsv_warp[:,:,1][anel_sem_agua > 0]
            v_vals = hsv_warp[:,:,2][anel_sem_agua > 0]

            if len(h_vals) > 20:
                h_barco = int(np.median(h_vals))
                s_barco = int(np.median(s_vals))
                v_barco = int(np.median(v_vals))

                lower_barco = np.array([max(0, h_barco - 5), max(0, s_barco - 20), max(0, v_barco - 20)])
                upper_barco = np.array([min(179, h_barco + 5), 255, 255])
                
                barco_travado = True

        # FASE B: RASTREAMENTO
        if barco_travado:
            mask_barco = cv2.inRange(hsv_warp, lower_barco, upper_barco)
            mask_barco = cv2.morphologyEx(mask_barco, cv2.MORPH_OPEN, kernel)
            
            # Salva para o Debug
            mask_barco_debug = mask_barco.copy()

            cnts_barco, _ = cv2.findContours(mask_barco, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts_barco:
                c_barco = max(cnts_barco, key=cv2.contourArea)
                M = cv2.moments(c_barco)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    posicao_barco = (cx, cy)
                    
                    cv2.drawContours(warp, [c_barco], -1, (0, 255, 0), 2)
                    cv2.circle(warp, (cx, cy), 5, (0, 255, 255), -1)
                    cv2.putText(warp, f"Barco", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- DRAW FINAL ---
        for c in cnts_oleo:
            if cv2.contourArea(c) > 20:
                cv2.drawContours(warp, [c], -1, (0, 0, 255), 2)

        status_barco = "BUSCANDO..." if not barco_travado else "RASTREANDO"
        cv2.putText(warp, f"Status: {status_barco}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if posicao_barco:
             cv2.putText(warp, f"Pos: {posicao_barco}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if not barco_travado and maior_mancha is not None:
             cv2.drawContours(warp, [maior_mancha], -1, (255, 0, 255), 3)

        # ===============================================
        #    CRIAÇÃO DO PAINEL DE DEBUG (Mosaico)
        # ===============================================
        
        # Converte as máscaras (grayscale) para BGR para poder escrever texto colorido e juntar
        debug_agua = cv2.cvtColor(mask_agua_ref, cv2.COLOR_GRAY2BGR)
        debug_oleo = cv2.cvtColor(mask_oleo, cv2.COLOR_GRAY2BGR)
        debug_anel = cv2.cvtColor(mask_anel_debug, cv2.COLOR_GRAY2BGR)
        debug_barco = cv2.cvtColor(mask_barco_debug, cv2.COLOR_GRAY2BGR)

        # Adiciona rótulos para você saber qual é qual
        cv2.putText(debug_agua, "MASCARA AGUA", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(debug_oleo, "MASCARA OLEO (ADAPTATIVO)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(debug_anel, "ANEL DE BUSCA (DISCOVERY)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(debug_barco, "RASTREAMENTO BARCO", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Empilha as imagens (2 em cima, 2 embaixo)
        top_row = np.hstack([debug_agua, debug_oleo])
        bottom_row = np.hstack([debug_anel, debug_barco])
        debug_grid = np.vstack([top_row, bottom_row])

        # Redimensiona o painel de debug para caber na tela (50% do tamanho)
        scale_percent = 50 
        width = int(debug_grid.shape[1] * scale_percent / 100)
        height = int(debug_grid.shape[0] * scale_percent / 100)
        debug_grid_resized = cv2.resize(debug_grid, (width, height), interpolation=cv2.INTER_AREA)

        out.write(warp)
        
        # Mostra as janelas
        cv2.imshow("Monitoramento Completo", warp)
        cv2.imshow("Debug Mascaras", debug_grid_resized) # <--- AQUI ESTÁ SUA NOVA TELA
        
        if posicao_barco is not None:
            Cx, Cy = posicao_barco
            lat, long = converter_posicao(Cx, Cy)

            writer.writerow([frames, f"{lat:.6f}", f"{long:.6f}"])
        else:
            writer.writerow([frames, "0", "0"])
        frames += 1

        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    tempo_decorrido_video = frames / 30
    tempo_absoluto_evento = video_start_time + tempo_decorrido_video

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(tempo_absoluto_evento)

if __name__ == "__main__":
    main()