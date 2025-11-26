import cv2
import csv
import math
import time
import sys
import numpy as np

# --- CONFIGURAÇÕES ---
VIDEO_PATH = "camera_rb.mp4" # Se preferir, troque para original.mp4
OUTPUT_PATH = "video_tracking_cor_v3.mp4"
CSV_FILENAME = "leituras.csv" 
LARGURA_FINAL = 1280
ALTURA_FINAL = 720
AREA_MINIMA_OBJETO = 50 
LIMITE_DELTA_STD = 50.0
COR_MANCHA_TRAVADA = None

# Dicionário para guardar a "memória" dos barcos
historico_objetos = {}
proximo_id = 0

def calcular_distancia(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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

# Verifica se a cor atual bate com a assinatura
def cores_sao_parecidas(cor_atual, cor_assinatura, tol_h=15, tol_s=50, tol_v=50):
    if cor_assinatura is None: return False
    
    h1, s1, v1 = cor_atual
    h2, s2, v2 = cor_assinatura
    
    diff_h = abs(h1 - h2)
    diff_s = abs(s1 - s2)
    diff_v = abs(v1 - v2)
    
    return (diff_h < tol_h) and (diff_s < tol_s) and (diff_v < tol_v)

def range_hsv(h, s, v, rg_h, rg_s, rg_v):
    tol_h = rg_h  
    tol_s = rg_s  
    tol_v = rg_v  
    
    lower = np.array([max(0, h - tol_h), max(20, s - tol_s), max(20, v - tol_v)])
    upper = np.array([min(179, h + tol_h), min(255, s + tol_s), min(255, v + tol_v)])
    return lower, upper

def calibrar_azul(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_valida = cv2.inRange(hsv, np.array([0, 20, 20]), np.array([180, 255, 255]))
    
    hist_h = cv2.calcHist([hsv], [0], mask_valida, [180], [0, 180])
    pico_h = np.argmax(hist_h)
    hist_s = cv2.calcHist([hsv], [1], mask_valida, [256], [0, 256])
    pico_s = np.argmax(hist_s)
    hist_v = cv2.calcHist([hsv], [2], mask_valida, [256], [0, 256])
    pico_v = np.argmax(hist_v)
    
    lower, upper = range_hsv(pico_h, pico_s, pico_v, 15, 40, 40)
    
    return lower, upper, (pico_h, pico_s, pico_v)

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

def main():
    global proximo_id, historico_objetos, COR_MANCHA_TRAVADA

    video_start_time = float(sys.argv[1])
    frames = 0

    arquivo_csv = open(CSV_FILENAME, mode='w', newline='')
    writer = csv.writer(arquivo_csv, delimiter=';') 
    writer.writerow(['id', 'area', 'frame', 'lat', 'long'])

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): return

    ret, frame_inicial = cap.read()
    if not ret: return

    lower_blue, upper_blue, picos_fundo = calibrar_azul(frame_inicial)
    pontos_tanque = encontrar_retangulo_tanque(frame_inicial, lower_blue, upper_blue)

    if pontos_tanque is None:
        return

    pts_destino = np.array([[0, 0], [LARGURA_FINAL - 1, 0], [LARGURA_FINAL - 1, ALTURA_FINAL - 1], [0, ALTURA_FINAL - 1]], dtype="float32")
    matriz_warp = cv2.getPerspectiveTransform(pontos_tanque, pts_destino)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30, (LARGURA_FINAL, ALTURA_FINAL))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    kernel = np.ones((5, 5), np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret: break

        warp = cv2.warpPerspective(frame, matriz_warp, (LARGURA_FINAL, ALTURA_FINAL))
        hsv_warp = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
        v_channel = hsv_warp[:, :, 2] 

        # 1. Segmentação
        mask_agua = cv2.inRange(hsv_warp, lower_blue, upper_blue)
        mask_objetos_bruta = cv2.bitwise_not(mask_agua)
        mask_objetos = cv2.morphologyEx(mask_objetos_bruta, cv2.MORPH_OPEN, kernel)
        
        cnts, _ = cv2.findContours(mask_objetos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objetos_frame_atual = []

        # 2. Extração
        for c in cnts:
            area = cv2.contourArea(c)
            if area > AREA_MINIMA_OBJETO:
                M = cv2.moments(c)
                if M["m00"] == 0: continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                ((cx_f, cy_f), radius_f) = cv2.minEnclosingCircle(c)
                radius = int(round(radius_f))
                x = max(0, int(round(cx_f - radius)))
                y = max(0, int(round(cy_f - radius)))
                w = 2 * radius; h = 2 * radius
                
                if x + w > LARGURA_FINAL: w = LARGURA_FINAL - x
                if y + h > ALTURA_FINAL: h = ALTURA_FINAL - y
                if w <= 0 or h <= 0: continue

                roi_hsv = hsv_warp[y:y+h, x:x+w]
                roi_v = roi_hsv[:, :, 2]
                
                mask_local = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask_local, [c], -1, 255, -1, offset=(-x, -y))
                
                # Para estatísticas gerais (ainda usamos a média bruta aqui para tracking rápido)
                mean, stddev = cv2.meanStdDev(roi_hsv, mask=mask_local)
                media_h, media_s, media_v = mean[0][0], mean[1][0], mean[2][0]
                std_v = stddev[2][0] 

                objetos_frame_atual.append({
                    'centro': (cX, cY),
                    'std': std_v,
                    'medias_cor': (media_h, media_s, media_v),
                    'roi_info': (x, y, w, h, roi_v, mask_local, roi_hsv)
                })

        # 3. Tracking
        novos_ids_detectados = []
        for obj in objetos_frame_atual:
            cx, cy = obj['centro']
            std_atual = obj['std']
            media_cor_atual = obj['medias_cor']
            x, y, w, h, roi_v, mask_local, roi_hsv = obj['roi_info']
            
            matched_id = None
            
            # Match Distância
            menor_distancia = 70.0 
            for obj_id, dados_hist in historico_objetos.items():
                hx, hy = dados_hist['centro']
                dist = calcular_distancia((cx, cy), (hx, hy))
                if dist < menor_distancia:
                    menor_distancia = dist
                    matched_id = obj_id
            
            # Match Cor (Recuperação)
            if matched_id is None:
                for obj_id, dados_hist in historico_objetos.items():
                    if dados_hist.get('culpado_confirmado', False) and 'assinatura_cor' in dados_hist:
                        assinatura = dados_hist['assinatura_cor']
                        # Só recupera se NÃO parecer água (redundância de segurança)
                        # Usamos tol_v=200 para ignorar sombras ao comparar com o FUNDO
                        if not cores_sao_parecidas(assinatura, picos_fundo, tol_h=5, tol_v=200):
                             if cores_sao_parecidas(media_cor_atual, assinatura, tol_h=10, tol_s=30, tol_v=30):
                                matched_id = obj_id
                                break

            if matched_id is None:
                matched_id = proximo_id
                historico_objetos[matched_id] = {
                    'id': matched_id,
                    'centro': (cx, cy),
                    'baseline_std': std_atual, 
                    'status_vazamento': False,
                    'culpado_confirmado': False,
                    'frames_tracked': 0
                }
                proximo_id += 1
            
            historico_objetos[matched_id]['centro'] = (cx, cy)
            historico_objetos[matched_id]['frames_tracked'] += 1
            
            # 4. Detecção de Vazamento Refinada
            baseline = historico_objetos[matched_id]['baseline_std']
            delta = abs(std_atual - baseline)
            is_leaking = (delta > LIMITE_DELTA_STD) or historico_objetos[matched_id]['status_vazamento']

            if is_leaking:
                historico_objetos[matched_id]['status_vazamento'] = True
                
                # Otsu para separar claro/escuro
                thresh_val, mask_otsu = cv2.threshold(roi_v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                mask_parte_escura = cv2.bitwise_and(cv2.bitwise_not(mask_otsu), mask_local)
                media_escura = cv2.mean(roi_v, mask=mask_parte_escura)[0]
                
                # --- NOVA LÓGICA DE EXTRAÇÃO DE COR DO BARCO ---
                # 1. Cria máscara básica do barco (parte clara do Otsu)
                mask_barco_cand = cv2.bitwise_and(mask_otsu, mask_local)
                
                # 2. Cria máscara de exclusão da ÁGUA localmente
                mask_agua_roi = cv2.inRange(roi_hsv, lower_blue, upper_blue)
                
                # 3. Subtrai a água da máscara do barco
                mask_barco_limpa = cv2.bitwise_and(mask_barco_cand, cv2.bitwise_not(mask_agua_roi))
                
                # Se sobrou algo que não é água, calculamos a média
                if cv2.countNonZero(mask_barco_limpa) > 10:
                    media_cor_barco_hsv = cv2.mean(roi_hsv, mask=mask_barco_limpa)
                    h_b, s_b, v_b = media_cor_barco_hsv[0], media_cor_barco_hsv[1], media_cor_barco_hsv[2]
                else:
                    # Se tudo era água, mantemos a média bruta (mas será filtrada abaixo)
                    h_b, s_b, v_b = media_cor_atual

                # --- VALIDAÇÃO CONTRA O FUNDO (CORREÇÃO DO PROBLEMA ID 65) ---
                if not historico_objetos[matched_id]['culpado_confirmado']:
                    
                    # Isso diz: "Se a cor for igual a da água, ignore, MESMO QUE seja muito escura (sombra)"
                    eh_agua_ou_sombra = cores_sao_parecidas((h_b, s_b, v_b), picos_fundo, tol_h=10, tol_s=40, tol_v=200)

                    if not eh_agua_ou_sombra:
                        # Se for diferente, aí sim confirmamos
                        historico_objetos[matched_id]['culpado_confirmado'] = True
                        historico_objetos[matched_id]['assinatura_cor'] = (h_b, s_b, v_b)

                if COR_MANCHA_TRAVADA is None and cv2.countNonZero(mask_parte_escura) > 10:
                    COR_MANCHA_TRAVADA = media_escura
                
                cv2.putText(warp, f"VAZANDO", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(warp, f"delta: {delta:.1f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            novos_ids_detectados.append(matched_id)
        
        ids_para_remover = [k for k in historico_objetos if k not in novos_ids_detectados]
        for k in ids_para_remover: del historico_objetos[k]
        
        area_total_oleo = 0
        if COR_MANCHA_TRAVADA is not None:
            mask_oleo = cv2.inRange(hsv_warp, np.array([0, 0, 0]), np.array([179, 255, int(COR_MANCHA_TRAVADA)]))
            mask_oleo = cv2.morphologyEx(mask_oleo, cv2.MORPH_OPEN, kernel)
            area_total_oleo = cv2.countNonZero(mask_oleo)
            cnts_oleo, _ = cv2.findContours(mask_oleo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(warp, cnts_oleo, -1, (0, 0, 255), -1)

        for dados_barco in historico_objetos.values():
            if dados_barco.get('culpado_confirmado', False):
                id_b = dados_barco['id']
                cx, cy = dados_barco['centro']
                lat, long = converter_posicao(cx, cy)
                #cv2.circle(warp, (cx, cy), 45, (0, 0, 255), 2)
                #cv2.putText(warp, f"CULPADO {id_b}", (cx-30, cy-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                writer.writerow([id_b, area_total_oleo, frames, f"{lat:.6f}", f"{long:.6f}"])

        cv2.putText(warp, f"AREA OLEO: {area_total_oleo}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        out.write(warp)
        cv2.imshow("Tracking Inteligente", warp)
        frames += 1
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    tempo_decorrido_video = frames / 30
    tempo_absoluto_evento = video_start_time + tempo_decorrido_video
            
    cap.release()
    out.release()
    arquivo_csv.close()
    cv2.destroyAllWindows()

    print(tempo_absoluto_evento)

if __name__ == "__main__":
    main()