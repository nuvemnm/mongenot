import serial
import threading
import queue
import subprocess
import os
import signal
import csv
from time import sleep
import pandas as pd
import time
from math import sqrt
from collections import Counter

# ---------------------------
# Configuração / Inicialização serial
# ---------------------------
SERIAL_PATH = '/dev/ttyS0'
BAUDRATE = 9600
while True:
    try:
        ser = serial.Serial(SERIAL_PATH, BAUDRATE, timeout=0)
        break
    except Exception:
        pass

data_receiver_queue = queue.Queue()
mission2_proc = None
processo_camera = None

# caminho do script de missão 2 (ajuste se necessário)
MISSao2_PATH = '/mnt/data/missao2.py'

# PATH do codigo do AIS


diretorio = '/home/pi/meu_projeto/dados'
#PATH do csv do codigo de AIS

nome_arquivo = 'leituras.csv'
#arquivo csv do AIS

caminho_completo = os.path.join(diretorio, nome_arquivo)
caminho_completo = os.path.join(diretorio, nome_arquivo)

df_video = pd.read_csv(caminho_completo, delimiter=';')
df_ais = pd.read_csv(caminho_completo, delimiter=';')


# ---------------------------
# Funções Auxiliares
# ---------------------------
def send_data_to_serial(data):
    ser.write(str(data).encode('utf-8'))

def save_data():
    while True:
        if not data_receiver_queue.empty():
            command = data_receiver_queue.get()
            try:
                with open('dados.txt', 'a') as file:
                    file.write(command + '')
            except Exception:
                pass
        else:
            sleep(0.1)

save_thread = threading.Thread(target=save_data, daemon=True)
save_thread.start()

def start_mission2_subprocess(timestamp_inicio):
    global mission2_proc
    if mission2_proc is not None:
        return False
    if not os.path.exists(MISSao2_PATH):
        send_data_to_serial('ERR_M2_NOFILE')
        return False
    try:    
        # Passa o timestamp de inicio como argumento para o script de processamento
        mission2_proc = subprocess.Popen(['python3', MISSao2_PATH, str(timestamp_inicio)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        saida_texto, erros = mission2_proc.communicate()
        return saida_texto
    except Exception:
        mission2_proc = None
        send_data_to_serial('ERR_M2_START')
        return False

def iniciar_gravacao():
    global processo_camera
    if processo_camera is not None: return

    nome_arquivo = f"video.mp4"
    # IMPORTANTE: O FPS aqui (30) deve bater com o cálculo de tempo depois
    comando = [
        "rpicam-vid", "-t", "0", "-o", nome_arquivo,
        "--width", "1920", "--height", "1080", "--framerate", "30"
    ]
    try:
        processo_camera = subprocess.Popen(comando, preexec_fn=os.setsid)
    except Exception as e:
        processo_camera = None

def parar_gravacao():
    global processo_camera
    if processo_camera is None: return
    os.killpg(os.getpgid(processo_camera.pid), signal.SIGINT)
    processo_camera.wait()
    processo_camera = None

def calcular_diferenca_graus(lon1, lat1, lon2, lat2):
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    return sqrt(dlat**2 + dlon**2)

# ---------------------------
# Lógica de Correlação (Interpolada por Frame)
# ---------------------------
def encontrar_mmsi(timestamp_inicio_video, df_video, df_ais):
    candidatos_mmsi = []
    FPS = 30.0 # Deve ser o mesmo usado na gravação

    # Verifica se as colunas necessárias existem
    # Ajuste os nomes abaixo conforme o cabeçalho real dos seus CSVs
    if 'frame' not in df_video.columns:
        print("Erro: Coluna 'frame' não encontrada no CSV do vídeo.")
        return "Erro CSV Video"

    # 1. Iterar por linhas do vídeo (Amostragem ::10 para performance)
    for index, row in df_video.iterrows():
        try:
            # --- CÁLCULO DE INTERPOLAÇÃO DE TEMPO ---
            frame_atual = float(row['frame'])
            tempo_decorrido_segundos = frame_atual / FPS
            tempo_absoluto_frame = timestamp_inicio_video + tempo_decorrido_segundos
            # ----------------------------------------

            lat_estimada = float(row['lat'])
            lon_estimada = float(row['long'])
            
            # 2. Janela de Tempo (+/- 5s) no AIS
            margem = 5.0
            ais_janela = df_ais[
                (df_ais['timestamp_local'] >= tempo_absoluto_frame - margem) &
                (df_ais['timestamp_local'] <= tempo_absoluto_frame + margem)
            ].copy()
            
            if ais_janela.empty: continue

            # 3. Match de Posição
            ais_janela['distancia'] = ais_janela.apply(
                lambda r: calcular_diferenca_graus(lon_estimada, lat_estimada, r['lon'], r['lat']), axis=1
            )
            
            match = ais_janela.loc[ais_janela['distancia'].idxmin()]
            
            # Tolerância ~500m (0.005 graus)
            if match['distancia'] < 0.005: 
                candidatos_mmsi.append(int(match['mmsi']))
        except Exception as e:
            continue

    if not candidatos_mmsi:
        return "Nenhum Match"
    
    contagem = Counter(candidatos_mmsi)
    vencedor, votos = contagem.most_common(1)[0]
    
    print(f"MMSI Vencedor: {vencedor} ({votos} votos)")
    return vencedor
# ---------------------------
# Loop principal (apenas comandos de missão + fila)
# ---------------------------
print('Iniciado. Aguardando comandos na serial (V / VS / VA / VAS)...')
try:
    while True:
        if ser.in_waiting > 0:
            command = ser.readline().decode('utf-8').rstrip()
            if command ==  'V':
                iniciar_gravacao()
            
            elif command == 'VS':
                parar_gravacao()
                if mission2_proc is None:
                    started = start_mission2_subprocess()
                    
                    
                    maior_valor = df_video['area'].max()
                    resultado = maior_valor
                    try:
                        send_data_to_serial(resultado)
                    except Exception:
                        pass

                mission2_proc = None

            if command == 'VA':
                ais = subprocess.Popen(['python3', MISSao2_PATH], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                sleep(0.5)
                inicio_video = time.time()
                iniciar_gravacao()
                #inicia codigo AIS

            
            elif command == 'VAS':
                parar_gravacao()
                #encerra codigo AIS
                if mission2_proc is None:
                    tempo_absoluto_evento = start_mission2_subprocess(inicio_video)
                    df_m2 = pd.read_csv(caminho_completo)
                    mmsi = encontrar_mmsi(tempo_absoluto_evento, df_video, df_ais)
                    try:
                        send_data_to_serial(mmsi)
                    except Exception:
                        pass
                    
                    try:
                        if started:
                            send_data_to_serial('M2_STARTED')
                        else:
                            send_data_to_serial('M2_NOT_STARTED')
                    except Exception:
                        pass
                else:
                    try:
                        send_data_to_serial('M2_ALREADY_RUNNING')
                    except Exception:
                        pass
            
            else:
                # comportamento idêntico ao código original: salva comando desconhecido na fila
                data_receiver_queue.put(command)
        sleep(0.1)
except KeyboardInterrupt:
    print('Interrompido por teclado. Saindo...')
finally:
    try:
        if mission2_proc is not None:
            try:
                mission2_proc.terminate()
                mission2_proc.wait(timeout=3)
            except Exception:
                try:
                    mission2_proc.kill()
                except Exception:
                    pass
    except Exception:
        pass
    try:
        ser.close()
    except Exception:
        pass
    print('Encerrado.')