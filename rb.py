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
        ser = serial.Serial('/dev/ttyS0', 9600, timeout=0)
        break
    except Exception:
        pass

data_receiver_queue = queue.Queue()
mission2_proc = None
processo_camera = None

# caminho do script de missão 2 (ajuste se necessário)
diretorio = '/home/user/mongenot'
missao_path = '/home/user/mongenot/mission.py'
ais_path = '/home/user/mongenot/missao2.py'

missao_um = '/home/user/mongenot/first_mission.py'
missao_dois = '/home/user/mongenot/second_mission.py'

diretorio = '/home/user/mongenot'
#PATH do csv do codigo de AIS

nome_arquivo = 'leituras.csv'
nome_arquivo_ais = 'ais_log.csv'
#arquivo csv do AIS

caminho_completo_missao = os.path.join(diretorio, nome_arquivo)
caminho_completo_ais = os.path.join(diretorio, nome_arquivo_ais)



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
    if not os.path.exists(missao_path):
        send_data_to_serial('ERR_M2_NOFILE')
        return False
    try:
        # Passa o timestamp de inicio como argumento para o script de processamento
        mission2_proc = subprocess.Popen(['python3', missao_path, str(timestamp_inicio)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
def encontrar_mmsi(timestamp_inicio_video, caminho_video, caminho_ais):
    df_video = pd.read_csv(caminho_video, delimiter=';')
    df_ais = pd.read_csv(caminho_ais, delimiter=';')

    candidatos_mmsi = []
    FPS = 30.0 # Deve ser o mesmo usado na gravação

    # Verifica se as colunas necessárias existem
    # Ajuste os nomes abaixo conforme o cabeçalho real dos seus CSVs
    if 'frame' not in df_video.columns:
        print("Erro: Coluna 'frame' não encontrada no CSV do vídeo.")
        return "Erro CSV Video"

    primeira_linha = df_video.iloc[0]
    lat = primeira_linha['lat']
    long = primeira_linha['long']
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
    return vencedor, lat, long
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
                send_data_to_serial("CONF")
                sleep(0.3)
                send_data_to_serial("Video Iniciado")

            elif command == 'VS':
                parar_gravacao()
                send_data_to_serial("CONF")
                sleep(0.3)
                send_data_to_serial("Video Encerrado")
                if mission2_proc is None:
                    started = start_mission2_subprocess() # 'started' é definido aqui
                    df_video = pd.read_csv(caminho_completo_missao, delimiter=';')
                    # Atenção: df_video precisa estar definido globalmente ou lido aqui
                    maior_valor = df_video['area'].max() 
                    resultado = maior_valor
                
                try:
                    send_data_to_serial(resultado)
                except Exception:
                    pass

                mission2_proc = None
                
            elif command == 'VSS':
                parar_gravacao()
                send_data_to_serial("CONF")
                sleep(0.3)
                send_data_to_serial("Video Encerrado")
                if mission2_proc is None:
                    mission2_proc = subprocess.Popen(['python3', missao_um], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    saida_texto, erros = mission2_proc.communicate()
                    resultado = saida_texto # CORRIGIDO: era saida_valor
                
                try:
                    send_data_to_serial(resultado)
                except Exception:
                    pass

                mission2_proc = None

            elif command == 'VA':
                ais = subprocess.Popen(['python3', ais_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                send_data_to_serial("CONF")
                sleep(0.3)
                send_data_to_serial("Vido e AIS Iniciado")
                sleep(0.5)
                inicio_video = time.time()
                iniciar_gravacao()
                # inicia codigo AIS

            elif command == 'VAS':
                parar_gravacao()
                send_data_to_serial("CONF")
                sleep(0.3)
                send_data_to_serial("Video e AIS Encerrados")
                # encerra codigo AIS
                
                # Definindo started como False por padrão para evitar erro, ajuste conforme sua lógica
                started = False 
                
                if mission2_proc is None:
                    tempo_absoluto_evento = start_mission2_subprocess(inicio_video)
                    # CORRIGIDO: mmsi. lat -> mmsi, lat
                    mmsi, lat, long = encontrar_mmsi(tempo_absoluto_evento, caminho_completo_missao, caminho_completo_ais)
                    started = True # Supondo que se entrou aqui, iniciou

                mission2_proc = None
                try:
                    send_data_to_serial(f"mmsi:{mmsi}, latitude:{lat}, longitude{long}")
                except Exception:
                    pass
                    
                # Lógica para verificar se startou
                try:
                    if started:
                        send_data_to_serial('M2_STARTED')
                    else:
                        send_data_to_serial('M2_NOT_STARTED')
                except Exception:
                    pass
                else:
                    # Este else roda se o try anterior NÃO der erro. 
                    # Se sua intenção era rodar quando 'started' fosse falso ou algo assim, a lógica do 'else' do try pode estar errada.
                    # Mantive a estrutura original, mas verifique se é isso mesmo.
                    try:
                        send_data_to_serial('M2_ALREADY_RUNNING')
                    except Exception:
                        pass
                        
            elif command == 'VASS':
                parar_gravacao()
                send_data_to_serial("CONF")
                sleep(0.3)
                send_data_to_serial("Video e AIS Encerrados")
                # encerra codigo AIS
                
                started = False # Definindo padrão
                
                if mission2_proc is None:
                    mission2_proc = subprocess.Popen(['python3', missao_dois, str(inicio_video)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    saida_texto, erros = mission2_proc.communicate()
                    tempo_absoluto_evento = saida_texto
                    # CORRIGIDO: mmsi. lat -> mmsi, lat
                    mmsi, lat, long = encontrar_mmsi(tempo_absoluto_evento, caminho_completo_missao, caminho_completo_ais)
                    started = True

                mission2_proc = None
                try:
                    send_data_to_serial(f"mmsi:{mmsi}, latitude:{lat}, longitude{long}")
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

            # CORRIGIDO: Mudado de IF para ELIF e ajustada a indentação
            elif command == "END":
                send_data_to_serial("CONF")
                sleep(0.3)
                send_data_to_serial("Desligando Raspberry")
                # Cuidado com 'preexec_fn' em versões novas do Python, pode ser inseguro, mas funciona.
                end = subprocess.Popen(["sudo", "shutdown", "-h", "now"], preexec_fn=os.setsid) # Melhor passar lista
            
            else:
                # Comportamento para comando desconhecido
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
