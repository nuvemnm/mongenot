#!/usr/bin/env python3
import subprocess
import socket
import time
import csv
import os

# IMPORTA O DECODER (presume que essas funções existem no seu módulo)
from ais_decoder import (
    parse_nmea_sentence,
    ais_payload_to_bits,
    decode_position_report
)

print("Iniciando AIS-catcher...")
ais_process = subprocess.Popen(
    ["AIS-catcher", "-f", "400.0M", "-u", "127.0.0.1", "10110", "-v", "2", "-o", "5"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

# espera rápida para o AIS-catcher iniciar
time.sleep(2)

UDP_IP = "127.0.0.1"
UDP_PORT = 10110

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("AIS iniciado. Aguardando mensagens...\n")

# Lista em memória (opcional)
ais_log = []

CSV_FILE = "ais_log.csv"

# Sempre recria o arquivo do zero ao iniciar o script
with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp_local", "timestamp", "mmsi", "latitude", "longitude", "time"])


# inicia o contador de tempo
start_time = time.time()

try:
    while True:
        data, addr = sock.recvfrom(4096)
        msg = data.decode(errors="ignore").strip()

        if not msg.startswith("!AIVDM"):
            continue

        try:
            # Decodificação
            payload = parse_nmea_sentence(msg)
            bits = ais_payload_to_bits(payload)
            decoded = decode_position_report(bits)

            # Filtra apenas message type 1 (se desejar manter)
            if decoded.get("message_id") != 1:
                continue

            # Timestamp local de recepção
            t_local = time.time()
            ais_log.append((t_local, decoded))

            # Campos para CSV
            timestamp_utc = decoded.get("timestamp", None)
            mmsi = decoded.get("mmsi", None)
            lat = decoded.get("latitude", None)
            lon = decoded.get("longitude", None)
            elapsed_time = time.time() - start_time

            # Grava no CSV
            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([t_local, timestamp_utc, mmsi, lat, lon, elapsed_time])

            # Impressão no console
            print("\n=== AIS RECEBIDO ===")
            print("Timestamp local:", t_local)
            for k, v in decoded.items():
                print(f"{k}: {v}")
            print(f"time (elapsed): {elapsed_time:.3f} s")

        except Exception as e:
            print("Erro ao decodificar:", e)
            continue

except KeyboardInterrupt:
    print("\nEncerrando...")

finally:
    try:
        ais_process.terminate()
        ais_process.wait(timeout=5)
    except Exception:
        pass
    try:
        sock.close()
    except Exception:
        pass

    print("AIS-catcher finalizado. CSV salvo em:", os.path.abspath(CSV_FILE))

