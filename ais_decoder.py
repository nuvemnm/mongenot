import re
def parse_nmea_sentence(nmea: str) -> str:
    """
    Extrai o campo payload de uma sentença NMEA AIVDM/AIVDO.
    Exemplo: !AIVDM,1,1,,A,13aEOK?P00PD2wVMdLDRhgvL289?,0*26
    Retorna a string do payload, ou levanta ValueError se inválido.
    """
    nmea = nmea.strip()
    if not nmea.startswith("!"):
        raise ValueError("Sentença NMEA inválida (não começa com '!').")
    # separar por vírgulas sem considerar checksum (após *)
    main_part = nmea.split('*')[0]
    fields = main_part.split(',')
    # payload no campo 5 (index 5), considerando formato padrão AIVDM/AIVDO
    # Ex.: [0]=!AIVDM [1]=1 [2]=1 [3]=<seq id> [4]=A [5]=payload [6]=fillbits
    if len(fields) < 6:
        raise ValueError("Sentença NMEA incompleta (menos de 6 campos).")
    payload = fields[5]
    if not payload:
        raise ValueError("Payload AIS ausente na sentença NMEA.")
    return payload
def ais_char_to_sixbit(char: str) -> str:
    """
    Converte 1 caractere AIS para bits (6-bit) seguindo ITU-R M.1371-5.
    """
    val = ord(char)
    # Mapeamento padrão: se ASCII < 88 => val -= 48, caso contrário val -= 56
    if val < 88:
        v = val - 48
    else:
        v = val - 56
    if not (0 <= v <= 63):
        raise ValueError(f"Caractere AIS fora do intervalo válido: '{char}' (ord={val}) -> {v}")
    return format(v, "06b")
def ais_payload_to_bits(payload: str) -> str:
    """Converte toda a string AIS (payload) para sequência contínua de bits."""
    return "".join(ais_char_to_sixbit(c) for c in payload)
def twos_complement(value: int, bit_width: int) -> int:
    """Interpreta 'value' como inteiro com 'bit_width' bits em complemento de dois e retorna o signed int."""
    sign_bit = 1 << (bit_width - 1)
    mask = (1 << bit_width) - 1
    value &= mask
    if value & sign_bit:
        return value - (1 << bit_width)
    else:
        return value
def decode_position_report(payload_bits: str) -> dict:
    """
    Decodifica mensagem AIS tipo 1 (Position Report Class A).
    Exige payload_bits com pelo menos 168 bits.
    Retorna dict com os campos solicitados pelo edital.
    """
    if len(payload_bits) < 168:
        raise ValueError(f"Payload em bits muito curto: {len(payload_bits)} bits (esperado >=168).")
    # Use apenas os primeiros 168 bits (mensagem tipo 1)
    bits = payload_bits[:168]
    fields = {}
    # Índices conforme ITU-R M.1371-5 (0-indexed)
    fields["message_id"] = int(bits[0:6], 2)
    fields["repeat_indicator"] = int(bits[6:8], 2)
    fields["mmsi"] = int(bits[8:38], 2)
    fields["navigational_status"] = int(bits[38:42], 2)
    rot_raw = int(bits[42:50], 2)  # 8 bits
    # ROT: 0x80 (128) significa "not available"
    if rot_raw == 128:
        fields["rot"] = None
    else:
        fields["rot"] = twos_complement(rot_raw, 8)
    sog_raw = int(bits[50:60], 2)  # 10 bits
    fields["sog"] = None if sog_raw == 1023 else (sog_raw / 10.0)
    fields["position_accuracy"] = int(bits[60:61], 2)
    lon_raw = int(bits[61:89], 2)  # 28 bits
    lon_signed = twos_complement(lon_raw, 28)
    # Valor reservado "not available" = 181 * 600000 (segundo especificação)
    special_lon = 181 * 600000
    fields["longitude"] = None if lon_raw == special_lon else (lon_signed / 600000.0)
    lat_raw = int(bits[89:116], 2)  # 27 bits
    lat_signed = twos_complement(lat_raw, 27)
    special_lat = 91 * 600000
    fields["latitude"] = None if lat_raw == special_lat else (lat_signed / 600000.0)
    cog_raw = int(bits[116:128], 2)  # 12 bits
    fields["cog"] = None if cog_raw == 3600 else (cog_raw / 10.0)
    heading_raw = int(bits[128:137], 2)  # 9 bits
    fields["true_heading"] = None if heading_raw == 511 else heading_raw
    fields["timestamp"] = int(bits[137:143], 2)  # 6 bits
    # Os demais campos (special manoeuvre, RAIM, communication state) não são exigidos,
    # mas pode-se extrair caso necessário:
    fields["special_manoeuvre"] = int(bits[143:145], 2)
    # bits 145:148 são 'spare' (3 bits)
    fields["raim_flag"] = int(bits[148:149], 2)
    fields["communication_state"] = int(bits[149:168], 2)
    return fields
# Exemplo de uso com uma sentença NMEA típica:
if __name__ == "__main__":
    # Exemplo NMEA (igual ao do seu enunciado)
    nmea_sentence = "!AIVDM,1,1,,A,13aEOK?P00PD2wVMdLDRhgvL289?,0*26"
    try:
        payload = parse_nmea_sentence(nmea_sentence)
        bits = ais_payload_to_bits(payload)
        # Garantia: se houver múltiplas frases concatenadas ou se for multi-sentence,
        # seria necessário juntar os payloads na ordem; aqui assumimos 1-frase por mensagem.
        decoded = decode_position_report(bits)
        for k, v in decoded.items():
            print(f"{k}: {v}")
    except Exception as e:
        print("Erro ao decodificar AIS:", e)
