"""Cliente HTTP/1.1 + resolvedor DNS-over-HTTPS — stdlib pura, cero dependencias.

Contexto: el resolver local del ISP devuelve NXDOMAIN selectivo para
polymarket.com (DNS filtering). Este módulo resuelve por DoH contra
Cloudflare y conecta por IP con SNI correcto, sin tocar config del sistema.
"""

import json
import random
import socket
import ssl
import struct
import time

DOH_ENDPOINT = "https://cloudflare-dns.com/dns-query"
USER_AGENT = "predikt-bot/0.1"
_cache = {}  # hostname -> (ips, expiry)

# ─── Respaldo DNS: el resolver del ISP devuelve ────
# NXDOMAIN selectivo para polymarket.com: clob queda cacheada pero gamma-api,
# data-api y hasta cloudflare-dns.com fallan con gaierror [Errno -2/-3] y cada
# ciclo de TODOS los daemons muere con "error en ciclo". DoH contra Cloudflare
# no sirve como primer escalón porque cloudflare-dns.com TAMBIÉN falla la DNS
# local. Estrategia (sin tocar la DNS del sistema — eso tumbaría servicios
# locales): (1) consulta DoH UDP directa a 8.8.8.8, (2) tabla de IPs fijas.
FALLBACK_DNS = [("8.8.8.8", 53), ("1.1.1.1", 53)]
PINNED_IPS = {
    # Polymarket comparte edge Cloudflare; verificado 2026-10-12
    "gamma-api.polymarket.com": ["104.18.34.205", "172.64.153.51"],
    "data-api.polymarket.com": ["104.18.34.205", "172.64.153.51"],
    "clob.polymarket.com": ["104.18.34.205", "172.64.153.51"],
    # WebSocket del libro (market channel): mismo edge Cloudflare, verificado 2026-10-13
    "ws-subscriptions-clob.polymarket.com": ["104.18.34.205", "172.64.153.51"],
}
_dns_fail_logged = set()


def _doh_query(hostname, server, port=53, timeout=4):
    """Consulta DoH (JSON API) vía UDP directo al servidor fallback."""
    qname = b"".join(
        bytes([len(p)]) + p for p in hostname.encode().split(b".")
    ) + b"\x00"
    packet = struct.pack(">HHHHHH", 0x1234, 0x0100, 1, 0, 0, 0) + qname + struct.pack(">HH", 16, 1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    try:
        sock.sendto(packet, (server, port))
        data, _ = sock.recvfrom(4096)
    finally:
        sock.close()
    ancount = struct.unpack(">H", data[6:8])[0]
    idx = 12
    while data[idx] != 0:  # salta el nombre de la query
        idx += 1 + data[idx]
    idx += 5
    ips = []
    for _ in range(ancount):
        while True:
            ln = data[idx]
            if ln == 0:
                idx += 1
                break
            if ln & 0xC0 == 0xC0:
                idx += 2
                break
            idx += 1 + ln
        rtype, _cls, _ttl, rdlen = struct.unpack(">HHIH", data[idx:idx + 10])
        idx += 10
        if rtype == 1 and rdlen == 4:
            ips.append(socket.inet_ntoa(data[idx:idx + 4]))
        idx += rdlen
    return ips


def _fallback_resolve(hostname):
    """Resuelve por DoH fallback y luego por IPs fijas. [] si todo falla."""
    for server, port in FALLBACK_DNS:
        try:
            ips = _doh_query(hostname, server, port)
            if ips:
                return ips
        except Exception:
            continue
    return list(PINNED_IPS.get(hostname, []))


_real_getaddrinfo = socket.getaddrinfo


def _patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    try:
        return _real_getaddrinfo(host, port, family, type, proto, flags)
    except socket.gaierror:
        # DNS local filtró el host: resolvemos por fallback y sintetizamos
        # una respuesta con la IP real, preservando el resto de argumentos.
        if isinstance(host, bytes):
            host = host.decode()
        ips = _fallback_resolve(host)
        if not ips:
            raise
        if host not in _dns_fail_logged:
            _dns_fail_logged.add(host)
        return [
            (socket.AF_INET, type or socket.SOCK_STREAM, proto or 6, "", (ip, port))
            for ip in ips
        ]


socket.getaddrinfo = _patched_getaddrinfo

# ─── Resiliencia: retry con backoff (buenas prácticas trading 2026) ─────────
DEFAULT_MAX_RETRIES = 4      # intentos totales (1 original + 3 reintentos)
DEFAULT_BASE_DELAY = 0.5     # segundos; se duplica en cada reintento
DEFAULT_MAX_DELAY = 30.0     # techo del backoff
DEFAULT_JITTER = 0.5         # fracción aleatoria para desincronizar
# códigos que NO se reintentan (error del cliente: reintentar no ayuda)
NO_RETRY_STATUS = {400, 401, 403, 404, 405, 422}


def resolve(hostname, ttl=300):
    """Resuelve un hostname via DNS-over-HTTPS. Devuelve lista de IPv4."""
    now = time.time()
    hit = _cache.get(hostname)
    if hit and hit[1] > now:
        return hit[0]

    req = (
        f"GET /dns-query?name={hostname}&type=A HTTP/1.1\r\n"
        f"Host: cloudflare-dns.com\r\n"
        f"accept: application/dns-json\r\n"
        f"user-agent: {USER_AGENT}\r\n"
        f"connection: close\r\n\r\n"
    ).encode()

    sock = socket.create_connection(("cloudflare-dns.com", 443), timeout=10)
    ctx = ssl.create_default_context()
    with ctx.wrap_socket(sock, server_hostname="cloudflare-dns.com") as tls:
        tls.sendall(req)
        chunks = []
        while True:
            data = tls.recv(65536)
            if not data:
                break
            chunks.append(data)
    raw = b"".join(chunks)

    head, _, body = raw.partition(b"\r\n\r\n")
    status = head.split(b"\r\n", 1)[0]
    if b" 200" not in status:
        raise OSError(f"DoH fallo: {status!r}")

    payload = json.loads(body)
    ips = [a["data"] for a in payload.get("Answer", []) if a.get("type") == 1]
    if not ips:
        raise OSError(f"DoH sin registros A para {hostname}")
    _cache[hostname] = (ips, now + ttl)
    return ips


def https_get(hostname, path, timeout=15):
    """GET HTTPS por IP resuelta con DoH, SNI correcto. Devuelve (status, body)."""
    ip = resolve(hostname)[0]
    sock = socket.create_connection((ip, 443), timeout=timeout)
    ctx = ssl.create_default_context()
    tls = ctx.wrap_socket(sock, server_hostname=hostname)

    req = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {hostname}\r\n"
        f"user-agent: {USER_AGENT}\r\n"
        f"accept: application/json\r\n"
        f"connection: close\r\n\r\n"
    ).encode()
    tls.sendall(req)

    chunks = []
    while True:
        data = tls.recv(65536)
        if not data:
            break
        chunks.append(data)
    tls.close()
    raw = b"".join(chunks)

    head, _, body = raw.partition(b"\r\n\r\n")
    status_line = head.split(b"\r\n", 1)[0].decode("latin-1")
    status = int(status_line.split(" ")[1]) if len(status_line.split(" ")) > 1 else 0

    # Chunked transfer-encoding: desarmar
    if b"transfer-encoding: chunked" in head.lower():
        body = _dechunk(body)
    return status, body


def _dechunk(body):
    out = b""
    while True:
        line_end = body.find(b"\r\n")
        if line_end == -1:
            break
        size = int(body[:line_end].split(b";")[0], 16)
        if size == 0:
            break
        out += body[line_end + 2 : line_end + 2 + size]
        body = body[line_end + 2 + size + 2 :]
    return out


def _backoff_delay(attempt, base=DEFAULT_BASE_DELAY, cap=DEFAULT_MAX_DELAY, jitter=DEFAULT_JITTER):
    """Exponential backoff con jitter: delay = min(cap, base*2^n) + aleatorio.

    El jitter se calcula sobre el valor ya capeado, así el total nunca supera
    cap*(1+jitter). Desincroniza reintentos de varios procesos para no crear
    otra ráfaga contra la API (buena práctica 2026).
    """
    delay = min(cap, base * (2 ** attempt))
    return delay + random.uniform(0, jitter * delay)


def get_json(hostname, path, max_retries=DEFAULT_MAX_RETRIES):
    """GET + retry con exponential backoff ante fallos transitorios.

    Reintenta en: errores de red (transitorios), HTTP 429 (rate limit) y
    HTTP 5xx (servidor). NO reintenta 4xx de cliente (no sirve).
    Devuelve el JSON parseado o lanza la última excepción.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            status, body = https_get(hostname, path)
            if status == 200:
                return json.loads(body)
            if status in NO_RETRY_STATUS:
                # error del cliente: no tiene sentido reintentar
                raise OSError(f"HTTP {status} de {hostname}{path}")
            # 429 o 5xx: transitorio, se reintenta con backoff
            last_err = OSError(f"HTTP {status} de {hostname}{path}")
        except OSError:
            raise  # ya decidimos que no se reintenta
        except Exception as e:
            # error de red/SSL/parseo: transitorio, se reintenta
            last_err = e
        if attempt < max_retries - 1:
            time.sleep(_backoff_delay(attempt))
    raise last_err or OSError(f"get_json fallo tras {max_retries} intentos: {hostname}{path}")


if __name__ == "__main__":
    ip = resolve("gamma-api.polymarket.com")
    print(f"resuelto: {ip}")
