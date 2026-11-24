"""Privacidad por diseño: modo sin conexión y guardia de red.

`enable_offline_mode()` debe llamarse ANTES de importar transformers, huggingface_hub
o gradio, para que las librerías lean las variables de entorno al cargarse.

`install_network_guard()` intercepta `socket.connect` y bloquea cualquier conexión
que no vaya a la propia máquina (loopback). Cada intento bloqueado queda registrado,
y ese registro es la evidencia que muestra la pestaña «Privacidad» de la demo.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import threading
from datetime import datetime

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "GRADIO_ANALYTICS_ENABLED": "False",
    "DO_NOT_TRACK": "1",
}

_blocked: list[dict] = []
_allowed_count = 0
_lock = threading.Lock()
_installed = False


def enable_offline_mode() -> None:
    for key, value in OFFLINE_ENV.items():
        os.environ[key] = value


def _is_local(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        # Un nombre de host que no es una IP: solo aceptamos «localhost».
        return False


def install_network_guard() -> None:
    """Bloquea conexiones salientes que no sean a loopback (idempotente)."""
    global _installed
    if _installed:
        return
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def _check(sock: socket.socket, address) -> None:
        global _allowed_count
        if sock.family not in (socket.AF_INET, socket.AF_INET6):
            return  # sockets Unix, etc.
        host = address[0] if isinstance(address, tuple) else str(address)
        port = address[1] if isinstance(address, tuple) and len(address) > 1 else None
        with _lock:
            if _is_local(host):
                _allowed_count += 1
                return
            _blocked.append(
                {"hora": datetime.now().strftime("%H:%M:%S"), "destino": host, "puerto": port}
            )
        raise ConnectionRefusedError(f"[SQLocal] Conexión externa bloqueada: {host}:{port}")

    def guarded_connect(self, address):
        _check(self, address)
        return original_connect(self, address)

    def guarded_connect_ex(self, address):
        _check(self, address)
        return original_connect_ex(self, address)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    _installed = True


def guard_status() -> dict:
    with _lock:
        return {
            "guardia_activa": _installed,
            "conexiones_locales_permitidas": _allowed_count,
            "conexiones_externas_bloqueadas": list(_blocked),
            "variables_entorno": {k: os.environ.get(k) for k in OFFLINE_ENV},
        }
