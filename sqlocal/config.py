"""Configuración compartida: rutas, modelo e hiperparámetros por defecto."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Rutas del proyecto
DATA_DIR = ROOT / "data"
APP_DIR = ROOT / "app"
DB_PATH = APP_DIR / "demo.db"
SCHEMA_PATH = APP_DIR / "schema.sql"
ADAPTER_DIR = ROOT / "adapter"
RESULTS_DIR = ROOT / "resultados"
MODELS_DIR = ROOT / "models"

# Los pesos se guardan dentro del proyecto para poder trabajar sin conexión.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))

# Modelo base (Apache 2.0). Se puede cambiar por la alternativa de 0.5B si falta memoria.
BASE_MODEL_ID = os.environ.get("SQLOCAL_BASE_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")

# Generación
MAX_NEW_TOKENS = int(os.environ.get("SQLOCAL_MAX_NEW_TOKENS", "128"))

# Ejecución de SQL
SQL_TIMEOUT_S = 5.0
MAX_ROWS = 1000

VARIANTS = {
    "base": "Modelo base (sin fine-tuning)",
    "finetuned": "Modelo fine-tuneado (base + adaptador LoRA)",
}


def adapter_available() -> bool:
    return (ADAPTER_DIR / "adapter_config.json").exists()


def results_path(variant: str) -> Path:
    return RESULTS_DIR / f"eval_{variant}.json"
