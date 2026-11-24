"""Descarga el modelo base a models/ (único paso que necesita Internet).

Después, entrenamiento, evaluación y demo funcionan sin conexión.
"""

import _bootstrap  # noqa: F401
from sqlocal.config import BASE_MODEL_ID, MODELS_DIR

if __name__ == "__main__":
    from huggingface_hub import snapshot_download

    path = snapshot_download(BASE_MODEL_ID, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    print(f"{BASE_MODEL_ID} descargado en {path}\n(caché: {MODELS_DIR / 'hf'})")
