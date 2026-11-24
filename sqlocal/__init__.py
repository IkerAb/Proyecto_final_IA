"""SQLocal: generación de SQL en inglés con un LLM local."""

import os
from pathlib import Path

# Debe fijarse antes de que cualquier librería importe huggingface_hub (p. ej. gradio),
# porque la ruta de la caché se lee una sola vez al importarlo.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "models" / "hf"))
