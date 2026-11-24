"""Construcción del prompt. Es el MISMO en entrenamiento, evaluación y demo.

Se usa el chat template del modelo Instruct (mensajes system/user/assistant) en lugar de
un prompt de texto plano, para que la comparación base vs. fine-tuneado sea justa: el
modelo base recibe la pregunta en el formato para el que fue entrenado.
"""

from __future__ import annotations

import re
from functools import lru_cache

from .config import SCHEMA_PATH

SYSTEM_PROMPT = (
    "You are an expert SQLite assistant. Given a database schema and a question, "
    "write one SQLite query that answers the question. "
    "Return only the SQL query, with no explanation."
)


@lru_cache(maxsize=1)
def get_schema() -> str:
    return SCHEMA_PATH.read_text().strip()


def build_messages(question: str, sql: str | None = None) -> list[dict]:
    """Mensajes de chat. Si se pasa `sql`, se añade como respuesta (ejemplo de entrenamiento)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"### Schema:\n{get_schema()}\n\n### Question:\n{question.strip()}"},
    ]
    if sql is not None:
        messages.append({"role": "assistant", "content": sql.strip()})
    return messages


_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> str:
    """Extrae la primera sentencia SQL de la respuesta del modelo."""
    text = (text or "").strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    # Desde el primer SELECT/WITH hasta el primer punto y coma.
    start = re.search(r"\b(SELECT|WITH)\b", text, re.IGNORECASE)
    if start:
        text = text[start.start():]
    if ";" in text:
        text = text[: text.index(";") + 1]
    return " ".join(text.split())
