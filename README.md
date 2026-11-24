# SQLocal

Fine-tuning local de un LLM pequeño (Qwen2.5-Coder-1.5B-Instruct + QLoRA) para traducir
preguntas en **inglés** a **SQL** sobre un esquema en inglés, sin que el esquema ni los datos
salgan de la máquina.

> Estado: **setup completo, modelo aún sin entrenar.** Base de datos, dataset, scripts de
> entrenamiento y evaluación, y demo con panel de desempeño listos. La propuesta original está en
> [`docs/propuesta_SQLocal.docx`](docs/propuesta_SQLocal.docx).

## Puesta en marcha

Requisitos: Python 3.10–3.12 (recomendado [uv](https://docs.astral.sh/uv/)), Git y, opcionalmente, Docker.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

```bash
python scripts/build_db.py            # app/demo.db con datos sintéticos (ya incluida)
python scripts/generate_dataset.py    # data/{train,val,test}.jsonl validados por ejecución (ya incluidos)
python scripts/download_model.py      # descarga el modelo base a models/ (único paso con Internet)
python app/app.py                     # demo en http://127.0.0.1:7860
```

## Flujo de trabajo

| Paso | Comando | Dónde |
|---|---|---|
| Evaluar el modelo base | `python scripts/evaluate.py --variant base` | CPU / Apple Silicon / GPU |
| Comprobar datos de entrenamiento | `python scripts/train.py --dry-run` | cualquier equipo |
| Entrenar QLoRA | `pip install -r requirements-train.txt && python scripts/train.py` | GPU NVIDIA (Colab T4) |
| Evaluar el fine-tuneado | `python scripts/evaluate.py --variant finetuned` | CPU / GPU |
| Ver resultados | `python app/app.py` → pestaña **Desempeño** | local |

Tras entrenar en Colab, copia la carpeta `adapter/` y `resultados/train_log.json` al repo: la demo
detecta el adaptador y muestra la curva de pérdida.

## La demo

- **Probar el modelo**: pregunta en inglés → SQL generada → resultado sobre `demo.db`. Si la
  pregunta está en el dataset, compara con la SQL de referencia.
- **Desempeño**: Execution Accuracy y Valid SQL Rate de base vs. fine-tuneado, desglose por tipo
  de consulta, curva de pérdida y explorador de ejemplos (aciertos, errores, mejorados/empeorados
  por el fine-tuning) con la SQL y el resultado de cada modelo. Permite lanzar una evaluación.
- **Dataset**: esquema, estadísticas por split y categoría, y todos los pares.
- **Privacidad**: estado de la guardia de red y registro de conexiones bloqueadas.

## Decisiones de diseño

- **Dataset** (`scripts/generate_dataset.py`): 77 plantillas × 3 paráfrasis, rellenadas con valores
  reales de la base, en 11 categorías. 50 plantillas son básicas (WHERE, agregaciones, GROUP BY,
  ORDER BY/LIMIT, JOIN simple) y 27 difíciles (JOIN de 3-4 tablas, HAVING, lógica de fechas con
  `julianday`/rangos/trimestres, varias condiciones con `IN`, subconsultas no correlacionadas):
  el modelo base ya resuelve casi todas las básicas, así que las difíciles son las que permiten
  medir la mejora del fine-tuning. Solo se conservan pares cuya SQL se ejecuta y devuelve un
  resultado no vacío. División **por plantilla** (`--split-mode template`, estratificada por
  categoría): las plantillas de test no aparecen en train, para no inflar la métrica.
  Resultado actual: 1249 pares (886 / 168 / 195).
- **Prompt**: chat template del modelo Instruct (system + user con esquema y pregunta), idéntico
  en entrenamiento, evaluación y demo ([`sqlocal/prompting.py`](sqlocal/prompting.py)).
- **Entrenamiento**: QLoRA 4-bit NF4, LoRA r=16 / α=32 sobre capas de atención y MLP, 3 épocas,
  lr 2e-4 cosine, early stopping por pérdida de validación; la pérdida solo cuenta la SQL.
- **Execution Accuracy** ([`sqlocal/db.py`](sqlocal/db.py)): mismas filas y valores que la
  referencia; el orden solo importa si la referencia tiene `ORDER BY`; se tolera otro orden de
  columnas; floats redondeados a 4 decimales.
- **Inferencia en CPU**: bitsandbytes 4-bit solo funciona en GPU, así que en CPU el adaptador se
  fusiona con el modelo y se ejecuta en bf16 (~3 GB de RAM).
- **Privacidad**: modo sin conexión de Hugging Face/Gradio, fuentes del sistema (sin Google
  Fonts) y una guardia que bloquea y registra cualquier conexión no local
  ([`sqlocal/privacy.py`](sqlocal/privacy.py)).

## Docker

```bash
python scripts/download_model.py     # una vez, fuera del contenedor
docker compose up --build            # http://127.0.0.1:7860
```

Prueba de aislamiento total (sin red en el contenedor):

```bash
docker run --rm --network none -v "$PWD/models:/app/models:ro" -v "$PWD/resultados:/app/resultados" sqlocal python scripts/evaluate.py --variant base --limit 5
```

## Estructura

```
├── app/            app.py (demo Gradio), schema.sql, demo.db
├── sqlocal/        código compartido: config, db, prompting, model, evaluation, privacy, templates
├── scripts/        build_db, generate_dataset, download_model, train, evaluate
├── data/           train/val/test.jsonl + stats.json
├── adapter/        adaptador LoRA (tras entrenar)
├── resultados/     eval_base.json, eval_finetuned.json, train_log.json
├── docs/           propuesta del proyecto
├── Dockerfile, docker-compose.yml
└── requirements.txt, requirements-train.txt
```

## Memoria del proyecto (apartados pendientes)

1. Portada · 2. Resumen ejecutivo · 3. Planteamiento del problema · 4. Objetivos ·
5. Alcance y limitaciones · 6. Marco teórico · 7. Esquema de la base de datos · 8. Dataset ·
9. Entrenamiento · 10. Evaluación · 11. Demo · 12. Privacidad por diseño · 13. Reproducibilidad ·
14. Limitaciones y trabajo futuro · 15. Conclusiones · 16. Referencias · 17. Anexos
