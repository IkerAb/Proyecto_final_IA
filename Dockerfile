# SQLocal · demo 100 % local en CPU
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/models/hf \
    SQLOCAL_CPU_DTYPE=bfloat16

WORKDIR /app

# torch en su variante CPU (mucho más ligera que la de CUDA)
COPY requirements.txt .
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY sqlocal ./sqlocal
COPY scripts ./scripts
COPY app ./app
COPY data ./data
COPY adapter ./adapter
COPY resultados ./resultados

# Los pesos del modelo base NO van en la imagen: se montan desde ./models (ver docker-compose.yml).
EXPOSE 7860
CMD ["python", "app/app.py", "--host", "0.0.0.0", "--port", "7860"]
