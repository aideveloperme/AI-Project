FROM python:3.12-slim

ARG CHROMA_VERSION
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app
COPY requirements.txt .
# All dependencies are baked in at bundle time; nothing is installed on the Spark.
RUN pip install --no-cache-dir -r requirements.txt "chromadb-client==${CHROMA_VERSION}"

COPY rag.py app.py ingest.py ./
RUN useradd --system --no-create-home app
USER app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
