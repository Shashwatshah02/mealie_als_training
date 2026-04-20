FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/train.py /app/train.py

ENV OPENBLAS_NUM_THREADS=1
ENV MLFLOW_TRACKING_URI=http://mlflow-service.platform.svc.cluster.local:5000
ENV MLFLOW_S3_ENDPOINT_URL=http://minio-service.platform.svc.cluster.local:9000
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin123
ENV MINIO_ENDPOINT=http://minio-service.platform.svc.cluster.local:9000
ENV MINIO_ACCESS_KEY=minioadmin
ENV MINIO_SECRET_KEY=minioadmin123

CMD ["python", "train.py"]
