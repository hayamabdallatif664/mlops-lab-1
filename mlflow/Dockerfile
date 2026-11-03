FROM python:3.11-slim

# Pinned to the same mlflow version as the inference image's client.
# The cache mount keeps downloaded wheels between builds (not in the image), so a
# build interrupted by a slow/broken connection resumes instead of restarting.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 10 mlflow==3.16.0

EXPOSE 5000

# Everything is written under /mlflow-data, the mount point of the named volume.
# --serve-artifacts + --artifacts-destination: clients upload/download model files
# over HTTP through this server (mlflow-artifacts:/ URIs) instead of needing direct
# access to its disk. --allowed-hosts: accept requests addressed to the Compose
# service name "mlflow" (MLflow 3 rejects unknown Host headers by default).
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--serve-artifacts", "--artifacts-destination", "/mlflow-data/mlruns", \
     "--allowed-hosts", "mlflow,mlflow:*,localhost,localhost:*,127.0.0.1,127.0.0.1:*"]
