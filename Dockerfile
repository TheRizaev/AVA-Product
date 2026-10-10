# Dalil QA - Hugging Face Spaces (Docker SDK, app_port 7860).
# opencv-python-headless wheels bundle their native libs: no apt packages are needed.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DALIL_DATA_DIR=/tmp/dalil

# HF Spaces runs the container as uid 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user dalil ./dalil
COPY --chown=user server ./server
COPY --chown=user web ./web

USER user
EXPOSE 7860

# One worker on purpose: the journal cache and per-station locks live in this process.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
