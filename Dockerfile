# AVA - Hugging Face Spaces (Docker SDK, app_port 7860).
# opencv-python-headless wheels bundle their native libs: no apt packages are needed.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# HF Spaces runs the container as uid 1000.
RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user ava ./ava
COPY --chown=user server ./server
COPY --chown=user web ./web

USER user
EXPOSE 7860

# One worker on purpose: the in-memory reference cache lives in this process.
# PORT is set by Railway / Render; Hugging Face Spaces uses 7860.
CMD ["sh", "-c", "exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
