FROM python:3.10-slim

WORKDIR /app

ENV TRANSFORMERS_NO_TORCH=1
ENV PYTHONUNBUFFERED=1
ENV RENDER=1

ENV MALLOC_ARENA_MAX=2

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install dependencies - try to make the installation more robust
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy application code
COPY . .

ENV PORT=10000
EXPOSE ${PORT}

# Run application
CMD gunicorn --workers=1 --threads=2 --timeout=120 --bind 0.0.0.0:${PORT} app:app
