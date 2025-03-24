FROM python:3.10-slim

WORKDIR /app

# Add environment variable to prevent torch auto-installation if triggered by transformers
ENV TRANSFORMERS_NO_TORCH=1
ENV PYTHONUNBUFFERED=1
ENV RENDER=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first to take advantage of Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy the application code
COPY . .

# Use PORT environment variable with fallback to 10000
ENV PORT=10000
EXPOSE ${PORT}

# Run the application
CMD gunicorn --workers=2 --threads=4 --timeout=120 --bind 0.0.0.0:${PORT} app:app
