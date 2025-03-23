FROM python:3.10-slim

WORKDIR /app

# Add environment variable to prevent torch auto-installation if triggered by transformers
ENV TRANSFORMERS_NO_TORCH=1

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .

RUN pip install --upgrade pip && pip install -r requirements-prod.txt --no-cache-dir

COPY . .

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
