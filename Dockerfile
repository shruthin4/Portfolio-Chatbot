FROM python:3.10-slim
WORKDIR /app
# Adding environment variable to prevent torch auto-installation if triggered by transformers
ENV TRANSFORMERS_NO_TORCH=1
ENV PYTHONUNBUFFERED=1
ENV RENDER=1
ENV MALLOC_ARENA_MAX=2


RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .


RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt wordnet stopwords

RUN python -m spacy download en_core_web_sm

COPY . .

ENV PORT=10000
EXPOSE ${PORT}

CMD gunicorn --workers=1 --threads=2 --timeout=120 --bind 0.0.0.0:${PORT} app:app
