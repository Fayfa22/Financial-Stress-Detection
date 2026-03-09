FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV NLTK_DATA=/usr/local/nltk_data

RUN mkdir -p /usr/local/nltk_data

RUN python - <<EOF
import nltk
nltk.download('punkt', download_dir='/usr/local/nltk_data', quiet=True)
nltk.download('stopwords', download_dir='/usr/local/nltk_data', quiet=True)
nltk.download('wordnet', download_dir='/usr/local/nltk_data', quiet=True)
nltk.download('omw-1.4', download_dir='/usr/local/nltk_data', quiet=True)
EOF

COPY api/ .
COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]