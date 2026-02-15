FROM python:3.11-slim

WORKDIR /app
COPY app/ /app/

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8000
CMD ["sh", "-lc", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
