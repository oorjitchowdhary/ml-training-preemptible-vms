FROM python:3.10-slim

WORKDIR /app

COPY . /app

ENV GOOGLE_APPLICATION_CREDENTIALS="/app/robs-project-382021-29095b54cf4c.json"

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "index.py"]
