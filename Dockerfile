FROM python:3.10

WORKDIR /app

COPY requirements.txt .
COPY fast_api.py .
COPY best_model.pkl .
COPY label_encoder.pkl .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "8000"]
