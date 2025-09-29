FROM python:3.13.7-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["uvicorn", "router:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]