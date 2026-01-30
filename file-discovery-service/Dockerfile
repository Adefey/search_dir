FROM python:3.14.2-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["uvicorn", "router:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]