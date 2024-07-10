FROM python:3.11-slim

RUN mkdir -p /mlflow-example

COPY . /mlflow-example

WORKDIR /mlflow-example

RUN pip install --no-cache-dir -U pip

RUN pip install -r requirements.txt

CMD ["python3", "tracking.py"]