FROM --platform=linux/amd64 python:3.10.4-buster

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

