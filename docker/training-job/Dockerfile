FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:23.06-py3

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
