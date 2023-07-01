FROM --platform=linux/amd64 jupyter/minimal-notebook:notebook-6.4.10

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

