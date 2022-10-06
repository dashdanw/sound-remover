FROM nvidia/cuda:11.8.0-base-ubuntu22.04

WORKDIR /app

COPY . .

RUN apt update && \
    apt install -y python3 python3-pip python-is-python3 python3-cachecontrol && \
    apt install -y python3-poetry && \
    poetry config virtualenvs.create false && \
    poetry install