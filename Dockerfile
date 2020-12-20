FROM nvcr.io/nvidia/pytorch:20.11-py3

WORKDIR /app

RUN pip install -r requirements.txt

ENV PYTONPATH .