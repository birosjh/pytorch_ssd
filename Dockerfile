FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y python3-opencv

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

ENV PYTONPATH .