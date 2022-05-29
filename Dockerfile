FROM nvidia/cuda:11.6.1-base-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt update && apt install -yq python3.8 python3-pip python3-opencv

RUN apt install -yq libfreetype6-dev pkg-config

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

ENV PYTONPATH .