FROM nvcr.io/nvidia/cuda:12.2.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt update && apt install -yq python3-pip python3-opencv software-properties-common libffi-dev

RUN apt-get install libcudnn8=8.9.7.*-1+cuda12.2 libcudnn8-dev=8.9.7.*-1+cuda12.2 libcudnn8-samples=8.9.7.*-1+cuda12.2

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt install -y python3.9

RUN apt install -yq libfreetype6-dev pkg-config

RUN pip3 install --upgrade pip

RUN pip3 install poetry

RUN poetry init

RUN poetry env use "/usr/bin/python3.9"

RUN poetry install

ENV PYTONPATH .