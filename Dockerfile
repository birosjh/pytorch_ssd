FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt update && apt install -yq python3-pip python3-opencv software-properties-common libffi-dev

ENV PATH="${PATH}:/root/.local/bin"

# RUN apt install -yq libfreetype6-dev pkg-config

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

RUN pip3 install --upgrade pip

RUN pip3 install poetry

RUN poetry install --no-root