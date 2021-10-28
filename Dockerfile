FROM python:3.8

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN apt update && apt install -y git python3-pip

RUN pip install torch

RUN git clone https://github.com/DeepRank/Deeprank-GNN.git
WORKDIR Deeprank-GNN
RUN pip install -e .
RUN pip install freesasa pyarrow fastparquet tables

COPY . /usr/src/app
WORKDIR /usr/src/app
