FROM continuumio/miniconda3:latest AS base
RUN conda install python=3.8

FROM base as torch
RUN conda install pytorch cudatoolkit=11.3 -c pytorch
RUN conda install pyg -c pyg
RUN conda install scikit-learn-intelex

FROM torch

COPY requirements.txt requirements.txt
RUN pip install -qqr requirements.txt