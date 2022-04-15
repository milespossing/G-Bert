FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN conda install pyg -c pyg
RUN conda install scikit-learn-intelex
COPY requirements.txt requirements.txt
RUN pip install -qqr requirements.txt