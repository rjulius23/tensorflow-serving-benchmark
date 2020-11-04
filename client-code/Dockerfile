FROM python:3.8

RUN apt-get update && apt-get install -y \
    git \
    curl \
    apache2-utils

ARG MODEL_NAME=identity
ARG CLIENT_PORT=8000
WORKDIR /

COPY requirements.txt .
RUN pip3 install -r requirements.txt && rm requirements.txt

RUN git clone -b r2.3 https://github.com/tensorflow/serving.git && \
    git clone -b r2.3 https://github.com/tensorflow/tensorflow.git

RUN mv tensorflow .tensorflow &&  \
    cd serving && \
    mv ../.tensorflow/tensorflow . && \
    python3 -m grpc.tools.protoc tensorflow_serving/apis/*.proto \
        --python_out=/usr/lib/python3/dist-packages \
        --grpc_python_out=/usr/lib/python3/dist-packages \
        --proto_path=. && \
    cd .. && \
    rm -r serving

COPY *.py /

EXPOSE $CLIENT_PORT
