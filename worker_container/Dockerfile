FROM tensorflow/tensorflow:2.2.0-gpu

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update

RUN apt-get install -y python3.7 python3-pip
RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install --upgrade setuptools

RUN python3.7 -m pip install tensorflow-gpu==2.2.0
RUN python3.7 -m pip install numpy
RUN python3.7 -m pip install tensorflow-datasets
RUN python3.7 -m pip install dataclasses
RUN python3.7 -m pip install aio-pika
RUN python3.7 -m pip install aiohttp
RUN python3.7 -m pip install optuna
RUN python3.7 -m pip install dataclasses-json

# docker run --name mlbox_master -u 1000:1000 --gpus=all --network host -e HOME=/project -it --rm -v /home/mario/Projects/ml-brain-optimizer:/project -w /project mlbo/worker:0.1.0 bash
# docker run --name mlbox_worker -u 1000:1000 --gpus=all --network host -e HOME=/project -it --rm -v /home/mario/Projects/ml-brain-optimizer:/project -w /project mlbo/worker:0.1.0 bash
# docker run -d --hostname rabbmitmq --name rabbitmq -p 15672:15672 -p 5672:5672 rabbitmq:management