#docker build -t mlbo/worker:0.1.0 .
#master
docker run --name mlbox_master -u 1000:1000 --gpus=all --network host -e HOME=/project -it --rm -v /home/mario/Projects/ml-brain-optimizer:/project -w /project mlbo/worker:0.1.0 bash
#slave 
#docker run --name mlbox_worker -u 1000:1000 --gpus=all --network host -e HOME=/project -it --rm -v /home/mario/Projects/ml-brain-optimizer:/project -w /project mlbo/worker:0.1.0 bash
#star rabbitmq
docker run -d --hostname rabbmitmq --name rabbitmq -p 15672:15672 -p 5672:5672 rabbitmq:management
    