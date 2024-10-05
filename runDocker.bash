sudo docker rm rabbitmq
sudo docker run -d --hostname rabbmitmq --name rabbitmq -p 15672:15672 -p 5672:5672 rabbitmq:management


