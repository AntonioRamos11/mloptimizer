#!/bin/bash
# This script runs the master and slave processes in separate Docker containers.
# It uses the vastai/base-image (which has NVIDIA CUDA support) and mounts the current directory into /workspace.

# Pull the latest image
echo "Pulling vastai/base-image..."
docker pull vastai/base-image:cuda-12.6.3-cudnn-devel-ubuntu22.04-py313
docker stop rabbitmq
docker rm rabbitmq
docker run -d --hostname rabbmitmq --name rabbitmq -p 15672:15672 -p 5672:5672 rabbitmq:management



# Start the master container (runs run_master.py)
echo "Starting master container..."
docker run --gpus all -d \
  --name mlopt_master \
  -v "$(pwd)":/workspace \
  vastai/base-image \
  bash -c "cd /workspace && python run_master.py"

# Start the slave container (runs run_slave.py)
echo "Starting slave container..."
docker run --gpus all -d \
  --name mlopt_slave \
  -v "$(pwd)":/workspace \
  vastai/base-image \
  bash -c "cd /workspace && python run_slave.py"

# Optionally, follow logs from both containers
echo "Following master container logs..."
docker logs -f mlopt_master &

echo "Following slave container logs..."
docker logs -f mlopt_slave &
