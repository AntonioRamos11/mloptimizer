#!/bin/bash
# deploy.sh - Optimized for execution INSIDE the vast.ai base image

# 1. Repository Setup
if [ ! -d "mloptimizer" ]; then
  echo "Cloning repository..."
  git clone https://github.com/AntonioRamos11/mloptimizer.git
fi

cd mloptimizer || { echo "Failed to enter repo directory"; exit 1; }
git pull

# 2. Conda Setup
eval "$(conda shell.bash hook)"
if ! conda env list | grep -q 'mlopt'; then
  echo "Creating conda environment..."
  conda env create -f environment.yml
fi

echo "Activating conda environment..."
conda activate mlopt || { echo "Failed to activate environment"; exit 1; }

# 3. RabbitMQ Setup (running in Docker container)
echo "Starting RabbitMQ..."
docker rm -f rabbitmq 2>/dev/null
docker run -d --network host \
  --name rabbitmq \
  -p 15672:15672 -p 5672:5672 \
  rabbitmq:management

# 4. Dependency Installation
echo "Installing Python requirements..."
pip install -r requirements.txt

# 5. Start Services
echo "Starting master process..."
nohup python run_master.py > master.log 2>&1 &

echo "Starting slave process..."
nohup python run_slave.py > slave.log 2>&1 &

echo "Deployment complete!"
echo "Master log: tail -f master.log"
echo "Slave log: tail -f slave.log"
echo "RabbitMQ GUI: http://<your-ip>:15672 (guest/guest)"