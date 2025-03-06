#!/bin/bash
# test.sh - Final corrected version

# 1. Repository Setup
REPO_DIR="mloptimizer"
if [ ! -d "$REPO_DIR" ]; then
  echo "Cloning repository..."
  git clone https://github.com/AntonioRamos11/mloptimizer.git $REPO_DIR
fi

cd $REPO_DIR || { echo "Failed to enter repo directory"; exit 1; }
git pull

# 2. Verify environment.yml exists
ENV_FILE="environment.yml"
if [ ! -f "$ENV_FILE" ]; then
  echo "Error: Missing $ENV_FILE in repository!"
  echo "Please ensure the file exists at: $(pwd)/$ENV_FILE"
  exit 1
fi

# 3. Conda Setup
export PATH="/root/miniconda/bin:$PATH"
if ! command -v conda &>/dev/null; then
  echo "Installing Miniconda..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p /root/miniconda
  rm miniconda.sh
  # Initialize conda for current shell
  source /root/miniconda/etc/profile.d/conda.sh
  conda init bash
fi

# Initialize conda properly
source /root/miniconda/etc/profile.d/conda.sh

# 4. Environment Setup
ENV_NAME="mlopt"
if ! conda env list | grep -q $ENV_NAME; then
  echo "Creating conda environment from $ENV_FILE..."
  conda env create -f $ENV_FILE
fi

echo "Activating conda environment..."
conda activate $ENV_NAME || { echo "Failed to activate environment"; exit 1; }

# 5. RabbitMQ Setup
echo "Starting RabbitMQ..."
docker rm -f rabbitmq 2>/dev/null
docker run -d --network host \
  --name rabbitmq \
  -p 15672:15672 -p 5672:5672 \
  rabbitmq:management

# 6. Install Python dependencies
echo "Installing Python requirements..."
pip install -r requirements.txt

# 7. Start Services
echo "Starting processes..."
pkill -f "python run_master.py"
pkill -f "python run_slave.py"
nohup python run_master.py > master.log 2>&1 &
nohup python run_slave.py > slave.log 2>&1 &

echo "Deployment successful!"
echo "Master log: tail -f $REPO_DIR/master.log"
echo "RabbitMQ GUI: http://$(curl -s ifconfig.me):15672 (guest/guest)"