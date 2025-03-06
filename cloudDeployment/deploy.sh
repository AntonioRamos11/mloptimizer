#!/bin/bash
# test.sh - With Docker installation

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
  source /root/miniconda/etc/profile.d/conda.sh
  conda init bash
fi
source /root/miniconda/etc/profile.d/conda.sh

# 4. Environment Setup
ENV_NAME="mlopt"
if ! conda env list | grep -q $ENV_NAME; then
  echo "Creating conda environment from $ENV_FILE..."
  conda env create -f $ENV_FILE
fi
conda activate $ENV_NAME || { echo "Failed to activate environment"; exit 1; }

# 5. Docker Installation
if ! command -v docker &>/dev/null; then
  echo "Installing Docker..."
  apt-get update -qq
  apt-get install -qq -y \
    ca-certificates \
    curl \
    gnupg

  # Add Docker's official GPG key
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg

  # Set up the repository
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

  # Install Docker Engine
  apt-get update -qq
  apt-get install -qq -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

  # Verify Docker installation
  docker --version || { echo "Docker installation failed"; exit 1; }
fi

# 6. RabbitMQ Setup
echo "Starting RabbitMQ..."
docker rm -f rabbitmq 2>/dev/null
docker run -d --network host \
  --name rabbitmq \
  -p 15672:15672 -p 5672:5672 \
  rabbitmq:management

# 7. Install Python dependencies
echo "Installing Python requirements..."
pip install -r requirements.txt

# 8. Start Services
echo "Starting processes..."
pkill -f "python run_master.py"
pkill -f "python run_slave.py"
nohup python run_master.py > master.log 2>&1 &
nohup python run_slave.py > slave.log 2>&1 &

echo "Deployment successful!"
echo "Master log: tail -f $REPO_DIR/master.log"
echo "RabbitMQ GUI: http://$(curl -s ifconfig.me):15672 (guest/guest)"