#!/bin/bash
# MLOptimizer Deployment Script - Python venv edition
# Usage: INSTANCE_HOST_URL="..." INSTANCE_PORT=... INSTANCE_MANAGMENT_URL="..." ./den2.sh

set -e  # Exit on error

echo "========================================"
echo "  MLOptimizer Deployment (venv)"
echo "========================================"

# 1. Repository Setup
REPO_DIR="mloptimizer"
if [ ! -d "$REPO_DIR" ]; then
  echo "Cloning repository..."
  git clone https://github.com/AntonioRamos11/mloptimizer.git $REPO_DIR
fi

cd $REPO_DIR || { echo "Failed to enter repo directory"; exit 1; }
echo "Current directory: $(pwd)"
echo "Pulling latest changes..."
git pull

# 2. Verify requirements file exists
REQ_FILE="requirements2.txt"
if [ ! -f "$REQ_FILE" ]; then
  echo "Error: Missing $REQ_FILE in repository!"
  echo "Please ensure the file exists at: $(pwd)/$REQ_FILE"
  exit 1
fi

# 3. Install system dependencies
echo "Installing system dependencies..."
apt-get update -qq
apt-get install -y python3.10-venv python3-pip || {
  echo "Warning: Could not install python3.10-venv, trying python3-venv..."
  apt-get install -y python3-venv python3-pip
}

# 4. Python Setup - Check for Python 3.10+
PYTHON_CMD=""
for cmd in python3.10 python3.11 python3.12 python3; do
  if command -v $cmd &>/dev/null; then
    version=$($cmd --version 2>&1 | grep -oP '\d+\.\d+')
    if awk "BEGIN {exit !($version >= 3.10)}"; then
      PYTHON_CMD=$cmd
      echo "Found Python $version at $(which $cmd)"
      break
    fi
  fi
done

if [ -z "$PYTHON_CMD" ]; then
  echo "Error: Python 3.10+ is required but not found!"
  echo "Please install Python 3.10 or newer"
  exit 1
fi

# 5. Virtual Environment Setup
VENV_DIR="venv_mlopt"
echo "Checking for virtual environment at $(pwd)/$VENV_DIR..."

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating Python virtual environment..."
  echo "Running: $PYTHON_CMD -m venv $VENV_DIR"
  
  # Create venv with verbose output
  if $PYTHON_CMD -m venv $VENV_DIR; then
    echo "✔ Virtual environment created successfully"
  else
    echo "✘ Failed to create virtual environment"
    echo "Trying alternative method..."
    
    # Try with --without-pip and install pip separately
    $PYTHON_CMD -m venv --without-pip $VENV_DIR
    source $VENV_DIR/bin/activate
    curl https://bootstrap.pypa.io/get-pip.py | python
    deactivate
  fi
else
  echo "✔ Virtual environment already exists"
fi

# Verify venv was created
if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "✘ Error: Virtual environment creation failed!"
  echo "Contents of current directory:"
  ls -la
  echo ""
  echo "Checking if venv module is available..."
  $PYTHON_CMD -m venv --help || echo "venv module not working"
  exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Verify activation
if [ -z "$VIRTUAL_ENV" ]; then
  echo "Error: Failed to activate virtual environment"
  echo "VIRTUAL_ENV variable is empty"
  exit 1
fi

echo "✔ Virtual environment activated: $VIRTUAL_ENV"

# 6. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# 7. Install TensorFlow with GPU support
echo "Installing TensorFlow with GPU support..."
pip install tensorflow[and-cuda]==2.19.0

# 8. Install Python dependencies
echo "Installing Python requirements..."
pip install -r $REQ_FILE

# Verify TensorFlow GPU
echo ""
echo "Verifying TensorFlow GPU installation..."
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}'); print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\"))} GPU(s) detected')"

# 9. Configure System Parameters
echo ""
echo "========================================"
echo "  Configuring System Parameters"
echo "========================================"

# Default to remote RabbitMQ (ngrok/telebit)
INSTANCE_PORT=${INSTANCE_PORT:-19775}
INSTANCE_MANAGMENT_URL=${INSTANCE_MANAGMENT_URL:-"https://selfish-donkey-2.telebit.io"}
INSTANCE_HOST_URL=${INSTANCE_HOST_URL:-"0.tcp.us-cal-1.ngrok.io"}
DATASET_NAME=${DATASET_NAME:-"mnist"}
CLOUD_MODE=${CLOUD_MODE:-1}

# Env vars are passed directly to Python processes below
# No need to patch system_parameters.py - it reads from os.getenv()

# 10. Start Services
echo "========================================"
echo "  Starting Services"
echo "========================================"

# Stop any running processes
echo "Stopping existing processes..."
pkill -f "python -u run_master.py" || true
pkill -f "python -u run_slave.py" || true
sleep 2

# Clean up only specific logs, not optimization logs
echo "Cleaning up old logs..."
mkdir -p logs logs/slave/errors logs/slave/training
rm -f logs/debug.log logs/master.log logs/slave.log logs/slave_gpu*.log

# Set environment variables for debugging
export TF_CPP_MIN_LOG_LEVEL="1"  # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
export PYTHONUNBUFFERED=1  # Ensure Python output isn't buffered
export CLOUD_MODE=$CLOUD_MODE

echo ""
echo "========================================"
echo "  Starting MLOptimizer Master Node"
echo "========================================"
CLOUD_MODE=$CLOUD_MODE INSTANCE_PORT=$INSTANCE_PORT INSTANCE_HOST_URL=$INSTANCE_HOST_URL INSTANCE_MANAGEMENT_URL=$INSTANCE_MANAGMENT_URL DATASET_NAME=$DATASET_NAME CUDA_VISIBLE_DEVICES=0 python -u run_master.py > logs/master.log 2>&1 &
master_pid=$!
echo "Master PID: $master_pid (GPU 0)"
echo "Master logs: tail -f logs/master.log"

sleep 5  # Give master time to initialize

echo ""
echo "========================================"
echo "  Starting MLOptimizer Slave Node(s)"
echo "========================================"

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
  NUM_GPUS=1
  echo "No GPUs detected, using CPU"
fi

echo "Detected $NUM_GPUS GPU(s)"

# Array to store slave PIDs
declare -a slave_pids

# Start one slave per GPU
for ((i=0; i<NUM_GPUS; i++)); do
  echo "Starting slave on GPU $i..."
  CLOUD_MODE=$CLOUD_MODE INSTANCE_PORT=$INSTANCE_PORT INSTANCE_HOST_URL=$INSTANCE_HOST_URL INSTANCE_MANAGMENT_URL=$INSTANCE_MANAGMENT_URL DATASET_NAME=$DATASET_NAME CUDA_VISIBLE_DEVICES=$i python -u run_slave.py >> logs/slave_gpu${i}.log 2>&1 &
  slave_pids[$i]=$!
  echo "  Slave $i PID: ${slave_pids[$i]} (GPU $i)"
done

echo ""
echo "========================================"
echo "  MLOptimizer is Running! 🚀"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Python: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"
echo "  Virtual Env: $(pwd)/$VENV_DIR"
echo "  RabbitMQ Host: ${INSTANCE_HOST_URL}:${INSTANCE_PORT}"
echo "  Management UI: ${INSTANCE_MANAGMENT_URL}"
echo "  Dataset: ${DATASET_NAME}"
echo "  GPUs: $NUM_GPUS"
echo "  Cloud Mode: $CLOUD_MODE"
echo ""
echo "Process IDs:"
echo "  Master: $master_pid (GPU 0)"
for ((i=0; i<NUM_GPUS; i++)); do
  echo "  Slave $i: ${slave_pids[$i]} (GPU $i)"
done
echo ""
echo "Monitor logs:"
echo "  Master:  tail -f logs/master.log"
for ((i=0; i<NUM_GPUS; i++)); do
  echo "  Slave $i: tail -f logs/slave_gpu${i}.log"
done
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{printf "  GPU %s: %s (Util: %s%%, Mem: %s/%s MB)\n", $1, $2, $3, $4, $5}' || echo "  nvidia-smi not available"
echo ""
echo "Stop services:"
echo "  kill $master_pid ${slave_pids[*]}"
echo "  OR: pkill -f 'run_master.py|run_slave.py'"
echo ""
echo "Reactivate environment later:"
echo "  cd $(pwd) && source $VENV_DIR/bin/activate"
echo ""
echo "Setup complete! 🎉"
echo "Each slave runs on its own GPU with OneDeviceStrategy"
echo "GPU utilization should reach 95-100% per GPU"