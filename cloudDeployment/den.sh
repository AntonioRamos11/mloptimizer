  #!/bin/bash
  # test.sh - With Docker installation
"""INSTANCE_HOST_URL="0.tcp.us-cal-1.ngrok.io" \
INSTANCE_PORT=19775 \
INSTANCE_MANAGMENT_URL="https://selfish-donkey-2.telebit.io" \
./den.sh"""
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

  # 3. Conda fSetup 
  export PATH="/root/miniconda/bin:$PATH"
  if ! command -v conda &>/dev/null; then
    echo "Installing Miniconda..."F
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



 

  # 7. Install Python dependencies
  echo "Installing Python requirements..."
  pip install -r requirements2.txt
  conda install tensorflow-gpu -y

  # 8. Configure System Parameters
  echo "Configuring system parameters..."
  
# Default to remote RabbitMQ (ngrok/telebit)
INSTANCE_PORT=${INSTANCE_PORT:-19775}
INSTANCE_MANAGMENT_URL=${INSTANCE_MANAGMENT_URL:-"https://selfish-donkey-2.telebit.io"}
INSTANCE_HOST_URL=${INSTANCE_HOST_URL:-"0.tcp.us-cal-1.ngrok.io"}
DATASET_NAME=${DATASET_NAME:-"mnist"}
MULTI_GPU_MODE=${MULTI_GPU_MODE:-true}

PARAM_FILE="system_parameters.py"

echo "Patching $PARAM_FILE ..."

# Use | as delimiter instead of / to handle URLs with slashes
# Fix any empty DATASET_NAME first
sed -i "s|^    DATASET_NAME: str = \"\"|    DATASET_NAME: str = 'mnist'  # Example dataset|" $PARAM_FILE

# Then apply the actual values
sed -i "s|^    INSTANCE_PORT:.*|    INSTANCE_PORT: int = ${INSTANCE_PORT}|" $PARAM_FILE
sed -i "s|^    INSTANCE_MANAGMENT_URL.*|    INSTANCE_MANAGMENT_URL = \"${INSTANCE_MANAGMENT_URL}\"|" $PARAM_FILE
sed -i "s|^    INSTANCE_HOST_URL:.*|    INSTANCE_HOST_URL: str = '${INSTANCE_HOST_URL}'|" $PARAM_FILE
sed -i "s|^    DATASET_NAME:.*# Example dataset|    DATASET_NAME: str = '${DATASET_NAME}'  # Example dataset|" $PARAM_FILE

echo "✔ Patched system_parameters.py"
echo "  - INSTANCE_PORT: ${INSTANCE_PORT}"
echo "  - INSTANCE_MANAGMENT_URL: ${INSTANCE_MANAGMENT_URL}"
echo "  - INSTANCE_HOST_URL: ${INSTANCE_HOST_URL}"
echo "  - DATASET_NAME: ${DATASET_NAME}"
echo ""
  # 9. Start Services




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
export OPTUNA_WARN_EXPERIMENTAL=1  # Enable Optuna warnings

echo ""
echo "========================================"
echo "  Starting MLOptimizer Master Node"
echo "========================================"
CUDA_VISIBLE_DEVICES=0 python -u run_master.py > logs/master.log 2>&1 &
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
fi

echo "Detected $NUM_GPUS GPU(s)"

# Start one slave per GPU
for ((i=0; i<NUM_GPUS; i++)); do
  echo "Starting slave on GPU $i..."
  CUDA_VISIBLE_DEVICES=$i python -u run_slave.py > logs/slave_gpu${i}.log 2>&1 &
  slave_pids[$i]=$!
  echo "  Slave $i PID: ${slave_pids[$i]} (GPU $i)"
done

echo ""
echo "========================================"
echo "  MLOptimizer is Running!"
echo "========================================"
echo "Configuration:"
echo "  RabbitMQ Host: ${INSTANCE_HOST_URL}:${INSTANCE_PORT}"
echo "  Management UI: ${INSTANCE_MANAGMENT_URL}"
echo "  Dataset: ${DATASET_NAME}"
echo "  GPUs: $NUM_GPUS"
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

echo "Setup complete! 🚀"
echo ""
echo "Each slave runs on its own GPU with OneDeviceStrategy"
echo "GPU utilization should reach 95-100% per GPU"