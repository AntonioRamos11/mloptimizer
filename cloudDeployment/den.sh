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



 

  # 7. Install Python dependencies
  echo "Installing Python requirements..."
  pip install -r requirements2.txt
  conda install tensorflow-gpu -y

  # 8. Configure System Parameters
  echo "Configuring system parameters..."
  
  # Environment variables with defaults
  INSTANCE_PORT=${INSTANCE_PORT:-5672}
  INSTANCE_MANAGMENT_URL=${INSTANCE_MANAGMENT_URL:-"localhost"}
  INSTANCE_HOST_URL=${INSTANCE_HOST_URL:-"localhost"}
  DATASET_NAME=${DATASET_NAME:-"mnist"}
  MULTI_GPU_MODE=${MULTI_GPU_MODE:-true}
  
  PARAM_FILE="system_parameters.py"
  
  echo "Patching $PARAM_FILE ..."
  
  sed -i "s/^    INSTANCE_PORT:.*/    INSTANCE_PORT: int = ${INSTANCE_PORT}/" $PARAM_FILE
  sed -i "s/^    INSTANCE_MANAGMENT_URL.*/    INSTANCE_MANAGMENT_URL = \"${INSTANCE_MANAGMENT_URL}\"/" $PARAM_FILE
  sed -i "s/^    INSTANCE_HOST_URL:.*/    INSTANCE_HOST_URL: str = \"${INSTANCE_HOST_URL}\"/" $PARAM_FILE
  sed -i "s/^    DATASET_NAME:.*/    DATASET_NAME: str = \"${DATASET_NAME}\"/" $PARAM_FILE
  
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
rm -f logs/debug.log logs/master.log logs/slave.log
# REMOVED: Don't delete optimization_job.log

# Set environment variables for debugging
export TF_CPP_MIN_LOG_LEVEL="1"  # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
export PYTHONUNBUFFERED=1  # Ensure Python output isn't buffered
export OPTUNA_WARN_EXPERIMENTAL=1  # Enable Optuna warnings

echo "Starting master process with debugging..."
mkdir -p logs
python -u run_master.py > logs/master.log 2>&1 &
master_pid=$!
echo "Master PID: $master_pid"

sleep 5  # Give master time to initialize

echo "Starting slave process with debugging..."
python -u run_slave.py > logs/slave.log 2>&1 &
slave_pid=$!
echo "Slave PID: $slave_pid"

echo "Starting process monitor..."
./debug_monitor.py

# End of script