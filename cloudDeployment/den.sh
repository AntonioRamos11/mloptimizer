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

  # 8. Start Services




# Stop any running processes
echo "Stopping existing processes..."
pkill -f "python -u run_master.py" || true
pkill -f "python  -u run_slave.py" || true
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