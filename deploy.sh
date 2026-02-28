#!/bin/bash
# MLOptimizer - Direct Deployment Script for Vast.ai / Cloud Containers
#
# =====================================================
# FIRST RUN IN NEW CONTAINER - RECOMMENDED COMMAND:
# =====================================================
# cd /workspace && rm -rf mloptimizer && \
# REPO_URL="https://github.com/AntonioRamos11/mloptimizer.git" \
# MODE=cloud ./deploy.sh \
#   --host "YOUR_NGROK_HOST" --port NGROK_PORT --mgmt-url "YOUR_MGMT_URL"
#
# Example with current settings:
# cd /workspace && rm -rf mloptimizer && \
# REPO_URL="https://github.com/AntonioRamos11/mloptimizer.git" \
# MODE=cloud ./deploy.sh \
#   --host "8.tcp.us-cal-1.ngrok.io" --port 10147 --mgmt-url "https://rubber-friend-nose-balance.trycloudflare.com"
#
# Options:
#   --mode cloud          : Clone repo, install deps, run training (master + slave)
#   --mode cloud-master  : Run only master (for separate computer)
#   --mode cloud-slave   : Run only slave(s) (for separate computer)
#   --mode resolve       : Only resolve and install dependencies
#   --mode install       : Only install requirements
#   --save-config yes    : Save ngrok settings as defaults
# =====================================================

#cd /workspace



set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo_info() { echo -e "${BLUE}ℹ${NC} $1"; }
echo_ok() { echo -e "${GREEN}✓${NC} $1"; }
echo_warn() { echo -e "${YELLOW}⚠${NC} $1"; }
echo_error() { echo -e "${RED}✗${NC} $1"; }

# Default settings
MODE="${MODE:-full}"           # full, master, slave, cloud, cloud-master, cloud-slave, install, check, resolve
HOST="${HOST:-localhost}"
PORT="${PORT:-5555}"
MGMT_URL="${MGMT_URL:-http://localhost:15672}"
DATASET="${DATASET:-mnist}"
GPU="${GPU:-0}"
CLOUD_MODE="${CLOUD_MODE:-1}"
USE_VENV="${USE_VENV:-yes}"
REPO_URL="${REPO_URL:-}"
SAVE_CONFIG="${SAVE_CONFIG:-no}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"; shift 2 ;;
        --repo)
            REPO_URL="$2"; shift 2 ;;
        --host)
            HOST="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --mgmt-url)
            MGMT_URL="$2"; shift 2 ;;
        --dataset)
            DATASET="$2"; shift 2 ;;
        --gpu)
            GPU="$2"; shift 2 ;;
        --cloud-mode)
            CLOUD_MODE="$2"; shift 2 ;;
        --save-config)
            SAVE_CONFIG="$2"; shift 2 ;;
        --no-venv)
            USE_VENV="no"; shift ;;
        *)
            echo_error "Unknown option: $1"; exit 1 ;;
    esac
done

# Clone repo if specified and not already in project directory
if [ -n "$REPO_URL" ]; then
    if [ -d "/workspace/mloptimizer/.git" ]; then
        echo_info "Updating repo..."
        cd /workspace/mloptimizer
        git pull
    else
        echo_info "Cloning repo..."
        rm -rf /workspace/mloptimizer
        git clone "$REPO_URL" /workspace/mloptimizer
        cd /workspace/mloptimizer
    fi
    SCRIPT_DIR="/workspace/mloptimizer"
fi

echo "========================================"
echo "  MLOptimizer - Vast.ai Deployer"
echo "========================================"
echo_info "Mode: $MODE"
echo_info "Dataset: $DATASET"
echo_info "Cloud Mode: $CLOUD_MODE"

# Find Python
echo ""
echo_info "Finding Python..."
PYTHON_CMD=""
for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v $cmd &>/dev/null; then
        version=$($cmd --version 2>&1 | grep -oP '\d+\.\d+' || true)
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ -n "$version" ] && [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD=$cmd
            echo_ok "Found Python $version: $cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo_error "Python 3.10+ not found!"
    exit 1
fi

# Get project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Install mode - just install requirements
if [ "$MODE" = "install" ]; then
    echo ""
    echo_info "Installing requirements..."
    
    # Create venv if needed
    if [ "$USE_VENV" = "yes" ] && [ ! -d "venv_mlopt" ]; then
        echo_info "Creating virtual environment..."
        $PYTHON_CMD -m venv venv_mlopt
    fi
    
    if [ -d "venv_mlopt" ]; then
        echo_ok "Activating virtual environment..."
        source venv_mlopt/bin/activate
    fi
    
    echo_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    echo_info "Installing requirements..."
    for req_file in requirements.txt requirements2.txt; do
        if [ -f "$req_file" ]; then
            echo_info "Installing from $req_file..."
            pip install -r "$req_file" || true
        fi
    done
    
    echo_ok "Installation complete!"
    exit 0
fi

# Check mode - just validate
if [ "$MODE" = "check" ]; then
    echo ""
    echo_info "Checking requirements..."
    
    source venv_mlopt/bin/activate 2>/dev/null || true
    
    for req_file in requirements.txt requirements2.txt; do
        if [ -f "$req_file" ]; then
            echo_info "Checking $req_file..."
            while IFS= read -r line; do
                [ -z "$line" ] && continue
                [[ "$line" =~ ^# ]] && continue
                pkg=$(echo "$line" | cut -d'=' -f1 | cut -d'<' -f1 | cut -d'>' -f1)
                if pip show "$pkg" &>/dev/null; then
                    echo_ok "  $pkg installed"
                else
                    echo_warn "  $pkg NOT installed"
                fi
            done < "$req_file"
        fi
    done
    
    echo_ok "Check complete!"
    exit 0
fi

# Resolve mode - generate locked requirements.txt using pip-compile
if [ "$MODE" = "resolve" ]; then
    echo ""
    echo_info "Resolving dependencies with pip-compile..."
    
    # Create venv if needed
    if [ "$USE_VENV" = "yes" ] && [ ! -d "venv_mlopt" ]; then
        echo_info "Creating virtual environment..."
        $PYTHON_CMD -m venv venv_mlopt
    fi
    
    if [ -d "venv_mlopt" ]; then
        echo_ok "Activating virtual environment..."
        source venv_mlopt/bin/activate
    fi
    
    echo_info "Upgrading pip and installing pip-tools..."
    pip install --upgrade pip setuptools wheel
    pip install pip-tools
    
    if [ ! -f "requirements.in" ]; then
        echo_error "requirements.in not found!"
        exit 1
    fi
    
    echo_info "Running pip to resolve and install dependencies..."
    # Install with --only-binary to avoid building from source (requires gfortran, etc.)
    pip install --only-binary=:all: -r requirements.in || pip install -r requirements.in
    
    echo_info "Verifying dependencies..."
    pip check
    
    echo_ok "Dependency resolution complete!"
    echo_info "Generated requirements.txt with locked versions"
    exit 0
fi

# Cloud mode - clone, resolve, and run training
if [ "$MODE" = "cloud" ]; then
    echo ""
    echo_info "Running in cloud mode..."
    
    # Kill existing mloptimizer processes before starting fresh
    echo_info "Cleaning up existing MLOptimizer processes..."

    PIDS=$(ps -eo pid,cmd | grep -E "run.py|run_master.py|run_slave.py" | grep -v grep | awk '{print $1}')

    if [ -n "$PIDS" ]; then
        echo_info "Found processes: $PIDS"

        for PID in $PIDS; do
            # kill children first
            pkill -TERM -P $PID 2>/dev/null || true
            kill -TERM $PID 2>/dev/null || true
        done

        sleep 3

        # force kill if still alive
        for PID in $PIDS; do
            if ps -p $PID > /dev/null 2>&1; then
                echo_warn "Force killing $PID"
                pkill -KILL -P $PID 2>/dev/null || true
                kill -KILL $PID 2>/dev/null || true
            fi
        done

        echo_ok "Old ML processes terminated"
    else
        echo_info "No existing ML processes found"
    fi
    
    # Create venv if needed
    if [ "$USE_VENV" = "yes" ] && [ ! -d "venv_mlopt" ]; then
        echo_info "Creating virtual environment..."
        $PYTHON_CMD -m venv venv_mlopt
    fi
    
    if [ -d "venv_mlopt" ]; then
        echo_ok "Activating virtual environment..."
        source venv_mlopt/bin/activate
    fi
    
    echo_info "Upgrading pip and installing pip-tools..."
    pip install --upgrade pip setuptools wheel
    pip install pip-tools
    
    if [ -f "requirements.in" ]; then
        echo_info "Installing dependencies from requirements.in..."
        pip install --only-binary=:all: -r requirements.in || pip install -r requirements.in
        pip check
        
        echo_info "Generating requirements2.txt (full freeze)..."
        pip freeze > requirements2.txt
        echo_ok "Generated requirements2.txt"
    else
        echo_warn "requirements.in not found, skipping dependency installation"
    fi
    
    echo_ok "Dependencies ready!"
    echo_info "Starting ML training in full mode..."
    
    # Create log directories
    mkdir -p logs logs/slave/errors logs/slave/training logs/master
    
    # Detect number of GPUs
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -eq 0 ] || [ -z "$NUM_GPUS" ]; then
        NUM_GPUS=1
        echo_warn "No GPUs detected, using CPU mode"
    fi
    echo_info "Detected $NUM_GPUS GPU(s)"
    
    # Export common environment variables
    export CLOUD_MODE=1
    export TF_CPP_MIN_LOG_LEVEL=1
    export PYTHONUNBUFFERED=1
    
    [ -n "$HOST" ] && export INSTANCE_HOST_URL="$HOST"
    [ -n "$PORT" ] && export INSTANCE_PORT="$PORT"
    [ -n "$MGMT_URL" ] && export INSTANCE_MANAGMENT_URL="$MGMT_URL"
    [ -n "$DATASET" ] && export DATASET_NAME="$DATASET"
    
    # Save config to system_parameters.py if requested
    if [ "$SAVE_CONFIG" = "yes" ] || [ "$SAVE_CONFIG" = "true" ]; then
        echo_info "Saving ngrok config to system_parameters.py..."
        if [ -f "system_parameters.py" ]; then
            sed -i "s|REMOTE_HOST_URL: str = \".*\"|REMOTE_HOST_URL: str = \"$HOST\"|" system_parameters.py
            sed -i "s|REMOTE_PORT: int = .*|REMOTE_PORT: int = $PORT|" system_parameters.py
            sed -i "s|REMOTE_MANAGEMENT_URL: str = \".*\"|REMOTE_MANAGEMENT_URL: str = \"$MGMT_URL\"|" system_parameters.py
            echo_ok "Config saved! Default ngrok URL is now: $HOST:$PORT"
        else
            echo_warn "system_parameters.py not found, skipping config save"
        fi
    fi
    
    echo_info "Logs will be saved to logs/ directory"
    echo_info "Use 'tail -f logs/master.log' or 'tail -f logs/slave_gpu*.log' to monitor"
    
    # Array to store process PIDs
    declare -a slave_pids
    
    # Start master on GPU 0
    echo_info "Starting Master on GPU 0..."
    export CUDA_VISIBLE_DEVICES=0
    python run.py --master \
        --host "$HOST" --port "$PORT" --mgmt-url "$MGMT_URL" \
        --dataset "$DATASET" --cloud-mode "$CLOUD_MODE" \
        > logs/master.log 2>&1 &
    MASTER_PID=$!
    echo_ok "Master started (PID: $MASTER_PID, GPU 0)"
    
    sleep 3
    
    # Start one slave per GPU
    for ((i=0; i<NUM_GPUS; i++)); do
        echo_info "Starting Slave on GPU $i..."
        export CUDA_VISIBLE_DEVICES=$i
        export TF_DATA_DIR="/tmp/tensorflow_datasets_gpu${i}"
        mkdir -p "$TF_DATA_DIR"
        python run.py --slave \
            --host "$HOST" --port "$PORT" --mgmt-url "$MGMT_URL" \
            --dataset "$DATASET" --cloud-mode "$CLOUD_MODE" --gpu $i \
            > logs/slave_gpu${i}.log 2>&1 &
        slave_pids[$i]=$!
        echo_ok "Slave $i started (PID: ${slave_pids[$i]}, GPU $i)"
    done
    
    echo ""
    echo_ok "========================================="
    echo_ok "  MLOptimizer Running with $NUM_GPUS GPU(s)"
    echo_ok "========================================="
    echo ""
    echo_info "Process IDs:"
    echo_info "  Master:  $MASTER_PID (GPU 0)"
    for ((i=0; i<NUM_GPUS; i++)); do
        echo_info "  Slave $i: ${slave_pids[$i]} (GPU $i)"
    done
    echo ""
    echo_info "Log files:"
    echo_info "  Master:  logs/master.log"
    for ((i=0; i<NUM_GPUS; i++)); do
        echo_info "  Slave $i: logs/slave_gpu${i}.log"
    done
    echo ""
    echo_info "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | awk -F', ' '{printf "    GPU %s: %s (%s/%s MB, Util: %s)\n", $1, $2, $3, $4, $5}' || echo "    nvidia-smi not available"
    echo ""
    echo_info "To monitor: tail -f logs/master.log logs/slave_gpu*.log"
    
    # Cleanup function to kill all processes
    cleanup() {
        echo ""
        echo_warn "Shutting down MLOptimizer..."
        kill $MASTER_PID 2>/dev/null || true
        for pid in "${slave_pids[@]}"; do
            kill $pid 2>/dev/null || true
        done
        echo_ok "All processes terminated"
    }
    
    # Trap signals to cleanup on exit
    trap cleanup SIGINT SIGTERM EXIT
    
    # Wait for master process
    wait $MASTER_PID
    
    exit 0
fi

# Cloud Master mode - run only master (for separate computer)
if [ "$MODE" = "cloud-master" ]; then
    echo ""
    echo_info "Running in cloud-master mode..."
    
    # Kill existing master processes
    echo_info "Cleaning up existing master processes..."
    PIDS=$(ps -eo pid,cmd | grep -E "run.py.*master|run_master.py" | grep -v grep | awk '{print $1}')
    if [ -n "$PIDS" ]; then
        for PID in $PIDS; do
            pkill -TERM -P $PID 2>/dev/null || true
            kill -TERM $PID 2>/dev/null || true
        done
        sleep 2
    fi
    
    # Create venv if needed
    if [ "$USE_VENV" = "yes" ] && [ ! -d "venv_mlopt" ]; then
        echo_info "Creating virtual environment..."
        $PYTHON_CMD -m venv venv_mlopt
    fi
    
    if [ -d "venv_mlopt" ]; then
        source venv_mlopt/bin/activate
    fi
    
    pip install --upgrade pip setuptools wheel 2>/dev/null
    
    if [ -f "requirements.in" ]; then
        pip install --only-binary=:all: -r requirements.in 2>/dev/null || pip install -r requirements.in 2>/dev/null || true
    fi
    
    mkdir -p logs/master
    
    export CLOUD_MODE=1
    export TF_CPP_MIN_LOG_LEVEL=1
    export PYTHONUNBUFFERED=1
    export CUDA_VISIBLE_DEVICES=0
    
    [ -n "$HOST" ] && export INSTANCE_HOST_URL="$HOST"
    [ -n "$PORT" ] && export INSTANCE_PORT="$PORT"
    [ -n "$MGMT_URL" ] && export INSTANCE_MANAGMENT_URL="$MGMT_URL"
    [ -n "$DATASET" ] && export DATASET_NAME="$DATASET"
    
    echo_info "Starting Master only..."
    python run.py --master \
        --host "$HOST" --port "$PORT" --mgmt-url "$MGMT_URL" \
        --dataset "$DATASET" --cloud-mode "$CLOUD_MODE" \
        > logs/master.log 2>&1 &
    MASTER_PID=$!
    echo_ok "Master started (PID: $MASTER_PID)"
    echo_info "Log: logs/master.log"
    
    wait $MASTER_PID
    exit 0
fi

# Cloud Slave mode - run only slave (for separate computer)
if [ "$MODE" = "cloud-slave" ]; then
    echo ""
    echo_info "Running in cloud-slave mode..."
    
    # Kill existing slave processes
    echo_info "Cleaning up existing slave processes..."
    PIDS=$(ps -eo pid,cmd | grep -E "run.py.*slave|run_slave.py" | grep -v grep | awk '{print $1}')
    if [ -n "$PIDS" ]; then
        for PID in $PIDS; do
            pkill -TERM -P $PID 2>/dev/null || true
            kill -TERM $PID 2>/dev/null || true
        done
        sleep 2
    fi
    
    # Create venv if needed
    if [ "$USE_VENV" = "yes" ] && [ ! -d "venv_mlopt" ]; then
        echo_info "Creating virtual environment..."
        $PYTHON_CMD -m venv venv_mlopt
    fi
    
    if [ -d "venv_mlopt" ]; then
        source venv_mlopt/bin/activate
    fi
    
    pip install --upgrade pip setuptools wheel 2>/dev/null
    
    if [ -f "requirements.in" ]; then
        pip install --only-binary=:all: -r requirements.in 2>/dev/null || pip install -r requirements.in 2>/dev/null || true
    fi
    
    mkdir -p logs/slave/errors logs/slave/training
    
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ "$NUM_GPUS" -eq 0 ] || [ -z "$NUM_GPUS" ]; then
        NUM_GPUS=1
    fi
    echo_info "Detected $NUM_GPUS GPU(s)"
    
    export CLOUD_MODE=1
    export TF_CPP_MIN_LOG_LEVEL=1
    export PYTHONUNBUFFERED=1
    
    [ -n "$HOST" ] && export INSTANCE_HOST_URL="$HOST"
    [ -n "$PORT" ] && export INSTANCE_PORT="$PORT"
    [ -n "$MGMT_URL" ] && export INSTANCE_MANAGMENT_URL="$MGMT_URL"
    [ -n "$DATASET" ] && export DATASET_NAME="$DATASET"
    
    declare -a slave_pids
    
    for ((i=0; i<NUM_GPUS; i++)); do
        echo_info "Starting Slave on GPU $i..."
        export CUDA_VISIBLE_DEVICES=$i
        export TF_DATA_DIR="/tmp/tensorflow_datasets_gpu${i}"
        mkdir -p "$TF_DATA_DIR"
        python run.py --slave \
            --host "$HOST" --port "$PORT" --mgmt-url "$MGMT_URL" \
            --dataset "$DATASET" --cloud-mode "$CLOUD_MODE" --gpu $i \
            > logs/slave_gpu${i}.log 2>&1 &
        slave_pids[$i]=$!
        echo_ok "Slave $i started (PID: ${slave_pids[$i]}, GPU $i)"
    done
    
    echo_ok "All slaves started"
    echo_info "Logs: logs/slave_gpu*.log"
    
    for pid in "${slave_pids[@]}"; do
        wait $pid
    done
    
    exit 0
fi

# Build environment
export VIRTUAL_ENV="$SCRIPT_DIR/venv_mlopt"

if [ -d "$VIRTUAL_ENV" ]; then
    echo_info "Activating virtual environment..."
    source "$VIRTUAL_ENV/bin/activate"
else
    echo_warn "Virtual environment not found, using system Python"
fi

# Export environment variables
export CLOUD_MODE
export TF_CPP_MIN_LOG_LEVEL=1
export PYTHONUNBUFFERED=1

[ -n "$HOST" ] && export INSTANCE_HOST_URL="$HOST"
[ -n "$PORT" ] && export INSTANCE_PORT="$PORT"
[ -n "$MGMT_URL" ] && export INSTANCE_MANAGMENT_URL="$MGMT_URL"
[ -n "$DATASET" ] && export DATASET_NAME="$DATASET"

# Run the Python runner
echo ""
echo_info "Starting MLOptimizer..."

case "$MODE" in
    full)
        echo_info "Running full mode (master + slave)..."
        export CUDA_VISIBLE_DEVICES="$GPU"
        python run.py --master --slave \
            --host "$HOST" --port "$PORT" --mgmt-url "$MGMT_URL" \
            --dataset "$DATASET" --cloud-mode "$CLOUD_MODE"
        ;;
    master)
        echo_info "Running master only..."
        export CUDA_VISIBLE_DEVICES=0
        python run.py --master \
            --host "$HOST" --port "$PORT" --mgmt-url "$MGMT_URL" \
            --dataset "$DATASET" --cloud-mode "$CLOUD_MODE"
        ;;
    slave)
        echo_info "Running slave only..."
        export CUDA_VISIBLE_DEVICES="$GPU"
        python run.py --slave --gpu "$GPU" \
            --host "$HOST" --port "$PORT" --mgmt-url "$MGMT_URL" \
            --dataset "$DATASET" --cloud-mode "$CLOUD_MODE"
        ;;
    *)
        echo_error "Unknown mode: $MODE"
        echo_info "Valid modes: full, master, slave, cloud, cloud-master, cloud-slave, install, check, resolve, cloud"
        exit 1
        ;;
esac
