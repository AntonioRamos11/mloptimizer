#!/bin/bash
# MLOptimizer - Direct Deployment Script for Vast.ai
# Usage: curl -sSL https://raw.githubusercontent.com/.../deploy.sh | bash -s -- --mode=full
# Or: ./deploy.sh --mode=full

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
MODE="${MODE:-full}"           # full, master, slave, install, check
HOST="${HOST:-}"
PORT="${PORT:-}"
MGMT_URL="${MGMT_URL:-}"
DATASET="${DATASET:-mnist}"
GPU="${GPU:-0}"
CLOUD_MODE="${CLOUD_MODE:-1}"
USE_VENV="${USE_VENV:-yes}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"; shift 2 ;;
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
        --no-venv)
            USE_VENV="no"; shift ;;
        *)
            echo_error "Unknown option: $1"; exit 1 ;;
    esac
done

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
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v $cmd &>/dev/null; then
        version=$($cmd --version 2>&1 | grep -oP '\d+\.\d+' || true)
        if [ -n "$version" ] && [ "$(echo "$version >= 3.10" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
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
        echo_info "Valid modes: full, master, slave, install, check"
        exit 1
        ;;
esac
