#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

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