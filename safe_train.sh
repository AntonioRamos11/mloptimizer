#!/bin/bash
#
# Safe launcher for slave training with memory monitoring
# Automatically kills training if memory gets too high
#

echo "=========================================="
echo "SAFE TRAINING LAUNCHER"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if preflight check passes
echo "Running pre-flight check..."
if ! python preflight_check.py; then
    echo -e "${RED}Pre-flight check failed!${NC}"
    echo "Fix the settings before training."
    exit 1
fi

echo ""
echo -e "${GREEN}✓ Pre-flight check passed${NC}"
echo ""

# Ask user to confirm
read -p "Start training with memory monitoring? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Create memory monitor script
cat > /tmp/memory_monitor.sh << 'EOF'
#!/bin/bash
while true; do
    MEM_PERCENT=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    
    if [ $MEM_PERCENT -gt 90 ]; then
        echo ""
        echo "=========================================="
        echo "⚠️  CRITICAL: Memory usage at ${MEM_PERCENT}%!"
        echo "Killing training to prevent freeze..."
        echo "=========================================="
        pkill -f "run_slave.py"
        pkill -f "python.*training"
        exit 1
    elif [ $MEM_PERCENT -gt 80 ]; then
        echo "⚠️  WARNING: Memory at ${MEM_PERCENT}%"
    fi
    
    sleep 5
done
EOF

chmod +x /tmp/memory_monitor.sh

# Start memory monitor in background
/tmp/memory_monitor.sh &
MONITOR_PID=$!

echo ""
echo "=========================================="
echo "Starting training with memory protection"
echo "Memory monitor PID: $MONITOR_PID"
echo "=========================================="
echo ""
echo "Training will auto-stop if memory exceeds 90%"
echo "Press Ctrl+C to stop manually"
echo ""

# Start training
python run_slave.py

# Cleanup
echo ""
echo "Training stopped. Cleaning up..."
kill $MONITOR_PID 2>/dev/null
rm /tmp/memory_monitor.sh 2>/dev/null

echo "Done."
