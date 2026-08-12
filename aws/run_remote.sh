#!/bin/bash
# Relaunch the Grad-CAM run, unbuffered and detached.
pkill -f run_tinycnn.py 2>/dev/null
sleep 2
cd "$HOME/training" || exit 1
setsid nohup env MPLBACKEND=Agg PYTHONUNBUFFERED=1 /opt/pytorch/bin/python run_tinycnn.py > "$HOME/gradcam.log" 2>&1 &
echo $! > "$HOME/gradcam.log.pid"
sleep 3
echo "PID=$(cat "$HOME/gradcam.log.pid")"
