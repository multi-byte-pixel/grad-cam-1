#!/bin/bash
# Fetch CIFAR-10 from fast S3, verify, then launch the combined run detached.
set -u
cd "$HOME/training" || exit 1
: > "$HOME/prep_status.log"

pkill -f run_tinycnn.py 2>/dev/null
sleep 1
mkdir -p data

echo "downloading cifar from S3 $(date +%T)" >> "$HOME/prep_status.log"
curl -sL -o data/cifar-10-python.tar.gz https://air-example-data.s3.amazonaws.com/cifar-10-python.tar.gz
MD5=$(md5sum data/cifar-10-python.tar.gz | cut -d' ' -f1)
echo "cifar md5=$MD5 $(date +%T)" >> "$HOME/prep_status.log"
if [ "$MD5" != "c58f30108f718f92721af3b95e74349a" ]; then
    echo "MD5 MISMATCH - abort" >> "$HOME/prep_status.log"
    exit 1
fi

echo "cifar OK, launching run" >> "$HOME/prep_status.log"
setsid nohup env MPLBACKEND=Agg PYTHONUNBUFFERED=1 /opt/pytorch/bin/python run_tinycnn.py > "$HOME/gradcam.log" 2>&1 &
echo $! > "$HOME/gradcam.log.pid"
echo "launched pid=$(cat "$HOME/gradcam.log.pid") $(date +%T)" >> "$HOME/prep_status.log"
