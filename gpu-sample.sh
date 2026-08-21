#!/bin/bash
# Sample GPU utilization + memory for N seconds
DUR=${1:-30}
END=$((SECONDS + DUR))
while [ $SECONDS -lt $END ]; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
    sleep 1
done
