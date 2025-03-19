#!/bin/bash
mkdir -p results  

echo "Starting EZ Diffusion Model Simulate-and-Recover Experiment..."

PYTHON_EXEC="/workspace/myenv/bin/python"  

for N in 10 40 4000; do
    $PYTHON_EXEC ./simulate_recover.py --n $N --iterations 1000
done

echo "Simulation complete. Results saved in results/ directory."


