#!/bin/bash
echo "Watching phase 1 (extend_multiseed_to_10.py, PID 337)..."
while true; do
  if grep -q "DONE -- n=10 results" logs/extend_multiseed.log 2>/dev/null; then
    echo "PHASE1_SUCCESS: multiseed extension to n=10 complete"
    break
  fi
  if ! tasklist //FI "PID eq 337" 2>/dev/null | grep -q "337"; then
    echo "PHASE1_FAILED: process 337 exited without completion marker -- check logs/extend_multiseed.log"
    exit 1
  fi
  sleep 30
done

echo "Starting phase 2 (attack magnitude sweep)..."
nohup .venv/Scripts/python.exe scripts/run_attack_magnitude_sweep.py > logs/attack_magnitude_sweep.log 2>&1 &
sweep_pid=$!
echo "PHASE2_STARTED pid=$sweep_pid"

while true; do
  if grep -q "Saved.*points to" logs/attack_magnitude_sweep.log 2>/dev/null; then
    echo "PHASE2_SUCCESS: attack magnitude sweep complete"
    break
  fi
  if ! tasklist //FI "PID eq $sweep_pid" 2>/dev/null | grep -q "$sweep_pid"; then
    echo "PHASE2_FAILED: process $sweep_pid exited without completion marker -- check logs/attack_magnitude_sweep.log"
    exit 1
  fi
  sleep 30
done

echo "ALL_DONE: both partner's (n=10 multiseed) and professor's (attack magnitude sweep) requests complete"
