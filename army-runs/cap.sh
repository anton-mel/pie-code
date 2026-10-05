#!/bin/zsh
# At 07:00 the llama.cpp team run stops where it is: its per-problem results are already saved, and
# the morning compares only problems both engines finished. The queue then runs the baselines.
until [ "$(date +%H%M)" -ge 0700 ] && [ "$(date +%H%M)" -lt 1200 ]; do sleep 60; done
pkill -f "army.py bench --engine llama --mode team"
