#!/usr/bin/env bash

# Set Maximum Iteration Number
max_iter=1000

# Set Iteration Counter
iter_cnt=0

# Iterate when SNU Module Crashes
until [ "$iter_cnt" -ge $max_iter ]
do
  (rosrun snu_module run_osr_snu_module.py agent)
  iter_cnt=$((iter_cnt+1))
  sleep 0.1
done
