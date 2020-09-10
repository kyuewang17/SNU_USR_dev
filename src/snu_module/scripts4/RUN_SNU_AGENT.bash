#!/usr/bin/env bash

# Make Agent Identification File (use Flags)
while getopts a:i:v: flag
do
  case "${flag}" in
      a) agent=${OPTARG};;
      i) id=${OPTARG};;
      v) version=${OPTARG};;
  esac
done

# Check for Agent Variable
if [ -z "${agent}" ]; then
  echo "Argument 'a' cannot be NULL...!"
  exit 1
else
  if [[ "${agent}" != "FIXED" ]] && [[ "${agent}" != "MOVING" ]]; then
    echo "Argument 'a' must be either 'FIXED' or 'MOVING'...!"
    exit 2
  fi
fi

# Check for ID Variable
if [ -z "${id}" ]; then
  echo "Argument 'i' cannot be NULL...!"
  exit 1
else
  if ! [[ ${id} =~ ^-?[0-9]+$ ]]; then
    echo "Argument 'i' must be an integer type...!"
    exit 3
  fi
fi

# Echo Input Agent Information
echo "--- Checking for Agent Information ---"
echo "Agent: ${agent}";
echo "ID: ${id}"

# Check for Agent Identification File, if does not exist, then generate
agent_file_base_path="/agent"
agent_file_name=${agent}"-0"${id}
if [ ! -d ${agent_file_base_path} ]; then
  echo "Agent Directory Does not Exist...!"
  sleep 0.5
  echo "Generating Agent Directory... (PATH: "${agent_file_base_path}")"
  sudo mkdir ${agent_file_base_path}
fi

# Check if Agent Identification File Exists,
agent_file_path=${agent_file_base_path}"/"${agent_file_name}
if [ ! -f ${agent_file_path} ]; then
  echo "Agent Identification File Does not Exist...!"
  sleep 0.5
  echo "Generating Agent Identification File...! (PATH: "${agent_file_path}")"
  sudo touch ${agent_file_path}
fi

sleep 1

# -------------------- #
# SNU Module Execution #
# -------------------- #
echo "Starting SNU Module...!"

# Set Maximum Iteration Number
max_iter=1000

# Set Iteration Counter
iter_cnt=0

# Iterate when SNU Module Crashes
until [ "$iter_cnt" -ge $max_iter ]
do
  (rosrun snu_module run_osr_snu_module.py -V ${version} agent)
  iter_cnt=$((iter_cnt+1))
  sleep 0.5
done
