#!/bin/bash

# Check if exactly one argument is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <HW_TAG> (e.g., ps2)"
  exit 1
fi

if [ "$1" = "ps1" ]; then
  HW_DIR="ps1-uninformed-search"
else
  HW_DIR=$1
fi

# exit if build fails
set -e 

docker compose build --build-arg HW_TAG=$1 --build-arg HW_DIR=$HW_DIR
docker compose up