#!/bin/bash
# Arguments:
#   $1: binary file to profile
#   $2: output directory

EVENTS_TO_ANALYZE=("cache-misses")
FILE_NAME=$(basename $1)

mkdir -p $2
sudo perf stat -e ${EVENTS_TO_ANALYZE} $1 2> $2/${FILE_NAME}
