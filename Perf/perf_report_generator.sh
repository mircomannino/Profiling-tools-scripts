#!/bin/bash
# Arguments:
#   $1: binary file to profile
#   $2: output directory

OUTPUT_DIR=./perf_results

EVENTS_TO_ANALYZE=("cache-misses")
FILE_NAME=$(basename $1)

mkdir -p ${OUTPUT_DIR}
sudo perf stat -e ${EVENTS_TO_ANALYZE} $1 2> ${OUTPUT_DIR}/${FILE_NAME}
