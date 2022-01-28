#!/bin/bash

N_REPETITIONS=50
declare -a n_threads=("1" "2" "3" "4" "5" "6" "7" "8")

BINARY_FILE=./bin/test_threadOverheadEmpty

OUTPUT_DIR=./results
mkdir -p ${OUTPUT_DIR}

for N_THREADS in "${n_threads[@]}"
do
    # Create output name
    OUTPUT_NAME=$(basename ${BINARY_FILE})_${N_THREADS}_${N_REPETITIONS}.txt
    # Execute the test and redirect output in a file
    ${BINARY_FILE} ${N_THREADS} ${N_REPETITIONS} | tee ${OUTPUT_DIR}/${OUTPUT_NAME}
    # Sleep 1 second before next run
    sleep 1
done