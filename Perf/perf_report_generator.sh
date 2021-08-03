#!/bin/bash
# Arguments:
#   $1: binary file to profile
#   $2: output directory
#   $3: Size of image
#   $4: Depth of image
#   $5: Size of kernels
#   $6: Number of kernels
#   $7: Order of for loops
#   $8: Number of tests to do

if [ "$#" -ne 8 ]; then
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Output directory"
    echo "  3-8) Arguments of binary file. See documentation"
    exit 1
fi

# Permissions
# echo "0" | sudo tee /proc/sys/kernel/kptr_restrict
# echo "0" | sudo tee /proc/sys/kernel/perf_event_paranoid

OUTPUT_DIR=$2

EVENTS_TO_ANALYZE=("cache-misses,cache-references,branches,branch-misses,cycles,instructions,L1-dcache-loads-misses,LLC-loads-misses")
FILE_NAME=$(basename $1)_$3_$4_$5_$6_$7_$8.txt
PERF_REPETITIONS=3

mkdir -p ${OUTPUT_DIR}
perf stat -r ${PERF_REPETITIONS} -e ${EVENTS_TO_ANALYZE} $1 2> ${OUTPUT_DIR}/${FILE_NAME} $3 $4 $5 $6 $7 $8
