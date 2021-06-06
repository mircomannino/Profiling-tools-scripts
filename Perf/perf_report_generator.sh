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

OUTPUT_DIR=$2

EVENTS_TO_ANALYZE=("cache-misses")
FILE_NAME=$(basename $1)_$3_$4_$5_$6_$7_$8.txt

mkdir -p ${OUTPUT_DIR}
sudo perf stat -e ${EVENTS_TO_ANALYZE} $1 2> ${OUTPUT_DIR}/${FILE_NAME} $3 $4 $5 $6 $7 $8
