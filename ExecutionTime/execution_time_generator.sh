#!/bin/bash
# Arguments:
#   $1: binary file to profile
#   $2: output directory
#   $3: Size of image
#   $4: Depth of image
#   $5: Size of kernels
#   $6: Number of kernels
#   $7: Blocked input channel size
#   $8: Blocked output channel size
#   $9: Blocked output width size
#   $10[$7]: Order of for loops
#   $11[$8]: Number of tests to do

if [[ ${BINARY_FILE} =~ "./bin/benchmark_Naive" ]] && [[ "$#" -ne 8 ]]; then
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Output directory"
    echo "  3-8) Arguments of binary file. See documentation"
    exit 1
fi
if [[ ${BINARY_FILE} = "./bin/benchmark_MemoryBlocking" ]] && [[ "$#" -ne 11 ]]; then
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Output directory"
    echo "  3-11) Arguments of binary file. See documentation"
    exit 1
fi


BINARY_FILE=$1
OUTPUT_DIR=$2

if [[ ${BINARY_FILE} =~ "./bin/benchmark_Naive" ]]; then # Naive
    FILE_NAME=$(basename $1)_$3_$4_$5_$6_$7_$8.txt
fi
if [[ ${BINARY_FILE} = "./bin/benchmark_MemoryBlocking" ]]; then # MemoryBlocking
    FILE_NAME=$(basename $1)_$3_$4_$5_$6_$7_$8_$9_${10}_${11}.txt
fi


mkdir -p ${OUTPUT_DIR}
if [[ ${BINARY_FILE} =~ "./bin/benchmark_Naive" ]]; then
    ${BINARY_FILE} $3 $4 $5 $6 $7 $8 | tee ${OUTPUT_DIR}/${FILE_NAME}
fi
if [ ${BINARY_FILE} = "./bin/benchmark_AlexNet" ]; then
    ${BINARY_FILE} $7 $8 | tee ${OUTPUT_DIR}/${FILE_NAME}
fi
if [ ${BINARY_FILE} = "./bin/benchmark_Compilers" ]; then
    ${BINARY_FILE} $3 $4 $5 $6 $7 $8 | tee ${OUTPUT_DIR}/${FILE_NAME}
fi
if [ ${BINARY_FILE} = "./bin/benchmark_MemoryBlocking" ]; then
    ${BINARY_FILE} $3 $4 $5 $6 $7 $8 $9 ${10} ${11} | tee ${OUTPUT_DIR}/${FILE_NAME}
fi
