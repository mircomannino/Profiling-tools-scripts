#!/bin/bash

# Arguments table
# ===================================================================================================================
# Arg   |       Naive           | MemoryBlocking                | Parallel            | Parallel + Memory blocking  |
# ===================================================================================================================
# $1    |                   binary file to profile                                                                  |
# $2    |                   output directory                                                                        |
# $3    |                   Size of the image (H and W)                                                             |
# $4    |                   Depth of the image                                                                      |
# $5    |                   Size of the kernel (H and W)                                                            |
# $6    |                   Depth of the kernel                                                                     |
# $7    |   Order of loops      | Blocked input channel size    |   Number of threads | Blocked input channel size  |
# $8    |   Number of tests     | Blocked output channel size   |   Order of loops    | Blocked output channel size |
# $9    |                       | Blocked output width size     |   Number of tests   | Blocked output width size   |
# $10   |                       | Order of loops                |                     | Number of threads           |
# $11   |                       | Number of tests               |                     | Order of the loops          |
# $12   |                       |                               |                     | Number of tests             |
# ===================================================================================================================

BINARY_FILE=$1
OUTPUT_DIR=$2

if [[ ${BINARY_FILE} =~ ./bin/benchmark_Parallel[a-zA-Z]+FULL$ ]] && [[ "$#" -ne 6 ]]; then # Enter if name is benchmark_Parallel[NetworkName]FULL
    echo "Detect Parallel + Memory blocking version with wrong arguments"
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Output directory"
    echo "  3-6) Arguements of binary file. See documentation"
    exit 1
elif [[ ${BINARY_FILE} == ./bin/benchmark_ParallelMemoryBlocking ]] && [[ "$#" -ne 12 ]]; then
    echo "Detect Parallel + Memory blocking version with wrong arguments"
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Output directory"
    echo "  3-12) Arguements of binary file. See documentation"
    exit 1
elif [[ ${BINARY_FILE} == ./bin/benchmark_Parallel ]] && [[ "$#" -ne 9 ]]; then
    echo "Detect Parallel version with wrong arguments"
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Output directory"
    echo "  3-9) Arguements of binary file. See documentation"
    exit 1
elif [[ ${BINARY_FILE} =~ ./bin/benchmark_MemoryBlocking ]] && [[ "$#" -ne 11 ]]; then
    echo "Detect Memory blocking version with wrong arguments"
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Output directory"
    echo "  3-11) Arguments of binary file. See documentation"

elif [[ ${BINARY_FILE} =~ ./bin/benchmark_Naive ]] && [[ "$#" -ne 8 ]]; then
    echo "Detect Naive version with wrong arguments"
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Output directory"
    echo "  3-8) Arguments of binary file. See documentation"
    exit 1
fi


# Setup output folder and arguments
if [[ ${BINARY_FILE} =~ ./bin/benchmark_Parallel[a-zA-Z]+FULL$ ]]; then
    OUT_FILE_NAME=$(basename $1)_$3_$4_$5_$6.txt
    ARGUMENTS="$3 $4 $5 $6"
elif [[ ${BINARY_FILE} =~ "./bin/benchmark_ParallelMemoryBlocking" ]]; then # Parallel + Memory blocking
    OUT_FILE_NAME=$(basename $1)_$3_$4_$5_$6_$7_$8_$9_${10}_${11}_${12}.txt
    ARGUMENTS="$3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}"
elif [[ ${BINARY_FILE} =~ "./bin/benchmark_Parallel" ]]; then # Parallel
    OUT_FILE_NAME=$(basename $1)_$3_$4_$5_$6_$7_$8_$9.txt
    ARGUMENTS="$3 $4 $5 $6 $7 $8 $9"
elif [[ ${BINARY_FILE} =~ "./bin/benchmark_MemoryBlocking" ]]; then # MemoryBlocking
    OUT_FILE_NAME=$(basename $1)_$3_$4_$5_$6_$7_$8_$9_${10}_${11}.txt
    ARGUMENTS="$3 $4 $5 $6 $7 $8 $9 ${10} ${11}"
elif [[ ${BINARY_FILE} =~ "./bin/benchmark_Naive" ]]; then # Naive
    OUT_FILE_NAME=$(basename $1)_$3_$4_$5_$6_$7_$8.txt
    ARGUMENTS="$3 $4 $5 $6 $7 $8"
fi

# Run the execution
mkdir -p ${OUTPUT_DIR}
${BINARY_FILE} ${ARGUMENTS} | tee ${OUTPUT_DIR}/${OUT_FILE_NAME}
