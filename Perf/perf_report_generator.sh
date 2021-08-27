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

OUTPUT_DIR=$2
BINARY_FILE=$1

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

# Permissions
# echo "0" | sudo tee /proc/sys/kernel/kptr_restrict
# echo "0" | sudo tee /proc/sys/kernel/perf_event_paranoid

EVENTS_TO_ANALYZE=("cache-misses,cache-references,branches,branch-misses,cycles,instructions,L1-dcache-loads-misses,LLC-loads-misses,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_single")
PERF_REPETITIONS=3


if [[ ${BINARY_FILE} = "./bin/benchmark_MemoryBlocking" ]]; then # MemoryBlocking
    FILE_NAME=$(basename $1)_$3_$4_$5_$6_$7_$8_$9_${10}_${11}.txt
else
    FILE_NAME=$(basename $1)_$3_$4_$5_$6_$7_$8.txt
fi

mkdir -p ${OUTPUT_DIR}
if [[ ${BINARY_FILE} = "./bin/benchmark_MemoryBlocking" ]]; then
    perf stat -r ${PERF_REPETITIONS} -e ${EVENTS_TO_ANALYZE} ${BINARY_FILE} 2> ${OUTPUT_DIR}/${FILE_NAME} $3 $4 $5 $6 $7 $8 $9 ${10} ${11}
elif
    perf stat -r ${PERF_REPETITIONS} -e ${EVENTS_TO_ANALYZE} ${BINARY_FILE} 2> ${OUTPUT_DIR}/${FILE_NAME} $3 $4 $5 $6 $7 $8
fi
