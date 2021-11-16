
#!/bin/bash

# Arguements table
# =====================================================================================
# Arg   |       Naive           | MemoryBlocking                | Parallel            |
# =====================================================================================
# $1    |                   binary file to profile                                    |
# $2    |                   output directory                                          |
# $3    |                   Size of the image (H and W)                               |
# $4    |                   Depth of the image                                        |
# $5    |                   Size of the kernel (H and W)                              |
# $6    |                   Depth of the kernel                                       |
# $7    |   Order of loops      | Blocked input channel size    |   Number of threads |
# $8    |   Number of tests     | Blocked output channel size   |   Order of loops    |
# $9    |                       | Blocked output width size     |   Number of tests   |
# $10   |                       | Order of loopa                |                     |
# $11   |                       | Number of tests                                     |
# =====================================================================================

OUTPUT_DIR=$2
BINARY_FILE=$1

if [[ ${BINARY_FILE} =~ "./bin/benchmark_Naive" ]] && [[ "$#" -ne 8 ]]; then
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Output directory"
    echo "  3-8) Arguments of binary file. See documentation"
    exit 1
fi
if [[ ${BINARY_FILE} =~ "./bin/benchmark_MemoryBlocking" ]] && [[ "$#" -ne 11 ]]; then
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Output directory"
    echo "  3-11) Arguments of binary file. See documentation"
    exit 1
fi
if [[ ${BINARY_FILE} =~ "./bin/benchmark_Parallel" ]] && [[ "$#" -ne 9 ]]; then
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Output directory"
    echo "  3-9) Arguements of binary file. See documentation"
fi

# Permissions
# echo "0" | sudo tee /proc/sys/kernel/kptr_restrict
# echo "0" | sudo tee /proc/sys/kernel/perf_event_paranoid

EVENTS_TO_ANALYZE=("cache-misses,cache-references,branches,branch-misses,cycles,instructions,L1-dcache-loads-misses,LLC-loads-misses,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_single")
PERF_REPETITIONS=3

# Setup output folder and arguments
if [[ ${BINARY_FILE} =~ "./bin/benchmark_Naive" ]]; then # Naive
    OUT_FILE_NAME=$(basename $1)_$3_$4_$5_$6_$7_$8.txt
    ARGUMENTS="$3 $4 $5 $6 $7 $8"
fi
if [[ ${BINARY_FILE} =~ "./bin/benchmark_MemoryBlocking" ]]; then # MemoryBlocking
    OUT_FILE_NAME=$(basename $1)_$3_$4_$5_$6_$7_$8_$9_${10}_${11}.txt
    ARGUMENTS="$3 $4 $5 $6 $7 $8 $9 ${10} ${11}"
fi
if [[ ${BINARY_FILE} =~ "./bin/benchmark_Parallel" ]]; then # Parallel
    OUT_FILE_NAME=$(basename $1)_$3_$4_$5_$6_$7_$8_$9.txt
    ARGUMENTS="$3 $4 $5 $6 $7 $8"
fi

# Run the execution
mkdir -p ${OUTPUT_DIR}
perf stat -r ${PERF_REPETITIONS} -e ${EVENTS_TO_ANALYZE} ${BINARY_FILE} 2> ${OUTPUT_DIR}/${OUT_FILE_NAME} ${ARGUMENTS}
# if [[ ${BINARY_FILE} =~ "./bin/benchmark_Naive" ]]; then
#     perf stat -r ${PERF_REPETITIONS} -e ${EVENTS_TO_ANALYZE} ${BINARY_FILE} 2> ${OUTPUT_DIR}/${OUT_FILE_NAME} $3 $4 $5 $6 $7 $8
# fi
# if [[ ${BINARY_FILE} =~ "./bin/benchmark_MemoryBlocking" ]]; then
#     perf stat -r ${PERF_REPETITIONS} -e ${EVENTS_TO_ANALYZE} ${BINARY_FILE} 2> ${OUTPUT_DIR}/${OUT_FILE_NAME} $3 $4 $5 $6 $7 $8 $9 ${10} ${11}
# fi
# if [[ ${BINARY_FILE} =~ "./bin/benchmark_Parallel" ]]; then
#     perf stat -r ${PERF_REPETITIONS} -e ${EVENTS_TO_ANALYZE} ${BINARY_FILE} 2> ${OUTPUT_DIR}/${OUT_FILE_NAME} $3 $4 $5 $6 $7 $8 $9
# fi
