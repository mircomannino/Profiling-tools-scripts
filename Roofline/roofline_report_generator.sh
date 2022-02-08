#!/bin/bash
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

if [[ ${BINARY_FILE} =~ ./bin/benchmark_Parallel[a-zA-Z]+FULL$ ]] && [[ "$#" -ne 7 ]]; then # Enter if name is benchmark_Parallel[NetworkName]FULL
    echo "Detect Parallel + Memory blocking version with wrong arguments"
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Root to store analysis data"
    echo "  3) Output directory"
    echo "  4-7) Arguements of binary file. See documentation"
    exit 1
elif [[ ${BINARY_FILE} =~ ./bin/benchmark_Sequential[a-zA-Z]+FULL$ ]] && [[ "$#" -ne 5 ]]; then # Enter if name is benchmark_Sequential[NetworkName]FULL
    echo "Detect Parallel + Memory blocking version with wrong arguments"
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Root to store analysis data"
    echo "  3) Output directory"
    echo "  4-5) Arguements of binary file. See documentation"
    exit 1
elif [[ ${BINARY_FILE} == ./bin/benchmark_ParallelMemoryBlocking ]] && [[ "$#" -ne 13 ]]; then
    echo "Detect Parallel + Memory blocking version with wrong arguments"
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Root to store analysis data"
    echo "  3) Output directory"
    echo "  4-13) Arguements of binary file. See documentation"
    exit 1
elif [[ ${BINARY_FILE} == ./bin/benchmark_Parallel ]] && [[ "$#" -ne 10 ]]; then
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Root to store analysis data"
    echo "  3) Output directory"
    echo "  4-10) Arguments of binary file. See documentation"
    exit 1
elif [[ ${BINARY_FILE} =~ ./bin/benchmark_MemoryBlocking ]] && [[ "$#" -ne 12 ]]; then
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Root to store analysis data"
    echo "  3) Output directory"
    echo "  4-12) Arguments of binary file. See documentation"
    exit 1
elif [[ ${BINARY_FILE} =~ ./bin/benchmark_Naive ]] && [[ "$#" -ne 9 ]]; then
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Root to store analysis data"
    echo "  3) Output directory"
    echo "  4-9) Arguments of binary file. See documentation"
    exit 1
fi

echo # New line
echo "-------------- [Roofline analysis: START] "${@}" --------------"

# Setup output folder and arguments
if [[ ${BINARY_FILE} =~ ./bin/benchmark_Parallel[a-zA-Z]+FULL$ ]]; then
    OUT_FILE_NAME=$(basename $1)_$4_$5_$6_$7.html
    ARGUMENTS="$4 $5 $6 $7"
elif [[ ${BINARY_FILE} =~ ./bin/benchmark_Sequential[a-zA-Z]+FULL$ ]]; then
    OUT_FILE_NAME=$(basename $1)_$4_$5.html
    ARGUMENTS="$4 $5"
elif [[ ${BINARY_FILE} =~ "./bin/benchmark_ParallelMemoryBlocking" ]]; then # Parallel
    OUT_FILE_NAME=$(basename $1)_$4_$5_$6_$7_$8_$9_${10}_${11}_${12}_${13}
    ARGUMENTS="$4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}"
elif [[ ${BINARY_FILE} =~ "./bin/benchmark_Parallel" ]]; then # Parallel
    OUT_FILE_NAME=$(basename $1)_$4_$5_$6_$7_$8_$9_${10}
    ARGUMENTS="$4 $5 $6 $7 $8 $9 ${10}"
elif [[ ${BINARY_FILE} =~ "./bin/benchmark_MemoryBlocking" ]]; then # MemoryBlocking
    OUT_FILE_NAME=$(basename $1)_$4_$5_$6_$7_$8_$9_${10}_${11}_${12}
    ARGUMENTS="$4 $5 $6 $7 $8 $9 ${10} ${11} ${12}"
elif [[ ${BINARY_FILE} =~ "./bin/benchmark_Naive" ]]; then # Naive
    OUT_FILE_NAME=$(basename $1)_$4_$5_$6_$7_$8_$9
    ARGUMENTS="$4 $5 $6 $7 $8 $9"
fi

ROOT_DATA_ANALYSIS=$2
OUTPUT_DIR=$3

# Run the execution
mkdir -p ${ROOT_DATA_ANALYSIS}
mkdir -p ${OUTPUT_DIR}

SAMPLING_INTERVAL=1 #ms

# Collect data
advixe-cl --collect=roofline --interval=${SAMPLING_INTERVAL} --project-dir=${ROOT_DATA_ANALYSIS} -- ${BINARY_FILE} ${ARGUMENTS}
# Make the report
advixe-cl --report=roofline --project-dir=${ROOT_DATA_ANALYSIS} --report-output=${OUTPUT_DIR}/${OUT_FILE_NAME}

echo # New line
echo "-------------- [Roofline analysis: END] "${@}" --------------"
