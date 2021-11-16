#!/bin/bash
# Arguments:
#   $1: binary file to profile
#   $2: root data
#   $3: output directory
#   $4: Size of image
#   $5: Depth of image
#   $6: Size of kernels
#   $7: Number of kernels
#   $8: Blocked input channel size
#   $9: Blocked output channel size
#   $10: Blocked output width size
#   $11[8]: Order of for loops
#   $12[9]: Number of tests to do

# Arguements table
# =====================================================================================
# Arg   |       Naive           | MemoryBlocking                | Parallel            |
# =====================================================================================
# $1    |                   binary file to profile                                    |
# $2    |                   VTune root data                                           |
# $3    |                   output directory                                          |
# $4    |                   Size of the image (H and W)                               |
# $5    |                   Depth of the image                                        |
# $6    |                   Size of the kernel (H and W)                              |
# $7    |                   Depth of the kernel                                       |
# $8    |   Order of loops      | Blocked input channel size    |   Number of threads |
# $9    |   Number of tests     | Blocked output channel size   |   Order of loops    |
# $10    |                       | Blocked output width size     |   Number of tests   |
# $11   |                       | Order of loopa                |                     |
# $12   |                       | Number of tests                                     |
# =====================================================================================

BIN_NAME=$(basename $1)
BINARY_FILE=$1

if [[ ${BINARY_FILE} =~ "./bin/benchmark_Naive" ]] && [[ "$#" -ne 9 ]]; then
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Root data for VTune"
    echo "  3) Output directory"
    echo "  4-9) Arguments of binary file. See documentation"
    exit 1
fi
if [[ ${BINARY_FILE} =~ "./bin/benchmark_MemoryBlocking" ]] && [[ "$#" -ne 12 ]]; then
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Root data for VTune"
    echo "  3) Output directory"
    echo "  4-12) Arguments of binary file. See documentation"
    exit 1
fi
if [[ ${BINARY_FILE} =~ "./bin/benchmark_Parallel" ]] && [[ "$#" -ne 10 ]]; then
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Root data for VTune"
    echo "  3) Output directory"
    echo "  4-10) Arguments of binary file. See documentation"
    exit 1
fi


# Setup identifiers and arguments
if [[ ${BINARY_FILE} =~ "./bin/benchmark_Naive" ]]; then # Naive
    BIN_IDENTIFIER=${BIN_NAME}_$4_$5_$6_$7_$8_$9
    ARGUMENTS="$4 $5 $6 $7 $8 $9"
fi
if [[ ${BINARY_FILE} =~ "./bin/benchmark_MemoryBlocking" ]]; then # MemoryBlocking
    BIN_IDENTIFIER=${BIN_NAME}_$4_$5_$6_$7_$8_$9_${10}_${11}_${12}
    ARGUMENTS="$4 $5 $6 $7 $8 $9 ${10} ${11} ${12}"
fi
if [[ ${BINARY_FILE} =~ "./bin/benchmark_Parallel" ]]; then # Parallel
    BIN_IDENTIFIER=${BIN_NAME}_$4_$5_$6_$7_$8_$9_${10}
    ARGUMENTS="$4 $5 $6 $7 $8 $9 ${10}"
fi

# VTune collect information - ROOT folder
ROOT_VTUNE=$2
mkdir -p ${ROOT_VTUNE}

# VTune collect information - BIN specific folder
BIN_VTUNE_DIR=${ROOT_VTUNE}/${BIN_IDENTIFIER}
mkdir -p ${BIN_VTUNE_DIR}

# VTune report information - ROOT folder
ROOT_REPORTS=$3
mkdir -p ${ROOT_REPORTS}

# VTune report information - BIN specific folder
FORMAT=csv
BIN_REPORTS_DIR=${ROOT_REPORTS}/${BIN_IDENTIFIER}
mkdir -p ${BIN_REPORTS_DIR}


# Add list of collection options
declare -a collect_types=(
        "hpc-performance"
        "uarch-exploration"
        "memory-access"
)


# Collect data with vtune command
for TYPE in "${collect_types[@]}"
do
        mkdir -p ${BIN_VTUNE_DIR}/${BIN_IDENTIFIER}_${TYPE}
        vtune -collect $TYPE -knob sampling-interval=0.5 -result-dir ${BIN_VTUNE_DIR}/${BIN_IDENTIFIER}_${TYPE} -- ${BINARY_FILE} ${ARGUMENTS}
done

# Create reports with vtune command
for TYPE in "${collect_types[@]}"
do
        vtune -report summary -result-dir ${BIN_VTUNE_DIR}/${BIN_IDENTIFIER}_${TYPE} -format ${FORMAT} -report-output ${BIN_REPORTS_DIR}/summary_${TYPE}.${FORMAT}
done
