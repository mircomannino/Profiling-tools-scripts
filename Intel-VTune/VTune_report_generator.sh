#!/bin/bash
# Arguments:
#   $1: binary file to profile
#   $2: root data
#   $3: output directory
#   $4: Size of image
#   $5: Depth of image
#   $6: Size of kernels
#   $7: Number of kernels
#   $8: Order of for loops
#   $9: Number of tests to do

if [ "$#" -ne 9 ]; then
    echo "Insert the following arguments:"
    echo "  1) Binary file to analyze"
    echo "  2) Root data for VTune"
    echo "  3) Output directory"
    echo "  4-9) Arguments of binary file. See documentation"
    exit 1
fi

BIN_NAME=$(basename $1)
BIN_IDENTIFIER=${BIN_NAME}_$4_$5_$6_$7_$8_$9

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
        vtune -collect $TYPE -result-dir ${BIN_VTUNE_DIR}/${BIN_IDENTIFIER}_${TYPE} -allow-multiple-runs $1 $4 $5 $6 $7 $8 $9
done

# Create reports with vtune command
for TYPE in "${collect_types[@]}"
do
        vtune -report summary -result-dir ${BIN_VTUNE_DIR}/${BIN_IDENTIFIER}_${TYPE} -format ${FORMAT} -report-output ${BIN_REPORTS_DIR}/summary_${TYPE}.${FORMAT}
        
done
