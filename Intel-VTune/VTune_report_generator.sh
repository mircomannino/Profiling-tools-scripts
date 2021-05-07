#!/bin/bash
# Arguments:
#       $1: binary file to profile

BIN_NAME=$(basename $1)

# VTune collect information - ROOT folder
ROOT_VTUNE=vtune_data
mkdir -p ${ROOT_VTUNE}

# VTune collect information - BIN specific folder
BIN_VTUNE_DIR=${ROOT_VTUNE}/${BIN_NAME}
mkdir -p ${BIN_VTUNE_DIR}

# VTune report information - ROOT folder
ROOT_REPORTS=reports
mkdir -p ${ROOT_REPORTS}

# VTune report information - BIN specific folder
FORMAT=html
BIN_REPORTS_DIR=${ROOT_REPORTS}/${BIN_NAME}
mkdir -p ${BIN_REPORTS_DIR}


# Add list of collection options
COLLECT_APP_ROOT=app_times
declare -a COLLECT_TYPES=("hpc-performance"
                          "uarch-exploration" "memory-access"
                         )


# Collect data with vtune command
for TYPE in "${COLLECT_TYPES[@]}"
do
        mkdir -p ${BIN_VTUNE_DIR}/${ROOT_VTUNE}_${TYPE}
        vtune -collect $TYPE -result-dir ${BIN_VTUNE_DIR}/${ROOT_VTUNE}_${TYPE} $1
done

# Create reports with vtune command
for TYPE in "${COLLECT_TYPES[@]}"
do
        vtune -report summary -result-dir ${BIN_VTUNE_DIR}/${ROOT_VTUNE}_${TYPE} -format ${FORMAT} -report-output $(pwd)/${BIN_REPORTS_DIR}/summary_${TYPE}.${FORMAT}
done
