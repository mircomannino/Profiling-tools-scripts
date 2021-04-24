#!/bin/bash
# Arguments:
#       $1: binary file to profile

# VTune collect information
ROOT_VTUNE=vtune_data
rm -rf ${ROOT_VTUNE}
mkdir ${ROOT_VTUNE}

# VTune report information
FORMAT=html
ROOT_REPORTS=reports
rm -rf ${ROOT_REPORTS}
mkdir ${ROOT_REPORTS}


# Add list of collection options
COLLECT_APP_ROOT=app_times
declare -a COLLECT_TYPES=("hotspots" "memory-consumption" 
                          "threading" "hpc-performance"
                          "uarch-exploration" "memory-access"
                         )


# Collect data with vtune command 
for TYPE in "${COLLECT_TYPES[@]}"
do 
        rm -rf ${ROOT_VTUNE}/${ROOT_VTUNE}_${TYPE}
        mkdir -p ${ROOT_VTUNE}/${ROOT_VTUNE}_${TYPE}
        vtune -collect $TYPE -result-dir ${ROOT_VTUNE}/${ROOT_VTUNE}_${TYPE} $1
done

# Create reports with vtune command
for TYPE in "${COLLECT_TYPES[@]}"
do
        vtune -report summary -result-dir ${ROOT_VTUNE}/${ROOT_VTUNE}_${TYPE} -format ${FORMAT} -report-output $(pwd)/${ROOT_REPORTS}/summary_${TYPE}.${FORMAT} 
done