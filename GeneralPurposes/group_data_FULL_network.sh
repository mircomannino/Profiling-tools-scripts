#!/bin/bash

# Download repository with aggregators scripts
rm -rf Profiling-tools-scripts
git clone https://github.com/mircomannino/Profiling-tools-scripts.git

ROOT_FOLDER=$(pwd)

for ANALYSIS_FOLDER in ./analysis_N*
do
    cd ${ANALYSIS_FOLDER}

    # Execution time
    for EXECUTION_TIME_FOLDER in ./ExecutionTime*
    do
        cd ${EXECUTION_TIME_FOLDER}
        CSV_NAME=$(basename `pwd`)
        FULL_NETWORK='--full-network'
        # Copy the python script to group data
        cp ${ROOT_FOLDER}/Profiling-tools-scripts/Aggregators/group_execution_times.py ./execution_times/group_execution_times.py
        # Exectue the python script to group data
        cd ./execution_times
        python3 group_execution_times.py -o ${CSV_NAME} ${FULL_NETWORK}
        # Delete python script
        rm group_execution_times.py
        cd ../..
    done

    cd ..
done

# Remove folder with grouping scripts
rm -rf Profiling-tools-scripts
