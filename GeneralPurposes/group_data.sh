#!/bin/bash

# Download repository with aggregators scripts
rm -rf Profiling-tools-scripts
git clone https://github.com/mircomannino/Profiling-tools-scripts.git

ROOT_FOLDER=$(pwd)
ALL_ROOFLINES_FOLDER=$(pwd)/roofline-models

for ANALYSIS_FOLDER in ./analysis_N*
do
    cd ${ANALYSIS_FOLDER}

    # Execution time
    for EXECUTION_TIME_FOLDER in ./ExecutionTime*
    do
        cd ${EXECUTION_TIME_FOLDER}
        CSV_NAME=$(basename `pwd`)
        # Copy the python script to group data
        cp ${ROOT_FOLDER}/Profiling-tools-scripts/Aggregators/group_execution_times.py ./execution_times/group_execution_times.py
        # Exectue the python script to group data
        cd ./execution_times
        python3 group_execution_times.py -o ${CSV_NAME}
        # Delete python script
        rm group_execution_times.py
        cd ../..
    done

    # Perf
    for PERF_ANALYSIS_FOLDER in ./Perf*
    do
        cd ${PERF_ANALYSIS_FOLDER}
        CSV_NAME=$(basename `pwd`)
        # Copy the python script to group data
        cp ${ROOT_FOLDER}/Profiling-tools-scripts/Aggregators/group_data_perf.py ./perf_reports/group_data_perf.py
        # Exectue the python script to group data
        cd ./perf_reports
        python3 group_data_perf.py -o ${CSV_NAME}
        # Delete python script
        rm group_data_perf.py
        cd ../..
    done

    # VTune
    for VTUNE_ANALYSIS_FOLDER in ./VTune*
    do
        cd ${VTUNE_ANALYSIS_FOLDER}
        CSV_NAME=$(basename `pwd`)
        # Copy the python script to group data
        cp ${ROOT_FOLDER}/Profiling-tools-scripts/Aggregators/group_data_VTune.py ./reports/group_data_VTune.py
        # Exectue the python script to group data
        cd ./reports
        python3 group_data_VTune.py -o ${CSV_NAME}
        # Delete python script
        rm group_data_VTune.py
        cd ../..
    done

    cd ..
done

# Remove folder with grouping scripts
rm -rf Profiling-tools-scripts
