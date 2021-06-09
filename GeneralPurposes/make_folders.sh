#!/bin/bash

# declare -a list_of_analysis=("1" "2" "3" "4" "5" "6" "7" "8")
declare -a list_of_analysis=("1")

# Clone the scripts file for profiling
rm -rf Profiling-tools-scripts
git clone https://github.com/mircomannino/Profiling-tools-scripts.git

for NUMBER in "${list_of_analysis[@]}"
do
    CURRENT_DIR=analysis_N${NUMBER}
    # Create the folder
    mkdir ${CURRENT_DIR}
    cd ${CURRENT_DIR}
    # Create a folder for each type of profiling analysis
    mkdir ExecutionTime_analysis_N${NUMBER}_1-repetitions
    mkdir ExecutionTime_analysis_N${NUMBER}_50-repetitions
    mkdir Perf_analysis_N${NUMBER}_1-repetitions
    mkdir Perf_analysis_N${NUMBER}_50-repetitions
    mkdir VTune_analysis_N${NUMBER}_1-repetitions
    mkdir VTune_analysis_N${NUMBER}_50-repetitions
    # Copy the profiling scripts in the right folder
    cp ../Profiling-tools-scripts/ExecutionTime/execution_time_generator.sh ./ExecutionTime_analysis_N${NUMBER}_1-repetitions
    cp ../Profiling-tools-scripts/ExecutionTime/execution_time_generator.sh ./ExecutionTime_analysis_N${NUMBER}_50-repetitions
    cp ../Profiling-tools-scripts/Perf/perf_report_generator.sh ./Perf_analysis_N${NUMBER}_1-repetitions
    cp ../Profiling-tools-scripts/Perf/perf_report_generator.sh ./Perf_analysis_N${NUMBER}_50-repetitions
    cp ../Profiling-tools-scripts/Intel-VTune/VTune_report_generator.sh ./VTune_analysis_N${NUMBER}_1-repetitions
    cp ../Profiling-tools-scripts/Intel-VTune/VTune_report_generator.sh ./VTune_analysis_N${NUMBER}_50-repetitions
    # Permissions
    chmod u+x ./ExecutionTime_analysis_N${NUMBER}_1-repetitions/*.sh
    chmod u+x ./ExecutionTime_analysis_N${NUMBER}_50-repetitions/*.sh
    chmod u+x ./Perf_analysis_N${NUMBER}_1-repetitions/*.sh
    chmod u+x ./Perf_analysis_N${NUMBER}_50-repetitions/*.sh
    chmod u+x ./VTune_analysis_N${NUMBER}_1-repetitions/*.sh
    chmod u+x ./VTune_analysis_N${NUMBER}_50-repetitions/*.sh
    # Return to root folder
    cd ..
done

# Delete the scripts file for profiling
rm -rf Profiling-tools-scripts
