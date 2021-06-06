#!/bin/bash

# declare -a list_of_analysis=("1" "2" "3" "4" "5" "6" "7" "8")
declare -a list_of_analysis=("1" "2")

# Clone the scripts file for profiling
git clone https://github.com/mircomannino/Profiling-tools-scripts.git

for NUMBER in "${list_of_analysis[@]}"
do
    CURRENT_DIR=analysis_N${NUMBER}
    # Create the folder
    mkdir ${CURRENT_DIR}
    cd ${CURRENT_DIR}
    # Create a folder for each type of profiling analysis
    mkdir Perf_analysis_N${NUMBER}_1-repetitions
    mkdir Perf_analysis_N${NUMBER}_50-repetitions
    mkdir VTune_analysis_N${NUMBER}_1-repetitions
    mkdir VTune_analysis_N${NUMBER}_50-repetitions
    # Copy the profiling scripts in the right folder
    cp ../Profiling-tools-scripts/Perf/perf_report_generator.sh ./Perf_analysis_N${NUMBER}_1-repetitions
    cp ../Profiling-tools-scripts/Perf/perf_report_generator.sh ./Perf_analysis_N${NUMBER}_50-repetitions
    cp ../Profiling-tools-scripts/Intel-VTune/VTune_report_generator.sh ./VTune_analysis_N${NUMBER}_1-repetitions
    cp ../Profiling-tools-scripts/Intel-VTune/VTune_report_generator.sh ./VTune_analysis_N${NUMBER}_50-repetitions
    # Return to root folder
    cd ..
done

# Delete the scripts file for profiling
rm -rf Profiling-tools-scripts