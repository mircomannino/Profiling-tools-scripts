#!/bin/bash

declare -a list_of_analysis=("1" "2" "3" "4" "5")

declare -a n_repetitions=("5")

# Clone the scripts file for profiling
rm -rf Profiling-tools-scripts
git clone https://github.com/mircomannino/Profiling-tools-scripts.git

for NUMBER in "${list_of_analysis[@]}"
do
    CURRENT_DIR=analysis_N${NUMBER}
    # Create the folder
    mkdir -p ${CURRENT_DIR}
    cd ${CURRENT_DIR}
    # Create a folder for each type of profiling analysis
    for N_REPETITIONS in "${n_repetitions[@]}"
    do
        ### Make folders ###
        mkdir -p ExecutionTime_analysis_N${NUMBER}_${N_REPETITIONS}-repetitions    # Execution time
        mkdir -p Perf_analysis_N${NUMBER}_${N_REPETITIONS}-repetitions             # Perf
        mkdir -p VTune_analysis_N${NUMBER}_${N_REPETITIONS}-repetitions          # VTune

        ### Copy the profiling scripts in the right folder ###
        cp ../Profiling-tools-scripts/ExecutionTime/execution_time_generator.sh ./ExecutionTime_analysis_N${NUMBER}_${N_REPETITIONS}-repetitions
        cp ../Profiling-tools-scripts/Perf/perf_report_generator.sh ./Perf_analysis_N${NUMBER}_${N_REPETITIONS}-repetitions
        cp ../Profiling-tools-scripts/Intel-VTune/VTune_report_generator.sh ./VTune_analysis_N${NUMBER}_${N_REPETITIONS}-repetitions

        ### Permissions ###
        chmod u+x ./ExecutionTime_analysis_N${NUMBER}_${N_REPETITIONS}-repetitions/*.sh
        chmod u+x ./Perf_analysis_N${NUMBER}_${N_REPETITIONS}-repetitions/*.sh
        chmod u+x ./VTune_analysis_N${NUMBER}_${N_REPETITIONS}-repetitions/*.sh

    done
    # Return to root folder
    cd ..
done

# Delete the scripts file for profiling
rm -rf Profiling-tools-scripts
