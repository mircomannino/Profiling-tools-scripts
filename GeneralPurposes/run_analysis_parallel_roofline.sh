#!/bin/bash

declare -a list_of_analysis=("1" "2" "3" "4" "5")

declare -a n_repetitions=("10")

declare -a n_threads=("1" "2" "3" "4" "5" "6" "7" "8")


BINARY_FILE=./bin/benchmark_ParallelAlexNetFULL
CORE_ALLOCATION_TYPE="PHYCORE1_THREAD1"

for NUMBER_ANALYSYS in "${list_of_analysis[@]}"
do
    CURRENT_DIR=analysis_N${NUMBER_ANALYSYS}
    LOOP_ORDER=${NUMBER_ANALYSYS}
    for N_REPETITIONS in "${n_repetitions[@]}"
    do
        for N_THREADS in "${n_threads[@]}"
        do
            # Roofline
            ROOT_DATA_ANALYSIS=${CURRENT_DIR}/Roofline_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/roofline-data
            OUT_DIR_ROOFLINE=${CURRENT_DIR}/Roofline_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/roofline-reports
            ./analysis_N${NUMBER_ANALYSYS}/Roofline_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/roofline_report_generator.sh \
            ${BINARY_FILE} ${ROOT_DATA_ANALYSIS} ${OUT_DIR_ROOFLINE} ${N_THREADS} ${LOOP_ORDER} ${N_REPETITIONS} ${CORE_ALLOCATION_TYPE}
            rm -rf ${ROOT_DATA_ANALYSIS}/* # Clean tmp data
        done
    done
done
