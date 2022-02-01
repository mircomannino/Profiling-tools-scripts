#!/bin/bash

declare -a list_of_analysis=("1" "2" "3" "4" "5")

declare -a n_repetitions=("5")

declare -a n_threads=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16")

BINARY_FILE=./bin/benchmark_ParallelAlexNetFULL
CORE_TYPE_ALLOCATION="DEFAULT"

for NUMBER_ANALYSYS in "${list_of_analysis[@]}"
do
    CURRENT_DIR=analysis_N${NUMBER_ANALYSYS}
    LOOP_ORDER=${NUMBER_ANALYSYS}
    for N_REPETITIONS in "${n_repetitions[@]}"
    do
        for N_THREADS in "${n_threads[@]}"
        do
            # Execution time
            OUT_DIR_TIME=${CURRENT_DIR}/ExecutionTime_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/execution_times
            ./analysis_N${NUMBER_ANALYSYS}/ExecutionTime_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/execution_time_generator.sh \
            ${BINARY_FILE_PARALLEL} ${OUT_DIR_TIME} ${ARGUMENT} ${N_THREADS} ${LOOP_ORDER} ${N_REPETITIONS} ${CORE_TYPE_ALLOCATION}
        done

    done

    done
done