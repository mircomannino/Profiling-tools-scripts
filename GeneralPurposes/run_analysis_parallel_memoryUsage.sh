#!/usr/bin/env bash

#!/bin/bash

declare -a list_of_analysis=("1" "2" "3" "4" "5")

declare -a n_repetitions=("5")

declare -a n_threads=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16")

BINARY_FILE=./bin/benchmark_ParallelAlexNetFULL
CORE_ALLOCATION_TYPE="PHYCORE1_THREAD1"

for NUMBER_ANALYSYS in "${list_of_analysis[@]}"
do
    CURRENT_DIR=analysis_N${NUMBER_ANALYSYS}
    LOOP_ORDER=${NUMBER_ANALYSYS}

    for N_REPETITIONS in "${n_repetitions[@]}"
    do
        ##### Parallel #####
        for N_THREADS in "${n_threads[@]}"
        do
            # # Execution time
            # OUT_DIR_TIME=${CURRENT_DIR}/ExecutionTime_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/execution_times
            # ./analysis_N${NUMBER_ANALYSYS}/ExecutionTime_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/execution_time_generator.sh \
            # ${BINARY_FILE_PARALLEL} ${OUT_DIR_TIME} ${N_THREADS} ${LOOP_ORDER} ${N_REPETITIONS} ${CORE_ALLOCATION_TYPE}

            # # Perf
            # OUT_DIR_PERF=${CURRENT_DIR}/Perf_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/perf_reports
            # ./analysis_N${NUMBER_ANALYSYS}/Perf_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/perf_report_generator.sh \
            # ${BINARY_FILE_PARALLEL} ${OUT_DIR_PERF} ${N_THREADS} ${LOOP_ORDER} ${N_REPETITIONS} ${CORE_ALLOCATION_TYPE}

            # VTune
            ROOT_VTUNE=${CURRENT_DIR}/VTune_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/vtune_data
            OUT_DIR_VTUNE=${CURRENT_DIR}/VTune_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/reports
            ./analysis_N${NUMBER_ANALYSYS}/VTune_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/VTune_report_generator.sh \
            ${BINARY_FILE_PARALLEL} ${ROOT_VTUNE} ${OUT_DIR_VTUNE} ${N_THREADS} ${LOOP_ORDER} ${N_REPETITIONS} ${CORE_ALLOCATION_TYPE}
            rm -rf ${ROOT_VTUNE}/* # Clean tmp data

        done
    done
done
