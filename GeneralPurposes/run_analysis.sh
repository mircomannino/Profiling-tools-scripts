#!/bin/bash

declare -a list_of_analysis=("1" "2" "3" "4" "5" "6" "7" "8" "9")

declare -a n_repetitions=("1" "50")

declare -a arguments=( # (Image size, Image depth, Kernel size, N. Kernel)
    "200 3 5 64"
    "200 3 5 128"
    "200 64 5 64"
    "200 64 5 128"
    "50 128 1 128"
    "50 128 1 512"
    "50 128 1 1024"
    "50 128 3 128"
    "50 128 3 512"
    "50 128 3 1024"
    "50 512 1 128"
    "50 512 1 512"
    "50 512 1 1024"
    "50 512 3 128"
    "50 512 3 512"
    "50 512 3 1024"
    "10 512 1 512"
    "10 512 1 1024"
    "10 512 3 512"
    "10 512 3 1024"
)

BINARY_FILE=./bin/benchmark_Naive

for NUMBER_ANALYSYS in "${list_of_analysis[@]}"
do
    CURRENT_DIR=analysis_N${NUMBER_ANALYSYS}
    LOOP_ORDER=${NUMBER_ANALYSYS}
    for ARGUMENT in "${arguments[@]}"
    do
        for N_REPETITIONS in "${n_repetitions[@]}"
        do
            # Execution time
            OUT_DIR_TIME=${CURRENT_DIR}/ExecutionTime_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/execution_times
            ./analysis_N${NUMBER_ANALYSYS}/ExecutionTime_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/execution_time_generator.sh \
            ${BINARY_FILE} ${OUT_DIR_TIME} ${ARGUMENT} ${LOOP_ORDER} ${N_REPETITIONS}

            # Perf
            OUT_DIR_PERF=${CURRENT_DIR}/Perf_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/perf_reports
            ./analysis_N${NUMBER_ANALYSYS}/Perf_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/perf_report_generator.sh \
            ${BINARY_FILE} ${OUT_DIR_PERF} ${ARGUMENT} ${LOOP_ORDER} ${N_REPETITIONS}

            # VTune
            ROOT_VTUNE=${CURRENT_DIR}/VTune_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/vtune_data
            OUT_DIR_VTUNE=${CURRENT_DIR}/VTune_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/reports
            ./analysis_N${NUMBER_ANALYSYS}/VTune_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/VTune_report_generator.sh \
            ${BINARY_FILE} ${ROOT_VTUNE} ${OUT_DIR_VTUNE} ${ARGUMENT} ${LOOP_ORDER} ${N_REPETITIONS}
        done

    done
done
