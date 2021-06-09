#!/bin/bash

declare -a list_of_analysis=("1")

declare -a n_repetitions=("1" "50")

declare -a arguments=( # (Image size, Image depth, Kernel size, N. Kernel)
    "10 3 5 16"
    "50 40 4 4"
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
