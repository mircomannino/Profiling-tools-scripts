#!/usr/bin/env bash

#!/bin/bash

declare -a list_of_analysis=("1" "2" "3" "4" "5")

declare -a n_repetitions=("5")

declare -a n_threads=(
    "4"
    "8"
)

declare -a blocking_size=(
    "-1 16 -1"
    "-1 32 -1"
    "-1 64 -1"
    "16 -1 -1"
    "32, -1 -1"
    "64, -1 -1"
)

declare -a arguments=( # (Image size, Image depth, Kernel size, N. Kernel, Cib size, Cob size, Wob size)
    "227 3 11 96"
    "27 96 5 256"
    "13 384 3 384"
    "13 384 3 256"
    "13 256 3 384"
)

BINARY_FILE_NAIVE=./bin/benchmark_NaiveKernelNKernels
BINARY_FILE_PARALLEL_MEM_BLOCKING=./bin/benchmark_ParallelMemoryBlockingSoft

for NUMBER_ANALYSYS in "${list_of_analysis[@]}"
do
    CURRENT_DIR=analysis_N${NUMBER_ANALYSYS}
    LOOP_ORDER=${NUMBER_ANALYSYS}
    for ARGUMENT in "${arguments[@]}"
    do

        for N_REPETITIONS in "${n_repetitions[@]}"
        do
            ##### Naive #####
            # Execution time
            OUT_DIR_TIME=${CURRENT_DIR}/ExecutionTime_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/execution_times
            ./analysis_N${NUMBER_ANALYSYS}/ExecutionTime_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/execution_time_generator.sh \
            ${BINARY_FILE_NAIVE} ${OUT_DIR_TIME} ${ARGUMENT} ${LOOP_ORDER} ${N_REPETITIONS}

            # Perf
            OUT_DIR_PERF=${CURRENT_DIR}/Perf_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/perf_reports
            ./analysis_N${NUMBER_ANALYSYS}/Perf_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/perf_report_generator.sh \
            ${BINARY_FILE_NAIVE} ${OUT_DIR_PERF} ${ARGUMENT} ${LOOP_ORDER} ${N_REPETITIONS}

            # VTune
            ROOT_VTUNE=${CURRENT_DIR}/VTune_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/vtune_data
            OUT_DIR_VTUNE=${CURRENT_DIR}/VTune_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/reports
            ./analysis_N${NUMBER_ANALYSYS}/VTune_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/VTune_report_generator.sh \
            ${BINARY_FILE_NAIVE} ${ROOT_VTUNE} ${OUT_DIR_VTUNE} ${ARGUMENT} ${LOOP_ORDER} ${N_REPETITIONS}
            rm -rf ${ROOT_VTUNE}/* # Clean tmp data

            ##### Parallel  #####
            for N_THREADS in "${n_threads[@]}"
            do
                for BLOCKING_SIZE in "${blocking_size[@]}"
                do
                    # Execution time
                    OUT_DIR_TIME=${CURRENT_DIR}/ExecutionTime_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/execution_times
                    ./analysis_N${NUMBER_ANALYSYS}/ExecutionTime_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/execution_time_generator.sh \
                    ${BINARY_FILE_PARALLEL_MEM_BLOCKING} ${OUT_DIR_TIME} ${ARGUMENT} ${BLOCKING_SIZE} ${N_THREADS} ${LOOP_ORDER} ${N_REPETITIONS}

                    # Perf
                    OUT_DIR_PERF=${CURRENT_DIR}/Perf_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/perf_reports
                    ./analysis_N${NUMBER_ANALYSYS}/Perf_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/perf_report_generator.sh \
                    ${BINARY_FILE_PARALLEL_MEM_BLOCKING} ${OUT_DIR_PERF} ${ARGUMENT} ${BLOCKING_SIZE} ${N_THREADS} ${LOOP_ORDER} ${N_REPETITIONS}

                    # VTune
                    ROOT_VTUNE=${CURRENT_DIR}/VTune_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/vtune_data
                    OUT_DIR_VTUNE=${CURRENT_DIR}/VTune_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/reports
                    ./analysis_N${NUMBER_ANALYSYS}/VTune_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/VTune_report_generator.sh \
                    ${BINARY_FILE_PARALLEL_MEM_BLOCKING} ${ROOT_VTUNE} ${OUT_DIR_VTUNE} ${ARGUMENT} ${BLOCKING_SIZE} ${N_THREADS} ${LOOP_ORDER} ${N_REPETITIONS}
                    rm -rf ${ROOT_VTUNE}/* # Clean tmp data
                done
            done

        done

    done
done
