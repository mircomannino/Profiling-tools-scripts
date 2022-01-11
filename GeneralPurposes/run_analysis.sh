#!/bin/bash

######################## CNN Architectures ########################
# AlexNet:
#     "227 3 11 96"
#     "27 96 5 256"
#     "13 384 3 384"
#     "13 384 3 256"
#     "13 256 3 384"

# ResNet-50:
    # "229 3 7 64"
    # "112 64 1 64"
    # "112 64 3 64"
    # "56 64 1 256"
    # "56 256 1 128"
    # "56 128 3 128"
    # "28 128 1 512"
    # "28 512 1 256"
    # "28 256 3 256"
    # "14 256 1 1024"
    # "14 1024 1 512"
    # "14 512 3 512"
    # "7 512 1 2048"

# Officials:
    # "229 3 7 64"
    # "112 64 1 64"
    # "112 64 3 64"
    # "56 64 1 256"
    # "56 256 1 128"
    # "56 128 3 128"
    # "28 128 1 512"
    # "28 512 1 256"
    # "28 256 3 256"
    # "14 256 1 1024"
    # "14 1024 1 512"
    # "14 512 3 512"
    # "7 512 1 2048"
    # "27 96 5 256"
    # "13 256 3 384"
    # "13 384 3 256"
    # "28 512 3 1024"
    # "14 1024 3 1024"
    # "7 1024 3 512"
    # "7 1024 3 2048"
    # "7 2048 3 2048"
###################################################################

declare -a list_of_analysis=("1" "2" "3" "4" "5")

declare -a n_repetitions=("5")

declare -a arguments=( # (Image size, Image depth, Kernel size, N. Kernel)
    "229 3 7 64"
    "112 64 1 64"
    "112 64 3 64"
    "56 64 1 256"
    "56 256 1 128"
    "56 128 3 128"
    "28 128 1 512"
    "28 512 1 256"
    "28 256 3 256"
    "14 256 1 1024"
    "14 1024 1 512"
    "14 512 3 512"
    "7 512 1 2048"
    "27 96 5 256"
    "13 256 3 384"
    "13 384 3 256"
    "28 512 3 1024"
    "14 1024 3 1024"
    "7 1024 3 512"
    "7 1024 3 2048"
    "7 2048 3 2048"
)


BINARY_FILE=./bin/benchmark_SequentialAlexNetFULL

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
            rm -rf ${ROOT_VTUNE}/* # Clean tmp data
        done
    done

    for N_REPETITIONS in "${n_repetitions[@]}"
    do
        # Roofline
        ROOT_DATA_ANALYSIS=${CURRENT_DIR}/Roofline_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/roofline-data
        OUT_DIR_ROOFLINE=${CURRENT_DIR}/Roofline_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/roofline-reports
        ./analysis_N${NUMBER_ANALYSYS}/Roofline_analysis_N${NUMBER_ANALYSYS}_${N_REPETITIONS}-repetitions/roofline_report_generator.sh \
        ${BINARY_FILE} ${ROOT_DATA_ANALYSIS} ${OUT_DIR_ROOFLINE} ${LOOP_ORDER} ${N_REPETITIONS}
        rm -rf ${ROOT_DATA_ANALYSIS}/*
    done
done
