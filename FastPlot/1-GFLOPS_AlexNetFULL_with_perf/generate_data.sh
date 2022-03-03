#!/bin/bash

declare -a n_analysis=("2" "3" "4")
declare -a n_threads=("1" "2" "3" "4" "5" "6" "7" "8" "9" "9" "10" "11" "12" "13" "14" "15" "16")

PERF_EVENTS=("fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.scalar_single")

ALLOCATION_TYPE="PHYCORE1_THREAD1"
CONVOLUTION_N_REPETITIONS=1
PERF_N_REPETITIONS=100

BIN=./bin/benchmark_ParallelAlexNetFULL

OUT_DIR=./results
mkdir -p ${OUT_DIR}

for N_ANALYSIS in ${n_analysis[@]}; do
    for N_THREADS in ${n_threads[@]}; do
        OUT_NAME=$(basename ${BIN})_order-N${N_ANALYSIS}_n-repetitions-${CONVOLUTION_N_REPETITIONS}_n-threads-${N_THREADS}

        ARGUMENTS="${N_THREADS} ${N_ANALYSIS} ${CONVOLUTION_N_REPETITIONS} ${ALLOCATION_TYPE}"
        echo ${ARGUMENTS}
        perf stat -o ${OUT_DIR}/${OUT_NAME}.txt -r ${PERF_N_REPETITIONS} -e ${PERF_EVENTS} ${BIN} ${ARGUMENTS}
    done
done
~                 