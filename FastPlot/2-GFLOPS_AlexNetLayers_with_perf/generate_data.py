import subprocess
import os

def bash_command(command: str):
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

n_analysis=["2", "3", "4"]
n_threads=["1", "2", "3", "4", "5", "6", "7", "8", "9", "9", "10", "11", "12", "13", "14", "15", "16"]
convolution_info={ # <input-H/W> <input-C> <kernel-H/W> <kernel-N> <stride>
    'layer-0': '227 3 11 96 4',
    'layer-1': '55 96 5 265 2',
    'layer-2': '26 256 3 384 1',
    'layer-3': '24 384 3 384 1',
    'layer-4': '22 384 3 256 1'
}

PERF_EVENTS=("fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.scalar_single")

ALLOCATION_TYPE = "PHYCORE1_THREAD1"
CONVOLUTION_N_REPETITIONS = "1"
PERF_N_REPETITIONS = "100"

BIN="./bin/benchmark_ParallelKernelNKernels"

OUT_DIR="./results"
bash_command("mkdir -p  {}".format(OUT_DIR))

n_iter = 0
for N_ANALYSIS in n_analysis:
    for N_THREADS in n_threads:
        for LAYER_ID, CONVOLUTION_INFO in convolution_info.items():
            OUT_NAME = '_'.join([
                os.path.basename(BIN),
                'order-N'+N_ANALYSIS,
                'n-repetitions-'+CONVOLUTION_N_REPETITIONS,
                'n-threads-'+N_THREADS,
                'layer-'+LAYER_ID,
                '.txt'
            ])

            ARGUMENTS = ' '.join([
                CONVOLUTION_INFO,
                N_THREADS,
                N_ANALYSIS,
                CONVOLUTION_N_REPETITIONS,
                ALLOCATION_TYPE
            ])

            bash_command(
                'perf stat -o {} -r {} -e {} {} {}'.format(
                    os.path.join(OUT_DIR,OUT_NAME),
                    PERF_N_REPETITIONS,
                    PERF_EVENTS,
                    BIN,
                    ARGUMENTS
                )
            )
            
            print("[{}/{}]".format(n_iter, len(n_analysis)*len(n_threads)*len(convolution_info.keys())))
            n_iter += 1
