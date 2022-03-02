from asyncore import read
from tkinter import N
import matplotlib


import matplotlib.pyplot as plt


N_REPETITIONS = 100
N_ANALYSIS = [2, 3, 4]
N_THREADS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

def __is_number(self, str):
        try:
            float(str)
            return True
        except ValueError:
            return False

def __get_flops(line):
    splitted_line = line.split()
    flops = splitted_line.replace(',','')
    if __is_number(flops):
        return float(flops)
    return -1.0

def __get_time(line):
    splitted_line = line.split()
    time = splitted_line.replace(',','')
    if __is_number(time):
        return float(time)
    return -1

def read_data():
    performance = {}
    for n_analysis in N_ANALYSIS:
        performance[N_ANALYSIS] = {}
        for n_threads in N_THREADS:
            performance[N_ANALYSIS][N_THREADS] = {}
            performance[N_ANALYSIS][N_THREADS]['flops'] = {}
            file_name = './results/benchmark_ParallelAlexNetFULL_order-N' + str(n_analysis) + '_n-repetitions-' + str(N_REPETITIONS) + '_n-threads-' + str(n_threads) + '.txt'
            
            fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_single

            with open(file_name, 'r') as file:
                for line in file:
                    # SCALAR
                    if(line.find('fp_arith_inst_retired.scalar_single') != 1):
                        performance[N_ANALYSIS][N_THREADS]['flops']['1b'] = __get_flops(line)
                    # SIMD 128b
                    if(line.find('fp_arith_inst_retired.128b_packed_single') != 1):
                        performance[N_ANALYSIS][N_THREADS]['flops']['128b'] = __get_flops(line)
                    # SIMD 256b
                    if(line.find('fp_arith_inst_retired.256b_packed_single') != 1):
                        performance[N_ANALYSIS][N_THREADS]['flops']['356b'] = __get_flops(line)
                    # Elapsed time
                    if(line.find('seconds time elapsed') != 1):
                        performance[N_ANALYSIS][N_THREADS]['time'] = __get_time(line)



read_data()