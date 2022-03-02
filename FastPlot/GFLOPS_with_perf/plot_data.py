import matplotlib.pyplot as plt
import numpy as np
from peak_interpolation import BlackYeti

##### FONT SIZES #####
FONTSIZE = {
    'REGULAR': 14,
    'TITLE': 20,
    'SUBTITLE': 27,
    'LEGEND': 12,
    'ANNOTATIONS': 18
}

THREAD_CORE_MAP = {
    'PHYCORE1_THREAD1': {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5, 
        6: 6,
        7: 7,
        8: 8,
        9: 8,
        10: 8,
        11: 8,
        12: 8,
        13: 8,
        14: 8,
        16: 8,
        16: 8,
    }
}


N_REPETITIONS = ""
N_ANALYSIS = [2, 3, 4]
N_THREADS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

def __is_number(str):
        try:
            float(str)
            return True
        except ValueError:
            return False

def __get_flops(line):
    splitted_line = line.split()
    flops = splitted_line[0].replace('.','')
    if __is_number(flops):
        return float(flops)
    return -1.0

def __get_time(line):
    splitted_line = line.split()
    time = splitted_line[0].replace(',','.')
    if __is_number(time):
        return float(time)
    return -1

def read_data():
    performance = {}
    for n_analysis in N_ANALYSIS:
        performance[n_analysis] = {}
        for n_threads in N_THREADS:
            performance[n_analysis][n_threads] = {}
            performance[n_analysis][n_threads]['flops'] = {}
            file_name = './results/benchmark_ParallelAlexNetFULL_order-N' + str(n_analysis) + '_n-repetitions-' + str(N_REPETITIONS) + '_n-threads-' + str(n_threads) + '.txt'
            with open(file_name, 'r') as file:
                for line in file:
                    # SCALAR
                    if(line.find('fp_arith_inst_retired.scalar_single') != -1):
                        performance[n_analysis][n_threads]['flops']['1b'] = __get_flops(line)
                    # SIMD 128b
                    if(line.find('fp_arith_inst_retired.128b_packed_single') != -1):
                        performance[n_analysis][n_threads]['flops']['128b'] = __get_flops(line)
                    # SIMD 256b
                    if(line.find('fp_arith_inst_retired.256b_packed_single') != -1):
                        performance[n_analysis][n_threads]['flops']['256b'] = __get_flops(line)
                    # Elapsed time
                    if(line.find('seconds time elapsed') != -1):
                        performance[n_analysis][n_threads]['time'] = __get_time(line)
    return performance

def group_data(performance: dict):
    performance_to_plot = {}
    for n_analysis_name, n_analysis_dict in performance.items(): 
        n_analysis_key = 'Order N'+ str(n_analysis_name) + ' performance'
        performance_to_plot[n_analysis_key] = []  
        for n_threads, result in n_analysis_dict.items():
            tot_flops = result['flops']['1b'] + 4 * result['flops']['128b'] + 8 * result['flops']['256b']
            time = result['time']
            performance_to_plot[n_analysis_key].append((tot_flops / time) / 10**9)
    return performance_to_plot

def plot_data(performance_to_plot: dict):
    fig, ax = plt.subplots(figsize=(16,7))
    for order_name, p in performance_to_plot.items():
        x_val = [i+1 for i in range(len(p))]
        y_val = p
        ax.plot(x_val, y_val, marker='o', linewidth=2, markersize=8, label=order_name)
    
    # Plot roofline
    blackyeti_perf = BlackYeti()
    # Max Performance L1
    max_performance_L1 = {k:blackyeti_perf.L1_interpolation(v) for k,v in THREAD_CORE_MAP['PHYCORE1_THREAD1'].items()}
    x_val = [k for k,v in max_performance_L1.items()]
    y_val = [v for k,v in max_performance_L1.items()]
    ax.plot(x_val, y_val, '--', c='blue', linewidth=2, markersize=12, label='L1 peak', zorder=3)
    # Max Performance L2
    max_performance_L2 = {k:blackyeti_perf.L2_interpolation(v) for k,v in THREAD_CORE_MAP['PHYCORE1_THREAD1'].items()}
    x_val = [k for k,v in max_performance_L2.items()]
    y_val = [v for k,v in max_performance_L2.items()]
    ax.plot(x_val, y_val, '--', c='green', linewidth=2, markersize=12, label='L2 peak', zorder=3)
    # Max Performance L3
    max_performance_L3 = {k:blackyeti_perf.L3_interpolation(v) for k,v in THREAD_CORE_MAP['PHYCORE1_THREAD1'].items()}
    x_val = [k for k,v in max_performance_L3.items()]
    y_val = [v for k,v in max_performance_L3.items()]
    ax.plot(x_val, y_val, '--', c='purple', linewidth=2, markersize=12, label='L3 peak', zorder=3)
    # Max Performance DRAM
    max_performance_DRAM = {k:blackyeti_perf.DRAM_interpolation(v) for k,v in THREAD_CORE_MAP['PHYCORE1_THREAD1'].items()}
    x_val = [k for k,v in max_performance_DRAM.items()]
    y_val = [v for k,v in max_performance_DRAM.items()]
    ax.plot(x_val, y_val, '--', c='red', linewidth=2, markersize=12, label='DRAM peak', zorder=3)
    
    ax.grid('y')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=FONTSIZE['LEGEND'], ncol=1)
    ax.set_xticks(np.arange(17))

    ax.set_ylabel('Performance [GFLOPS]', fontsize=FONTSIZE['REGULAR'])
    ax.set_xlabel('Number of threads', fontsize=FONTSIZE['REGULAR'])

    ax.set_title('Performance of AlexNet and maximum peaks at 0.167 FLOPs/Byte', fontsize=FONTSIZE['TITLE'], pad=20)

    plt.tight_layout()
    plt.savefig('./charts/GFLOPS.pdf')

if __name__=='__main__':
    performance = read_data()
    performance_to_plot = group_data(performance)
    plot_data(performance_to_plot=performance_to_plot)