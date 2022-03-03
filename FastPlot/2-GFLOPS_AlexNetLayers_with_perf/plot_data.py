import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from peak_interpolation import BlackYeti

##### FONT SIZES #####
FONTSIZE = {
    'REGULAR': 18,
    'TITLE': 30,
    'SUBTITLE': 27,
    'LEGEND': 20,
    'ANNOTATIONS': 14,
    'LABEL': 15
}

ORDER_COLORS = {
    'Order N2 performance': 'purple',
    'Order N3 performance': 'pink',
    'Order N4 performance': 'coral',
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
        15: 8,
        16: 8,
        16: 8,
    }
}


N_REPETITIONS = "1"
N_ANALYSIS = [2, 3, 4]
N_THREADS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# N_THREADS = [1, 2, 4, 8, 16]
LAYERS = [0, 1, 2, 3, 4]

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
        for layer in LAYERS:
            performance[n_analysis][layer] = {}
            for n_threads in N_THREADS:
                performance[n_analysis][layer][n_threads] = {}
                performance[n_analysis][layer][n_threads]['flops'] = {}
                file_name = './results/benchmark_ParallelKernelNKernels_order-N' + str(n_analysis) + '_n-repetitions-' + str(N_REPETITIONS) + '_n-threads-' + str(n_threads) + '_layer-layer-' + str(layer) + '_.txt'
                with open(file_name, 'r') as file:
                    for line in file:
                        # SCALAR
                        if(line.find('fp_arith_inst_retired.scalar_single') != -1):
                            performance[n_analysis][layer][n_threads]['flops']['1b'] = __get_flops(line)
                        # SIMD 128b
                        if(line.find('fp_arith_inst_retired.128b_packed_single') != -1):
                            performance[n_analysis][layer][n_threads]['flops']['128b'] = __get_flops(line)
                        # SIMD 256b
                        if(line.find('fp_arith_inst_retired.256b_packed_single') != -1):
                            performance[n_analysis][layer][n_threads]['flops']['256b'] = __get_flops(line)
                        # Elapsed time
                        if(line.find('seconds time elapsed') != -1):
                            performance[n_analysis][layer][n_threads]['time'] = __get_time(line)
    return performance

def group_data(performance: dict):
    performance_to_plot = {}
    for n_analysis_name, n_analysis_dict in performance.items(): 
        n_analysis_key = 'Order N'+ str(n_analysis_name) + ' performance'
        performance_to_plot[n_analysis_key] = {}
        for layer in LAYERS:
            layer_key = 'layer-' + str(layer)
            performance_to_plot[n_analysis_key][layer_key] = []
            for n_threads, result in n_analysis_dict[layer].items():
                tot_flops = result['flops']['1b'] + 4 * result['flops']['128b'] + 8 * result['flops']['256b']
                time = result['time']
                performance_to_plot[n_analysis_key][layer_key].append((tot_flops / time) / 10**9)
    return performance_to_plot

def plot_data(performance_to_plot: dict):
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(30,15))
    index = np.arange(1,len(N_THREADS)+1)

    results = {n_analysis: pd.DataFrame(result, index=index) for n_analysis, result in performance_to_plot.items()}

    for i, (n_analysis_name, result) in enumerate(results.items()):
         # Plot performance
        result.plot(ax=ax[i], kind='bar', edgecolor='black', rot=0, legend=False, colormap='Set3', zorder=3)

        # Plot roofline
        blackyeti_perf = BlackYeti()
        max_performance_L1 = {k:[blackyeti_perf.L1_interpolation(v)] for k,v in THREAD_CORE_MAP['PHYCORE1_THREAD1'].items()}
        max_performance_L2 = {k:blackyeti_perf.L2_interpolation(v) for k,v in THREAD_CORE_MAP['PHYCORE1_THREAD1'].items()}
        max_performance_L3 = {k:blackyeti_perf.L3_interpolation(v) for k,v in THREAD_CORE_MAP['PHYCORE1_THREAD1'].items()}
        max_performance_DRAM = {k:blackyeti_perf.DRAM_interpolation(v) for k,v in THREAD_CORE_MAP['PHYCORE1_THREAD1'].items()}
        ordered_patches = sorted([p.get_x() for p in list(ax[i].patches)])
        thread_counter = 1
        for j in range(0, len(ordered_patches), 5):
            xmin = ordered_patches[j]
            xmax = ordered_patches[j+4]+0.1
            ax[i].hlines(max_performance_L1[thread_counter], xmin=xmin, xmax=xmax, color='blue', linewidth=3, label='L1 peak', zorder=3)
            ax[i].hlines(max_performance_L2[thread_counter], xmin=xmin, xmax=xmax, color='green', linewidth=3, label='L2 peak', zorder=3)
            ax[i].hlines(max_performance_L3[thread_counter], xmin=xmin, xmax=xmax, color='purple', linewidth=3, label='L3 peak', zorder=3)
            ax[i].hlines(max_performance_DRAM[thread_counter], xmin=xmin, xmax=xmax, color='brown', linewidth=3, label='DRAM peak', zorder=3)
            thread_counter += 1
        
        # Adjust plot
        ax[i].grid('y')
        ax[i].set_yscale('log')
        for axis in [ax[i].yaxis]:
            axis.set_major_formatter(ScalarFormatter())

        ax[i].tick_params(labelsize=FONTSIZE['LABEL'])
        ax[i].set_title(n_analysis_name.replace('performance',''), fontsize=FONTSIZE['REGULAR'])
        ax[i].set_ylabel('Performance [GFLOPS]', fontsize=FONTSIZE['REGULAR'])
        ax[i].set_xlabel('Number of threads', fontsize=FONTSIZE['REGULAR'])


    plt.tight_layout()

    # Legend
    fig.subplots_adjust(top=0.83, hspace = .5)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[0].legend(by_label.values(), by_label.keys(), ncol=9, fontsize=FONTSIZE['LEGEND'], loc='upper left', bbox_to_anchor=(0.0, 1.5))

    title = 'Performance of AlexNet layers and maximum peaks at 0.167 FLOPs/Byte'
    plt.text(x=0.5, y=0.95, s=title, ha="center", transform=fig.transFigure, fontsize=FONTSIZE['TITLE'])
    
    
    plt.savefig('./charts/GFLOPS_per_layer.pdf')

if __name__=='__main__':
    performance = read_data()
    performance_to_plot = group_data(performance)
    plot_data(performance_to_plot=performance_to_plot)