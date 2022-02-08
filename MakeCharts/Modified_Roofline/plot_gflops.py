from turtle import color
from cv2 import transform
from matplotlib import markers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from gflops_performance import analysis_28B
from peak_interpolation import BlackYeti

if __name__=='__main__':

    # Collect data
    ANALYSIS = {'PHYCORE1_THREAD1': analysis_28B}
    selected_analysis = 'PHYCORE1_THREAD1'
    ORDERS = ['N2', 'N3', 'N4']
    results = {}
    for order_name in ORDERS:
        results[order_name] = {}
        for name_analysis, dict_analysis in ANALYSIS.items():
            results[order_name][name_analysis] = {k:v['GFLOPS_s'] for k,v in dict_analysis[order_name].items()}
        results[order_name] = pd.DataFrame(results[order_name])


    # Plot subplots
    fig, ax = plt.subplots(len(ORDERS), 1, sharey=False, figsize=(15,14))
    

    for i, order_name in enumerate(ORDERS):
        # Actual performance
        results[order_name].plot(
            kind='bar', 
            ax=ax[i], 
            rot=0,
            colormap='Pastel1',
            legend=False,
            zorder=3
        )
        ax[i].grid(axis='y')
        ax[i].set_title('Order ' + order_name)
        ax[i].set_ylabel('Performance [GFLOPs/S]')
        ax[i].set_xlabel('Number of threads')
        ax[i].set_yscale('log')
        from matplotlib.ticker import ScalarFormatter
        for axis in [ax[i].xaxis, ax[i].yaxis]:
            axis.set_major_formatter(ScalarFormatter())

        blackyeti_perf = BlackYeti()
        # Max Performance L1
        max_performance_L1 = {k:blackyeti_perf.L1_interpolation(v['n-cores']) for k,v in ANALYSIS[selected_analysis][order_name].items()}
        x_val = [(k-1) for k,v in max_performance_L1.items()]
        y_val = [v for k,v in max_performance_L1.items()]
        ax[i].plot(x_val, y_val, '+', c='blue', linewidth=2, markersize=12, label='L1 peak', zorder=3)
        # Max Performance L2
        max_performance_L2 = {k:blackyeti_perf.L2_interpolation(v['n-cores']) for k,v in ANALYSIS[selected_analysis][order_name].items()}
        x_val = [(k-1) for k,v in max_performance_L2.items()]
        y_val = [v for k,v in max_performance_L2.items()]
        ax[i].plot(x_val, y_val, '+', c='green', linewidth=2, markersize=12, label='L2 peak', zorder=3)
        # Max Performance L3
        max_performance_L3 = {k:blackyeti_perf.L3_interpolation(v['n-cores']) for k,v in ANALYSIS[selected_analysis][order_name].items()}
        x_val = [(k-1) for k,v in max_performance_L3.items()]
        y_val = [v for k,v in max_performance_L3.items()]
        ax[i].plot(x_val, y_val, '+', c='purple', linewidth=2, markersize=12, label='L3 peak', zorder=3)
        # Max Performance DRAM
        max_performance_DRAM = {k:blackyeti_perf.DRAM_interpolation(v['n-cores']) for k,v in ANALYSIS[selected_analysis][order_name].items()}
        x_val = [(k-1) for k,v in max_performance_DRAM.items()]
        y_val = [v for k,v in max_performance_DRAM.items()]
        ax[i].plot(x_val, y_val, '+', c='red', linewidth=2, markersize=12, label='DRAM peak', zorder=3)


        # Set y range
        # ax[i].set_yticks(np.arange(0, 350, 50))
        # Print values
        for p in ax[i].patches:
            value = np.round(p.get_height(), decimals=2)
            ax[i].annotate(str(value), (p.get_x(), p.get_height()/2), fontsize=10/len(ANALYSIS))

    # Adjust layout
    plt.tight_layout()
    #Legend
    ax[0].legend(loc='upper left', ncol=3, bbox_to_anchor=(0, 1.4))
    # Title
    plt.suptitle('AlexNet performance in multicore machine')
    plt.text(x=0.5, y=0.96, s='Cores utilization: '+selected_analysis, color='grey', ha='center', transform=fig.transFigure)
    fig.subplots_adjust(top=0.9, hspace = .5)
    # Save pdf
    pdf_name = '_'.join([k for k in ANALYSIS.keys()]) + '.pdf'
    plt.savefig('./plots/'+pdf_name)