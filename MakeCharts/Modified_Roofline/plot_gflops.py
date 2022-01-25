from matplotlib import markers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from gflops_performance import analysis_28A
from peak_interpolation import BlackYeti

if __name__=='__main__':

    # Collect data
    ANALYSIS = {'ICC-vec': analysis_28A}
    selected_analysis = 'ICC-vec'
    ORDERS = ['N1', 'N2', 'N3', 'N4', 'N5']
    results = {}
    for order_name in ORDERS:
        results[order_name] = {}
        for name_analysis, dict_analysis in ANALYSIS.items():
            results[order_name][name_analysis] = {k:v['GFLOPS_s'] for k,v in dict_analysis[order_name].items()}
        results[order_name] = pd.DataFrame(results[order_name])


    # Plot subplots
    fig, ax = plt.subplots(len(ORDERS), 1, sharey=True, figsize=(13,15))
    

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

        blackyeti_perf = BlackYeti()
        # Max Performance L1
        max_performance_L1 = {k:blackyeti_perf.L1_interpolation(v['n-cores']) for k,v in ANALYSIS[selected_analysis][order_name].items()}
        x_val = [(k-1) for k,v in max_performance_L1.items()]
        y_val = [v for k,v in max_performance_L1.items()]
        ax[i].plot(x_val, y_val, 'd-', c='green', linewidth=2, markersize=4, label='L1 Roof', zorder=3)
        # Max Performance L2
        max_performance_L2 = {k:blackyeti_perf.L2_interpolation(v['n-cores']) for k,v in ANALYSIS[selected_analysis][order_name].items()}
        x_val = [(k-1) for k,v in max_performance_L2.items()]
        y_val = [v for k,v in max_performance_L2.items()]
        ax[i].plot(x_val, y_val, 'd-', c='yellow', linewidth=2, markersize=4, label='L2 Roof', zorder=3)
        # Max Performance L3
        max_performance_L3 = {k:blackyeti_perf.L3_interpolation(v['n-cores']) for k,v in ANALYSIS[selected_analysis][order_name].items()}
        x_val = [(k-1) for k,v in max_performance_L3.items()]
        y_val = [v for k,v in max_performance_L3.items()]
        ax[i].plot(x_val, y_val, 'd-', c='orange', linewidth=2, markersize=4, label='L3 Roof', zorder=3)
        # Max Performance DRAM
        max_performance_DRAM = {k:blackyeti_perf.DRAM_interpolation(v['n-cores']) for k,v in ANALYSIS[selected_analysis][order_name].items()}
        x_val = [(k-1) for k,v in max_performance_DRAM.items()]
        y_val = [v for k,v in max_performance_DRAM.items()]
        ax[i].plot(x_val, y_val, 'd-', c='red', linewidth=2, markersize=4, label='DRAM Roof', zorder=3)


        # Set y range
        ax[i].set_yticks(np.arange(0, 350, 50))
        # Print values
        for p in ax[i].patches:
            value = np.round(p.get_height(), decimals=2)
            ax[i].annotate(str(value), (p.get_x(), p.get_height()/2), fontsize=10/len(ANALYSIS))

    #Legend
    ax[0].legend(loc='upper left', ncol=(2), bbox_to_anchor=(0, 1.15))
    # Common xlabel and ylabel
    ax_big = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    ax_big.set_xlabel('Number of threads', labelpad=10)
    ax_big.set_ylabel('Performance [GFLOPs/S]', labelpad=20)
    # Title
    plt.suptitle('AlexNet performance in multicore machine (1 to 8 threads)')
    # Adjust layout
    plt.tight_layout()
    # Save pdf
    pdf_name = '_'.join([k for k in ANALYSIS.keys()]) + '.pdf'
    plt.savefig('./plots/'+pdf_name)