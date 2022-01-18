# Script used to generate roofline performance model

import matplotlib.pyplot as plt
import numpy as np
from machines_specs import INTEL_CORE_i9_9900K
from CNNs_results import ALEX_NET_27A, ALEX_NET_27B, ALEX_NET_27C

PERFORMANCE_COLORS = {
    'SCALAR_GFLOPS_s': 'purple',
    'SP_VECTOR_GFLOPS_s': 'blue',
    'SP_FMA_GFLOPS_s': 'orange'
}

ORDER_COLORS = {
    'order-1': 'blue',
    'order-2': 'orange',
    'order-3': 'green',
    'order-4': 'red',
    'order-5': 'purple'
}

class RooflineCreator:
    def __init__(self, SCALAR_GFLOPS_s, SP_VECTOR_GFLOPS_s, SP_FMA_GFLOPS_s, L1_GB_s, L2_GB_s, L3_GB_s, DRAM_GB_s):
        '''
        Args:
            # TODO:
        '''
        self.performance_peacks = {
            'SCALAR_GFLOPS_s': float(SCALAR_GFLOPS_s),
            'SP_VECTOR_GFLOPS_s': float(SP_VECTOR_GFLOPS_s),
            'SP_FMA_GFLOPS_s': float(SP_FMA_GFLOPS_s)
        }
        self.bandwidth_peacks = {
            'L1_GB_s': float(L1_GB_s),
            'L2_GB_s': float(L2_GB_s),
            'L3_GB_s': float(L3_GB_s),
            'DRAM_GB_s': float(DRAM_GB_s)
        }
        self.OI_ridge_points = self.__compute_OI_ridge_points()

    def __compute_OI_ridge_points(self):
        '''
        Compute OI of each possible ridge point between performance peacks
        and bandwidth peacks.
        '''
        OI_ridge_points = {}
        for bandwidth_name, bandwidth_peack in self.bandwidth_peacks.items():
            OI_ridge_points[bandwidth_name] = {}
            for performance_name, performance_peack in self.performance_peacks.items():
                OI_ridge_point = performance_peack * (1./bandwidth_peack)
                OI_ridge_points[bandwidth_name][performance_name] = OI_ridge_point
        return OI_ridge_points

    def __plot_bandwidth_peack(self, ax, bandwidth_name, bandwidth_peack, x_min, x_max):
        x_values = list(np.linspace(x_min, x_max, 2))
        y_values = [x*bandwidth_peack for x in x_values]
        ax.plot(x_values, y_values, label=bandwidth_name, linewidth=1, alpha=0.7, color='black')
        # Plot label near the line
        # label_on_plot = bandwidth_name.replace('_GB_s','') + ' bandwidth peack: ' + str(bandwidth_peack) + 'GB/s'
        # middle_x = 0.1
        # ax.text(middle_x, (bandwidth_peack*middle_x*1.2), label_on_plot, rotation=45)

    def __plot_performance_peack(self, ax, performance_name, performance_peack, x_min, x_max):
        x_values = list(np.linspace(x_min, x_max, 2))
        y_values = [performance_peack]*2
        # ax.plot(x_values, y_values, label=performance_name, color=PERFORMANCE_COLORS[performance_name], linewidth=2)
        ax.plot(x_values, y_values, label=performance_name, linewidth=1, alpha=0.7, color='black')
        # Plot label near the line
        label_on_plot = performance_name.replace('GFLOPS_s','') + 'performance peack: ' + str(performance_peack) + 'GFLOPs/s'
        ax.text(x_max*0.2, performance_peack*1.05, label_on_plot, color=PERFORMANCE_COLORS[performance_name], horizontalalignment='left')

    def make_roofline_model(self, ax, show_ridge_points=True):
        # Set the plot
        ax.set_yscale('log')
        ax.set_ylabel('GFLOPS/s')
        ax.set_xscale('log')
        ax.set_xlabel('FLOPS/Byte')
        ax.grid()

        # Plot ridge points
        if(show_ridge_points):
            for bandwidth_name, bandwidth_OI_ridge_points in self.OI_ridge_points.items():
                for performance_name, OI_ridge_point in bandwidth_OI_ridge_points.items():
                    x = OI_ridge_point
                    y = self.performance_peacks[performance_name]
                    ax.scatter(x, y, s=100, marker='x', color=PERFORMANCE_COLORS[performance_name])


        # Plot bandwidth lines
        for bandwidth_name, bandwidth_peack in self.bandwidth_peacks.items():
            self.__plot_bandwidth_peack(
                ax = ax,
                bandwidth_name = bandwidth_name,
                bandwidth_peack = bandwidth_peack,
                x_min = 0,
                x_max = max([v for k,v in self.OI_ridge_points[bandwidth_name].items()])
            )

        # Plot performance lines
        for performance_name, performance_peack in self.performance_peacks.items():
            # search the min OI for the perfmance peack line
            OI_min = float('inf')
            for _, bandwidth_OI_ridge_points in self.OI_ridge_points.items():
                if(bandwidth_OI_ridge_points[performance_name] < OI_min):
                    OI_min = bandwidth_OI_ridge_points[performance_name]
            self.__plot_performance_peack(
                ax = ax,
                performance_name = performance_name,
                performance_peack = performance_peack,
                x_min = OI_min,
                x_max = 100
            )


        return fig, ax

    def add_measurements(self, ax, name, OI, GFLOPS_s, color, marker):
        ax.scatter(OI, GFLOPS_s, label=name, s=100, color=color, marker=marker)


if __name__=="__main__":

    my_roofline_creator_1CORES = RooflineCreator(
        INTEL_CORE_i9_9900K['1-cores']['performance']['SCALAR_GFLOPS_s'],
        INTEL_CORE_i9_9900K['1-cores']['performance']['SP_VECTOR_GFLOPS_s'],
        INTEL_CORE_i9_9900K['1-cores']['performance']['SP_FMA_GFLOPS_s'],
        INTEL_CORE_i9_9900K['1-cores']['bandwidth']['L1_GB_s'],
        INTEL_CORE_i9_9900K['1-cores']['bandwidth']['L2_GB_s'],
        INTEL_CORE_i9_9900K['1-cores']['bandwidth']['L3_GB_s'],
        INTEL_CORE_i9_9900K['1-cores']['bandwidth']['DRAM_GB_s'],
    )

    fig, ax = plt.subplots()
    my_roofline_creator_1CORES.make_roofline_model(ax, show_ridge_points=False)

    # Add AlexNet ICC vec benchmark
    for order_number, order_benchmark in ALEX_NET_27A['results'].items():
        my_roofline_creator_1CORES.add_measurements(
            ax,
            order_number,
            order_benchmark['OI'],
            order_benchmark['GFLOPS_s'],
            color = ORDER_COLORS[order_number],
            marker = 's'
        )

     # Add AlexNet ICC no-vec benchmark
    for order_number, order_benchmark in ALEX_NET_27B['results'].items():
        my_roofline_creator_1CORES.add_measurements(
            ax,
            order_number,
            order_benchmark['OI'],
            order_benchmark['GFLOPS_s'],
            color = ORDER_COLORS[order_number],
            marker = 'x')

     # Add AlexNet POLLY benchmark
    for order_number, order_benchmark in ALEX_NET_27C['results'].items():
        my_roofline_creator_1CORES.add_measurements(
            ax,
            order_number,
            order_benchmark['OI'],
            order_benchmark['GFLOPS_s'],
            color = ORDER_COLORS[order_number],
            marker = '*')

    plt.legend()
    plt.show()
