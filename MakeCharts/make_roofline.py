# Script used to generate roofline performance model

import matplotlib.pyplot as plt
import numpy as np
from machines_specs import INTEL_CORE_i9_9900K

PERFORMANCE_COLORS = {
    'SCALAR_GFLOPS_s': 'purple',
    'SP_VECTOR_GFLOPS_s': 'blue',
    'SP_FMA_GFLOPS_s': 'orange'
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
        ax.plot(x_values, y_values, label=bandwidth_name)

    def __plot_performance_peack(self, ax, performance_name, performance_peack, x_min, x_max):
        x_values = list(np.linspace(x_min, x_max, 2))
        y_values = [performance_peack]*2
        ax.plot(x_values, y_values, label=performance_name, color=PERFORMANCE_COLORS[performance_name])

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
                    ax.scatter(x, y, s=80, marker='x', color=PERFORMANCE_COLORS[performance_name])


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


        # # Plot the main line of the model
        # x_values, y_values = self.__get_roofline_function_values(0, 100, 1000)
        # ax.plot(x_values, y_values, 'b')
        #
        # self.peack_GFLOP_s = 800
        # x_values = list(np.linspace(0, 100, 1000))
        # y_values = [min(self.peack_GFLOP_s, x*self.peack_GB_s) for x in x_values]
        # ax.plot(x_values, y_values, 'r')

        return fig, ax

    def add_measurements(self, ax, name, OI, GFLOPS_s):
        ax.scatter(OI, GFLOPS_s, label=name, s=100)


if __name__=="__main__":

    my_roofline_creator_8CORES = RooflineCreator(
        INTEL_CORE_i9_9900K['8-cores']['performance']['SCALAR_GFLOPS_s'],
        INTEL_CORE_i9_9900K['8-cores']['performance']['SP_VECTOR_GFLOPS_s'],
        INTEL_CORE_i9_9900K['8-cores']['performance']['SP_FMA_GFLOPS_s'],
        INTEL_CORE_i9_9900K['8-cores']['bandwidth']['L1_GB_s'],
        INTEL_CORE_i9_9900K['8-cores']['bandwidth']['L2_GB_s'],
        INTEL_CORE_i9_9900K['8-cores']['bandwidth']['L3_GB_s'],
        INTEL_CORE_i9_9900K['8-cores']['bandwidth']['DRAM_GB_s'],
    )

    fig, ax = plt.subplots()
    my_roofline_creator_8CORES.make_roofline_model(ax)
    my_roofline_creator_8CORES.add_measurements(ax, 'ICC-Parallel-order2', 0.01, 10)

    plt.legend()
    plt.show()
