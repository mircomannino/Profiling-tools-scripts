# Script used to generate roofline performance model

import matplotlib.pyplot as plt
import numpy as np

MACHINES_SPECS = { # Name: [GFLOP/s peack, L1 Gb/s, L2 Gb/s, L3 Gb/s, DRAM Gb/s]
    'Intel_Core_i9-9900K': {
        'peack_GFLOP_s': 460.0, 
        'L1_GB_s': -1, 
        'L2_GB_s':-1, 
        'L3_GB_s':-1, 
        'DRAM_GB_s': 76.8
    }
}

class RooflineCreator:
    def __init__(self, peack_GFLOP_s, peack_GB_s):
        '''
        Args:
            peack_GFLOP_s:      Machine parameter. Max achievable GFLOP/s
            peack_GB_s:         Machine parameter. Max achievable GB/s
        '''
        self.peack_GFLOP_s = peack_GFLOP_s
        self.peack_GB_s = peack_GB_s
    
    def __roofline_function(self, OI):
        return min(self.peack_GFLOP_s, OI*self.peack_GB_s)
    
    def __get_roofline_function_values(self, x_min, x_max, n_values):
        x_values = list(np.linspace(x_min, x_max, n_values))
        y_values = [self.__roofline_function(x) for x in x_values]
        return (x_values, y_values)

    def make_roofline_model(self):
        fig, ax = plt.subplots()

        # Plot the main line of the model
        x_values, y_values = self.__get_roofline_function_values(0, 15, 1000)
        ax.plot(x_values, y_values, 'b')

        self.peack_GB_s = 150
        x_values = list(np.linspace(0, 15, 1000))
        y_values = [min(self.peack_GFLOP_s, x*self.peack_GB_s) for x in x_values]
        ax.plot(x_values, y_values, 'r')

        plt.show()



if __name__=="__main__":    

    peack_GFLOP_s = MACHINES_SPECS['Intel_Core_i9-9900K']['peack_GFLOP_s']
    peack_GB_s =    MACHINES_SPECS['Intel_Core_i9-9900K']['DRAM_GB_s']

    my_roofline_creator = RooflineCreator(peack_GFLOP_s, peack_GB_s)
    my_roofline_creator.make_roofline_model()
