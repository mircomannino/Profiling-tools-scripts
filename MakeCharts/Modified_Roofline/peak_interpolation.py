# Script used to compute the maximum perforamance peak for a given number of core 
from configparser import Interpolation
import numpy as np
import matplotlib.pyplot as plt

class BlackYeti:
    def __init__(self):
        # Initialize with the known bandwidth peaks
        L1 = { # Max bandwidth [GB/s]
            '1-cores': 425.28,
            '2-cores': 851.16,
            '4-cores': 1702.32,
            '6-cores': 2553.47,
            '8-cores': 3404.63
        }
        L2 = { # Max bandwidth [GB/s]
            '1-cores': 150.44,
            '2-cores': 300.87,
            '4-cores': 601.74,
            '6-cores': 902.47,
            '8-cores': 1203.63
        }
        L3 = { # Max bandwidth [GB/s]
            '1-cores': 67.85,
            '2-cores': 135.7,
            '4-cores': 271.39,
            '6-cores': 407.09,
            '8-cores': 542.79
        }
        DRAM = { # Max bandwidth [GB/s]
            '1-cores': 23.52,
            '2-cores': 27.35,
            '4-cores': 30.86,
            '6-cores': 31.23,
            '8-cores': 31.36
        }

        # Initialize max peaks at 0.16-0.17 FLOPs/Byte (OI)
        self.max_OI = {}
        self.max_OI['L1'] = {}
        self.max_OI['L2'] = {}
        self.max_OI['L3'] = {}
        self.max_OI['DRAM'] = {}
        self.OI = 0.167
        for n_cores in L1.keys():
            self.max_OI['L1'][n_cores] = L1[n_cores] * self.OI
            self.max_OI['L2'][n_cores] = L2[n_cores] * self.OI
            self.max_OI['L3'][n_cores] = L3[n_cores] * self.OI
            self.max_OI['DRAM'][n_cores] = DRAM[n_cores] * self.OI
        
        print(self.max_OI)

    def L1_interpolation(self, n_cores):
        x_val = np.array([1, 2, 4, 6, 8]) # Number of cores
        y_val = np.array([v for k,v in self.max_OI['L1'].items()])
        return np.interp(n_cores, x_val, y_val)
    
    def L2_interpolation(self, n_cores):
        x_val = np.array([1, 2, 4, 6, 8]) # Number of cores
        y_val = np.array([v for k,v in self.max_OI['L2'].items()])
        return np.interp(n_cores, x_val, y_val)
    
    def L3_interpolation(self, n_cores):
        x_val = np.array([1, 2, 4, 6, 8]) # Number of cores
        y_val = np.array([v for k,v in self.max_OI['L3'].items()])
        return np.interp(n_cores, x_val, y_val)

    def DRAM_interpolation(self, n_cores):
        x_val = np.array([1, 2, 4, 6, 8]) # Number of cores
        y_val = np.array([v for k,v in self.max_OI['DRAM'].items()])
        return np.interp(n_cores, x_val, y_val)

if __name__=='__main__':
    b = BlackYeti()

    # Demo of the interpolation
    x = np.array([1, 2, 4, 6, 8])
    y_L1 = b.L1_interpolation(x)
    y_L2 = b.L2_interpolation(x)
    y_L3 = b.L3_interpolation(x)
    y_DRAM = b.DRAM_interpolation(x)

    plt.plot(x, y_L1, 'ro-', label='L1')
    plt.plot(x, y_L2, 'bo-', label='L2')
    plt.plot(x, y_L3, 'go-', label='L3')
    plt.plot(x, y_DRAM, 'yo-', label='DRAM')

    plt.grid('y')

    plt.title('Max Performance GFLOPs/S at 0.167 FLOPs/Byte')
    plt.ylabel('Performance [GFLOPs/S]')
    plt.xlabel('Number of cores')
    plt.legend()
    plt.show()
