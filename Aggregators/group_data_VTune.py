# Python script used to group data from the single txt file in the folder
import os
import shutil
import pandas as pd
import csv
import argparse

class AggregatorVTuneData:
    '''
    output_path:        Path to the output directory
    results:                Dictionary of dictionaries
                            The first level key is associated to the file name.
                            The second level key is associated to each parameter to store.
                            {
                                'benchmark_Naive_10.txt: {branch_misses: 0.45, cpi: 0.56}
                                'benchmark_Naive_50.txt: {branch_misses: 0.12, cpi: 0.34}
                            }
    '''

    def __init__(self, output_path):
        '''
        Args:
            output_path:    Path to the output directory
        '''
        self.results = {}
        self.output_path = output_path + ".csv"

    def __to_float(self, value):
        '''
        Returns the float value or 0.0 if the value is not valid
        '''
        try:
            return float(value)
        except ValueError:
            return 0.

    def group_data(self):
        print('\n===============================================================================')
        print('Grouping data of: ', self.output_path)
        # file to open for each test
        parameter_files = [
            'summary_hpc-performance.csv',
            'summary_uarch-exploration.csv',
            'summary_memory-access.csv'
        ]
        # Get all the files in the folder and sort them
        subdirectories = os.listdir()
        subdirectories = [subdirectory for subdirectory in subdirectories if (os.path.isdir(subdirectory) and subdirectory.find('benchmark')!=-1)]
        subdirectories.sort(key = lambda x : int(x.split('_')[2].split('.')[0]))

        # Iterate all the file of the current folder
        for subdirectory in subdirectories:
            self.results[subdirectory] = {}
            for parameter_file in parameter_files:
                with open(os.path.join(os.getcwd(), subdirectory, parameter_file)) as test_file:
                    csv_reader = csv.reader(test_file, delimiter=',')
                    next(csv_reader)    # Skip header row
                    for line in csv_reader:
                        if len(line) == 1:
                            line = line[0].split('\t')  # ['Hierarchy level \t Metric Name \t Metric Value'] --> ['Hierarchy Level', 'Metric Name', 'Metric Value']
                        metric_name = line[1]

                        # # TODO: change all the "if statements"

                        ### HPC PERFORMANCE ###
                        if(parameter_file == 'summary_hpc-performance.csv'):
                            if(metric_name.find('SP GFLOPS') != -1):
                                self.results[subdirectory]['SP_GFLOPS'] = self.__to_float(line[2])
                            if(metric_name.find('CPI') != -1):
                                self.results[subdirectory]['CPI'] = self.__to_float(line[2])
                            if(metric_name.find('FP Arith/Mem Rd Instr. Ratio') != -1):
                                self.results[subdirectory]['FP_OP-OVER-MEM_READ'] = self.__to_float(line[2])
                            if(metric_name.find('FP Arith/Mem Wr Instr. Ratio') != -1):
                                self.results[subdirectory]['FP_OP-OVER-MEM_WRITE'] = self.__to_float(line[2])
                            if(metric_name.find('Cache Bound') != -1):
                                self.results[subdirectory]['CACHE-BOUND'] = self.__to_float(line[2])
                            if(metric_name.find('Average CPU Frequency') != -1):
                                self.results[subdirectory]['AVG-CPU-FREQUENCY(GHz)'] = self.__to_float(line[2]) / 10**9


                        ### UARCH EXPLORATION ###
                        if(parameter_file == 'summary_uarch-exploration.csv'):
                            if(metric_name.find('Branch Mispredict') != -1):
                                self.results[subdirectory]['BRANCH-MISSES'] = self.__to_float(line[2])
                            if(metric_name.find('Vector Capacity Usage (FPU)') != -1):
                                self.results[subdirectory]['VECTOR-CAPACITY-USAGE'] = self.__to_float(line[2])
                            if(metric_name.find('Memory Latency') != -1):
                                self.results[subdirectory]['MEMORY-LATENCY'] = self.__to_float(line[2])
                            if(metric_name.find('Front-End Bound') != -1):
                                self.results[subdirectory]['FRONT-END-BOUND'] = self.__to_float(line[2])
                            if(metric_name.find('Back-End Bound') != -1):
                                self.results[subdirectory]['BACK-END-BOUND'] = self.__to_float(line[2])
                            if(metric_name.find('Retiring') != -1):
                                self.results[subdirectory]['RETIRING'] = self.__to_float(line[2])
                            if(metric_name.find('Cycles of 0 Ports Utilized') != -1):
                                self.results[subdirectory]['CYCLES-0-PORTS-UTILIZED'] = self.__to_float(line[2])
                            if(metric_name.find('Cycles of 1 Port Utilized') != -1):
                                self.results[subdirectory]['CYCLES-1-PORT-UTILIZED'] = self.__to_float(line[2])
                            if(metric_name.find('Cycles of 2 Ports Utilized') != -1):
                                self.results[subdirectory]['CYCLES-2-PORTS-UTILIZED'] = self.__to_float(line[2])
                            if(metric_name.find('Cycles of 3+ Ports Utilized') != -1):
                                self.results[subdirectory]['CYCLES-3+-PORTS-UTILIZED'] = self.__to_float(line[2])
                            if(metric_name.find('FB Full') != -1):
                                self.results[subdirectory]['FB-FILL'] = self.__to_float(line[2])
                            if(metric_name.find('Core Bound') != -1):
                                self.results[subdirectory]['CORE-BOUND'] = self.__to_float(line[2])
                            if(metric_name.find('DTLB Overhead') != -1):
                                self.results[subdirectory]['DTLB-OVERHEAD'] = self.__to_float(line[2])
                            if(metric_name.find('Load Operation Utilization') != -1):
                                self.results[subdirectory]['LOAD-OPERATION-UTILIZATION'] = self.__to_float(line[2])
                            if(metric_name.find('Store Operation Utilization') != -1):
                                self.results[subdirectory]['STORE-OPERATION-UTILIZATION'] = self.__to_float(line[2])
                            if(metric_name.find('ALU Operation Utilization') != -1):
                                self.results[subdirectory]['ALU-OPERATION-UTILIZATION'] = self.__to_float(line[2])
                            if(metric_name.find('Bad Speculation') != -1):
                                self.results[subdirectory]['BAD-SPECULATION'] = self.__to_float(line[2])
                            if(metric_name.find('Machine Clears') != -1):
                                self.results[subdirectory]['MACHINE-CLEARS'] = self.__to_float(line[2])


                        ### MEMORY ACCESS ###
                        if(parameter_file == 'summary_memory-access.csv'):
                            if(metric_name.find('L1 Bound') != -1):
                                self.results[subdirectory]['L1-BOUND'] = self.__to_float(line[2])
                            if(metric_name.find('L2 Bound') != -1):
                                self.results[subdirectory]['L2-BOUND'] = self.__to_float(line[2])
                            if(metric_name.find('L3 Bound') != -1):
                                self.results[subdirectory]['L3-BOUND'] = self.__to_float(line[2])
                            if(metric_name.find('LLC Miss Count') != -1):
                                self.results[subdirectory]['LLC-MISSES-COUNT'] = self.__to_float(line[2])
                            if(metric_name.find('DRAM Bound') != -1):
                                self.results[subdirectory]['DRAM-BOUND'] = self.__to_float(line[2])
                            if(metric_name.find('Memory Bound') != -1):
                                self.results[subdirectory]['MEMORY-BOUND'] = self.__to_float(line[2])

        # Show the final collected data
        print('Data grouped!')
        for file_name, parameters in self.results.items():
            print(file_name)
            for parameter, val in parameters.items():
                print('\t', parameter, ': ', val, sep='')

    def save_data_to_csv(self):
        if(len(self.results) != 0):
            df = pd.DataFrame(self.results)
            df = df.T # Transpose
            df.to_csv(self.output_path)

def create_parser():
    '''
    Function used to parse the arguments of the command line
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_path', '-o', type=str, required=True,
                        help='Output path))')
    return parser.parse_args()

if __name__ == '__main__':
    parser = create_parser()
    if parser.output_path == "":
        print('Error: insert a valid output path')
        exit()

    my_aggregator = AggregatorVTuneData(parser.output_path)
    my_aggregator.group_data()
    my_aggregator.save_data_to_csv()
