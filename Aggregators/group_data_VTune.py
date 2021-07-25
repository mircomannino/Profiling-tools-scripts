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
                    for line in csv_reader:
                        line = line[0].split('\t')  # line --> ['Hierarchy Level', 'Metric Name', 'Metric Value']
                        metric_name = line[1]
                        metric_value = (float(line[2]) if len(line == 3) else 0.) # because some time VTune does not put results

                        # # TODO: change all the "if statements"

                        ### HPC PERFORMANCE ###
                        if(parameter_file == 'summary_hpc-performance.csv'):
                            if(metric_name.find('SP GFLOPS') != -1):
                                self.results[subdirectory]['SP_GFLOPS'] = metric_value
                            if(metric_name.find('CPI') != -1):
                                self.results[subdirectory]['CPI'] = metric_value

                        ### UARCH EXPLORATION ###
                        if(parameter_file == 'summary_uarch-exploration.csv'):
                            if(metric_name.find('Branch Mispredict') != -1):
                                self.results[subdirectory]['BRANCH-MISSES'] = metric_value
                            if(metric_name.find('Vector Capacity Usage (FPU)') != -1):
                                self.results[subdirectory]['VECTOR-CAPACITY-USAGE'] = metric_value
                            if(metric_name.find('Memory Latency') != -1):
                                self.results[subdirectory]['MEMORY-LATENCY'] = metric_value
                            if(metric_name.find('Front-End Bound') != -1):
                                self.results[subdirectory]['FRONT-END-BOUND'] = metric_value
                            if(metric_name.find('Back-End Bound') != -1):
                                self.results[subdirectory]['BACK-END-BOUND'] = metric_value
                            if(metric_name.find('Retiring') != -1):
                                self.results[subdirectory]['RETIRING'] = metric_value
                            if(metric_name.find('Cycles of 3+ Ports Utilized') != -1):
                                self.results[subdirectory]['CYCLES-3+-PORTS-UTILIZED'] = metric_value

                        ### MEMORY ACCESS ###
                        if(parameter_file == 'summary_memory-access.csv'):
                            if(metric_name.find('L1 Bound') != -1):
                                self.results[subdirectory]['L1-BOUND'] = metric_value
                            if(metric_name.find('L2 Bound') != -1):
                                self.results[subdirectory]['L2-BOUND'] = metric_value
                            if(metric_name.find('L3 Bound') != -1):
                                self.results[subdirectory]['L3-BOUND'] = metric_value
                            if(metric_name.find('LLC Miss Count') != -1):    
                                self.results[subdirectory]['LLC-MISSES-COUNT'] = metric_value
                            if(metric_name.find('Memory Bound') != -1):
                                self.results[subdirectory]['MEMORY-BOUND'] = metric_value

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
