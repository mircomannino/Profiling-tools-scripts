# Python script used to group data from the single txt file in the folder
import os
import pandas as pd
import csv
import argparse


class AggregatorRoofline:
    '''
    Attributes:
        output_path:        Path to the output file
        results:            Dictionary of dictionaries
                            The first level key is associated to the file name.
                            The second level key is associated to each parameter to store.
                            {
                                'benchmark_Naive_1_1_50_DEFAULT.txt: {median_time: 0.45, division_time: 0.56}
                                'benchmark_Naive_2_1_50_DEFAULT.txt: {median_time: 0.12, division_time: 0.34}
                            }
        nLayers:            Dict with the nunmber of layers of each network tested (default AlexNet)
        
    '''
    def __init__(self, output_path):
        '''
        Args:
            output_path:    Path to the output directory
        '''
        self.results = {}
        self.output_path = output_path + ".csv"
        self.nLayers = {
            'AlexNet': 5
        }

    def group_data(self):
        print('\n===============================================================================')
        print('Grouping data of: ', self.output_path)
        # Get all the files in the folder and sort them
        file_in_folder = os.listdir()
        file_in_folder = [file_name for file_name in file_in_folder if file_name.endswith('.csv')]
        if self.output_path in file_in_folder: file_in_folder.remove(self.output_path)
        file_in_folder.sort(key = lambda x : int(x.split('_')[2].split('.')[0]))
        # Iterate all the file of the current folder
        for test_file_name in file_in_folder:
            with open(os.path.join(os.getcwd(), test_file_name)) as test_file:
                self.results[test_file_name] = {}
                N_THREADS = float(test_file_name.split('_')[2]) # benchmark_ParallelAlexNetFULL_1_2_10_PHYCORE1_THREAD1.csv -> 1
                N_REPETITIONS = float(test_file_name.split('_')[4]) # benchmark_ParallelAlexNetFULL_1_2_10_PHYCORE1_THREAD1.csv -> 10
                reader = csv.reader(test_file)
                # Skip first 6 rows
                for i in range(5): reader.__next__()
                header = reader.__next__()
                for i, h in enumerate(header):
                    print(i, h)
                for line in reader:
                    if(len(line)>1 and line[1].find('convolve') != -1): # Enter only in function results (es. convolveThread)
                        if line[20] != '' and float(line[20].replace(',','.')) != 0.:
                            gflops = float(line[20].replace(',','.')) / self.nLayers['AlexNet'] / N_REPETITIONS / N_THREADS
                            self.results[test_file_name]['GFLOPS'] = gflops
                            self.results[test_file_name]['Debug'] = float(line[20].replace(',','.'))

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

    def __get_time_info(self, line):
        splitted_line = line.split()
        execution_time = splitted_line[4]
        return float(execution_time)

def create_parser():
    '''
    Function used to parse the arguments of the command line
    '''
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_path', '-o', type=str, required=True,
                        help='Output path))')
    parser.add_argument('--full-network', action='store_true',
                        help='Use this flag when data is from a FULL network analysis')
    return parser.parse_args()

if __name__ == '__main__':
    parser = create_parser()
    if parser.output_path == "":
        print('Error: insert a valid output path')
        exit()

    my_aggregator = AggregatorRoofline(parser.output_path)
    my_aggregator.group_data()
    my_aggregator.save_data_to_csv()
