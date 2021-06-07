# Python script used to group data from the single txt file in the folder
import os
import shutil
import pandas as pd
import csv
import argparse

class AggregatorPerfData:
    '''
    Attributes:
        output_path:        Path to the output file
        results:            Dictionary of dictionaries
                            The first level key is associated to the file name.
                            The second level key is associated to each parameter to store.
                            {
                                'benchmark_Naive_10_3_32_3_16.txt: {branch_misses: 0.45, cpi: 0.56}
                                'benchmark_Naive_50_3_32_3_16.txt: {branch_misses: 0.12, cpi: 0.34}
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
        # Get all the files in the folder and sort them 
        file_in_folder = os.listdir()
        file_in_folder = [file_name for file_name in file_in_folder if file_name.endswith('.txt')]
        file_in_folder.sort(key = lambda x : int(x.split('_')[2].split('.')[0]))
        # Iterate all the file of the current folder
        for test_file_name in file_in_folder:
            self.results[test_file_name] = {}
            with open(os.path.join(os.getcwd(), test_file_name)) as test_file:
                for line in test_file:
                    if(line.find("branch-misses") != -1):           # BRANCH-MISSES
                        self.results[test_file_name]['BRANCH-MISSES'] = self.__get_branch_info(line)
                    if(line.find("instructions") != -1):            # INSTRUCTIONS (--> IPC)
                        self.results[test_file_name]['CPI'] = self.__get_cpi_info(line)
                    if(line.find("L1-dcache-loads-misses") != -1):  # L1 DATA CACHE MISSES
                        self.results[test_file_name]['L1-MISSES-COUNT'] = self.__get_L1_miss_count(line)
                    if(line.find("LLC-loads-misses") != -1):        # LLC DATA CACHE MISSES
                        self.results[test_file_name]['LLC-MISSES-COUNT'] = self.__get_LLC_miss_count(line)
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


    
    def __get_branch_info(self, line: str):
        splitted_line = line.split()
        branch_misses_percentage = splitted_line[3]
        branch_misses_percentage = branch_misses_percentage.replace('%', '')
        branch_misses_percentage = branch_misses_percentage.replace(',', '.')
        return float(branch_misses_percentage)
    
    def __get_cpi_info(self, line: str):
        splitted_line = line.split()
        ipc = float(splitted_line[3].replace(',', '.'))
        cpi = str(1 / ipc)
        return cpi
    
    def __get_L1_miss_count(self, line: str):
        splitted_line = line.split()
        L1_misses_count = splitted_line[0].replace('.', '')
        return float(L1_misses_count)
    
    def __get_LLC_miss_count(self, line: str):
        splitted_line = line.split()
        LLC_misses_count = splitted_line[0].replace('.', '')
        return float(LLC_misses_count)



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

    my_aggregator = AggregatorPerfData(parser.output_path)
    my_aggregator.group_data()
    my_aggregator.save_data_to_csv()