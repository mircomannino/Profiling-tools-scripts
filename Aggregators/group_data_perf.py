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
        print('Perf file in folder:',file_in_folder)
        exit
        # Iterate all the file of the current folder
        for test_file_name in file_in_folder:
            # Remove event suffix
            test_file_name_key = test_file_name.replace('_memory','')
            test_file_name_key = test_file_name.replace('_generalPurpose','')
            print(test_file_name, test_file_name_key)
            exit()
            self.results[test_file_name_key] = {}
            with open(os.path.join(os.getcwd(), test_file_name)) as test_file:
                for line in test_file:
                    if(line.find("branch-misses") != -1):           # BRANCH-MISSES
                        self.results[test_file_name_key]['BRANCH-MISSES'] = self.__get_branch_info(line)
                    if(line.find("instructions") != -1):            # INSTRUCTIONS (--> IPC)
                        self.results[test_file_name_key]['CPI'] = self.__get_cpi_info(line)
                    if(line.find("L1-dcache-loads-misses") != -1):  # L1 DATA CACHE MISSES
                        self.results[test_file_name_key]['L1-MISSES-COUNT'] = self.__get_L1_miss_count(line)
                    if(line.find("LLC-loads-misses") != -1):        # LLC DATA CACHE MISSES
                        self.results[test_file_name_key]['LLC-MISSES-COUNT'] = self.__get_LLC_miss_count(line)
                    if(line.find('instructions') != -1):
                        self.results[test_file_name_key]['N-INSTRUCTIONS'] = self.__get_N_instructions(line)
                    if(line.find('cache-misses') != -1):
                        self.results[test_file_name_key]['CACHE-MISSES-PERCENTAGE'] = self.__get_cache_miss_percentage(line)
                        self.results[test_file_name_key]['CACHE-MISSES-NUMBER'] = self.__get_cache_miss_number(line)

                    if(line.find('cache-references') != -1):
                        self.results[test_file_name_key]['CACHE-REF-NUMBER'] = self.__get_cache_ref_number(line)
                    if(line.find('fp_arith_inst_retired.128b_packed_single') != -1):
                        self.results[test_file_name_key]['N-128b-PACKED-SINGLE'] = self.__get_128b_packed_single(line)
                    if(line.find('fp_arith_inst_retired.256b_packed_single') != -1):
                        self.results[test_file_name_key]['N-256b-PACKED-SINGLE'] = self.__get_256b_packed_single(line)
                self.results[test_file_name_key]['N-256b-PACKED-SINGLE-OVER-N-INSTRUCTIONS'] = float(self.results[test_file_name_key]['N-256b-PACKED-SINGLE'] / self.results[test_file_name]['N-INSTRUCTIONS'])
                self.results[test_file_name_key]['CACHE-OVER-INSTRUCTIONS'] = float(self.results[test_file_name_key]['CACHE-MISSES-NUMBER']) / float(self.results[test_file_name]['N-INSTRUCTIONS'])
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
        if self.__is_number(branch_misses_percentage):
            return float(branch_misses_percentage)
        return -1.0

    def __get_cpi_info(self, line: str):
        splitted_line = line.split()
        ipc = float(splitted_line[3].replace(',', '.'))
        cpi = str(1 / ipc)
        if self.__is_number(cpi):
            return float(cpi)
        return -1.0

    def __get_L1_miss_count(self, line: str):
        splitted_line = line.split()
        L1_misses_count = splitted_line[0].replace('.', '')
        if self.__is_number(L1_misses_count):
            return float(L1_misses_count)
        return -1.0

    def __get_LLC_miss_count(self, line: str):
        splitted_line = line.split()
        LLC_misses_count = splitted_line[0].replace('.', '')
        if self.__is_number(LLC_misses_count):
            return float(LLC_misses_count)
        return -1.0

    def __get_N_instructions(self, line: str):
        splitted_line = line.split()
        LLC_misses_count = splitted_line[0].replace('.', '')
        if self.__is_number(LLC_misses_count):
            return float(LLC_misses_count)
        return -1.0

    def __get_cache_miss_percentage(self, line: str):
        splitted_line = line.split()
        cache_misses_percentage = splitted_line[3].replace(',','.')
        if self.__is_number(cache_misses_percentage):
            return float(cache_misses_percentage)
        return -1.0

    def __get_cache_miss_number(self, line: str):
        splitted_line = line.split()
        cache_misses_number = splitted_line[0].replace('.','')
        if self.__is_number(cache_misses_number):
            return float(cache_misses_number)
        return -1.0

    def __get_cache_ref_number(self, line: str):
        splitted_line = line.split()
        cache_ref_number = splitted_line[0].replace('.','')
        if self.__is_number(cache_ref_number):
            return float(cache_ref_number)
        return -1.0

    def __get_128b_packed_single(self, line: str):
        splitted_line = line.split()
        n_128b_packed_single = splitted_line[0].replace('.', '')
        if self.__is_number(n_128b_packed_single):
            return float(n_128b_packed_single)
        return -1.0

    def __get_256b_packed_single(self, line: str):
        splitted_line = line.split()
        n_256b_packed_single = splitted_line[0].replace('.', '')
        if self.__is_number(n_256b_packed_single):
            return float(n_256b_packed_single)
        return -1.0

    def __is_number(self, str):
        try:
            float(str)
            return True
        except ValueError:
            return False



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
