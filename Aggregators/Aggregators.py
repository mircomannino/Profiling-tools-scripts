# Python script used to group data from the single txt file in the folder
import os
import shutil
import pandas as pd
import csv


class AggregatorExecutionTime:
    '''
    Attributes:
        output_dir_path:    Path to the output directory
        results:            Dictionary of dictionaries
                            The first level key is associated to the file name.
                            The second level key is associated to each parameter to store.
                            {
                                'benchmark_Naive_10.txt: {median_time: 4.0, division_time: 1.2}
                                'benchmark_Naive_50.txt: {median_time: 4.0, division_time: 1.2}
                            }
    '''
    def __init__(self, output_dir_path):
        '''
        Args:
            output_dir_path:    Path to the output directory
        '''
        self.output_dir_path = output_dir_path
        self.results = {}

        # Remove the output folder if already exists
        if(os.path.isdir(self.output_dir_path)):
            print('deleting ', self.output_dir_path, ' ...')
            shutil.rmtree(self.output_dir_path)
        
        # Create the new folder
        os.mkdir(self.output_dir_path) 

        # Set the root for output data
        self.root = os.path.join(os.getcwd(), self.output_dir_path)
    
    
    def group_data(self):
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

class AggregatorVTuneData:
    '''
    output_dir_path:        Path to the output directory
    results:                Dictionary of dictionaries
                            The first level key is associated to the file name.
                            The second level key is associated to each parameter to store.
                            {
                                'benchmark_Naive_10.txt: {branch_misses: 0.45, cpi: 0.56}
                                'benchmark_Naive_50.txt: {branch_misses: 0.12, cpi: 0.34}
                            }
    '''

    def __init__(self, output_dir_path):
        '''
        Args:
            output_dir_path:    Path to the output directory
        '''
        self.output_dir_path = output_dir_path
        self.results = {}

        # Remove the output folder if already exists
        if(os.path.isdir(self.output_dir_path)):
            print('deleting ', self.output_dir_path, ' ...')
            shutil.rmtree(self.output_dir_path)
        
        # Create the new folder
        os.mkdir(self.output_dir_path) 

        # Set the root for output data
        self.root = os.path.join(os.getcwd(), self.output_dir_path)
    
    def group_data(self):
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

        subdirectories = ['benchmark_Naive_10', 'benchmark_Naive_10']
        # Iterate all the file of the current folder
        for subdirectory in subdirectories:
            self.results[subdirectory] = {}
            for parameter_file in parameter_files:
                with open(os.path.join(os.getcwd(), subdirectory, parameter_file)) as test_file:
                    csv_reader = csv.reader(test_file, delimiter=',')
                    for line in test_file:
                        if(parameter_file == 'summary_hpc-performance.csv'):
                            if(line.find('SP GFLOPS') != -1):   # GFLOPS
                                self.results[subdirectory]['SP_GFLOPS'] = line.split()[3]
                            if(line.find('CPI') != -1):         # CPI
                                self.results[subdirectory]['CPI'] = line.split()[3]
                        if(parameter_file == 'summary_uarch-exploration.csv'):
                            if(line.find('Branch Mispredict') != -1):   # BRANCH-MISSES
                                self.results[subdirectory]['BRANCH-MISSES'] = line.split()[3]
                        if(parameter_file == 'summary_memory-access.csv'):
                            if(line.find('L1 Bound') != -1):     # L1 Bound
                                self.results[subdirectory]['L1-BOUND'] = line.split()[3]
                            if(line.find('L2 Bound') != -1):     # L2 Bound
                                self.results[subdirectory]['L2-BOUND'] = line.split()[3]
                            if(line.find('L2 Bound') != -1):     # L2 Bound
                                self.results[subdirectory]['L2-BOUND'] = line.split()[3]
                            if(line.find('LLC Miss Count') != -1):     # LLC Misses-COUNT
                                self.results[subdirectory]['LLC-MISSES-COUNT'] = line.split()[4]

        # Show the final collected data
        print('Data grouped!')
        for file_name, parameters in self.results.items():
            print(file_name)
            for parameter, val in parameters.items():
                print('\t', parameter, ': ', val, sep='')
        
    def save_data_to_csv(self, output_prefix=''):
        if(len(self.results) != 0):
            df = pd.DataFrame(self.results)
            df = df.T # Transpose 

            file_name = output_prefix + '_grouped_data.csv'
            df.to_csv(os.path.join(self.root, file_name))



class AggregatorPerfData:
    '''
    Attributes:
        output_dir_path:    Path to the output directory
        results:            Dictionary of dictionaries
                            The first level key is associated to the file name.
                            The second level key is associated to each parameter to store.
                            {
                                'benchmark_Naive_10.txt: {branch_misses: 0.45, cpi: 0.56}
                                'benchmark_Naive_50.txt: {branch_misses: 0.12, cpi: 0.34}
                            }
    '''
    def __init__(self, output_dir_path):
        '''
        Args:
            output_dir_path:    Path to the output directory
        '''
        self.output_dir_path = output_dir_path
        self.results = {}

        # Remove the output folder if already exists
        if(os.path.isdir(self.output_dir_path)):
            print('deleting ', self.output_dir_path, ' ...')
            shutil.rmtree(self.output_dir_path)
        
        # Create the new folder
        os.mkdir(self.output_dir_path) 

        # Set the root for output data
        self.root = os.path.join(os.getcwd(), self.output_dir_path)
    
    
    def group_data(self):
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
    
    def save_data_to_csv(self, output_prefix=''):
        if(len(self.results) != 0):
            df = pd.DataFrame(self.results)
            df = df.T # Transpose 

            file_name = output_prefix + '_grouped_data.csv'
            df.to_csv(os.path.join(self.root, file_name))


    
    def __get_branch_info(self, line: str):
        splitted_line = line.split()
        branch_misses_percentage = splitted_line[3]
        branch_misses_percentage = branch_misses_percentage.replace('%', '')
        branch_misses_percentage = branch_misses_percentage.replace(',', '.')
        return branch_misses_percentage
    
    def __get_cpi_info(self, line: str):
        splitted_line = line.split()
        ipc = float(splitted_line[3].replace(',', '.'))
        cpi = str(1 / ipc)
        return cpi
    
    def __get_L1_miss_count(self, line: str):
        splitted_line = line.split()
        L1_misses_count = splitted_line[0].replace('.', '')
        return L1_misses_count
    
    def __get_LLC_miss_count(self, line: str):
        splitted_line = line.split()
        LLC_misses_count = splitted_line[0].replace('.', '')
        return LLC_misses_count




if __name__ == '__main__':
    my_aggregator = AggregatorVTuneData('./grouped_data')
    my_aggregator.group_data()
    my_aggregator.save_data_to_csv('Analysis_N1_Perf_1-repetions')