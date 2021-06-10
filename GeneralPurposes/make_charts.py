# Python script used to create bar charts from previous analysis
import pandas as pd
import numpy as np
import os
import re 
import matplotlib.pyplot as plt

class ChartsCreator:
    '''
    Attributes:
        ouput_path:     Path of the folder in which the results will be stored
    '''
    def __init__(self, output_path):
        self.output_path = output_path
        self.analysis_colors = {
            'N1': 'g',
            'N2': 'b',
            'N3': 'r',
            'N4': '00FFFF',
            'N5': '0000FF',
            'N6': 'FF00FF',
            'N7': '800080',
            'N8': 'FF7F50'
        }
        self.data_folder_by_tool = {
            'ExecutionTime': 'execution_times',
            'Perf': 'perf_reports',
            'VTune': 'reports'
        }
    
    def make_chart(self, parameter_to_plot, n_repetitions, tool):
        '''
        Args:
            parameter_to_plot:      Name of the parameter to use in the charts
            n_repetitions:          Number of repetitions used in the analysis
            tool:                   Name of the tool from which the results come from [ExecutionTime, Perf, VTune]
        '''
        # Get all the folders with analysis
        analysis_directories = os.listdir()
        analysis_directories = [analysis_directory for analysis_directory in analysis_directories if (
            os.path.isdir(analysis_directory) and analysis_directory.find('analysis_N')!=-1)]
        
        # Collect data from each analysis folder
        results = {}
        for analysis_directory in analysis_directories:
            # Get Number of the analysis
            n_analysis = analysis_directory.replace('analysis_', '')    # 'analysis_N1' ---> 'N1'
            print(n_analysis)

            # Get data folder path 
            data_folder = tool + '_'
            data_folder += analysis_directory + '_'
            data_folder += str(n_repetitions) + '-repetitions'
            data_folder = os.path.join(analysis_directory, data_folder, self.data_folder_by_tool[tool]) 
            print(data_folder)

            # Read the csv file in a DataFrame
            benchmarks_data_path = [file_ for file_ in os.listdir(data_folder) if file_.endswith('.csv')][0]
            benchmarks_data = pd.read_csv(os.path.join(data_folder, benchmarks_data_path))

            # Get only the column of interest 
            for index, row in benchmarks_data.iterrows():
                # Get info from dataframe
                dimensions = row[0].replace('.txt', '')[:-4]     # benchmark_Naive_x_x_x_x_x_x.txt ---> benchmark_Naive_x_x_x_x
                value = row[parameter_to_plot]
                # Store values
                if dimensions not in results.keys():
                    results[dimensions] = {}
                results[dimensions][n_analysis] = value 

        # Plot results
        self.__plot_results(results, 'MIRCO')

    def __plot_results(self, results: dict, name: str):
        '''
        The dict with results must have in the following format:
            results[benchmark_Naive_10_1_1_3] = {'N1': 0.3, 'N2': 0.3, 'N3': 1.3}
            results[benchmark_Naive_50_1_1_3] = {'N1': 1.2, 'N2': 1.1, 'N3': 2.0}
        '''
        # Order the results by analysis order and get some info for plotting
        results_ordered_by_analysis = {}
        name_of_dimensions = []
        for dimensions in results.keys():
            for n_analysis, value in sorted(results[dimensions].items()):
                # Append the result
                if n_analysis not in results_ordered_by_analysis:
                    results_ordered_by_analysis[n_analysis] = []
                results_ordered_by_analysis[n_analysis].append(value)
                # Append the name of the dimensions
                if dimensions not in name_of_dimensions:
                    name_of_dimensions.append(dimensions)

        # Create the plot
        n_groups = len(name_of_dimensions)
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.1
        opacity = 0.8

        # Populate the plot
        offset = 0
        for n_analysis in results_ordered_by_analysis.keys():
            rects = (plt.bar(
                index + offset*bar_width, 
                results_ordered_by_analysis[n_analysis],
                bar_width,
                alpha = opacity,
                color = self.analysis_colors[n_analysis],
                label = n_analysis 
            ))
            offset += 1

        # Save the plot
        plt.xlabel('Dimensions')
        plt.ylabel('Time (seconds)')
        plt.title('Execution time')
        plt.xticks(index + bar_width/2, name_of_dimensions)
        plt.legend()

        plt.tight_layout()
        plt.show()

        # TODO: Save the plot in a file with the appropriate name



        
        

if __name__ == "__main__":
    my_chart_creator = ChartsCreator('./charts')
    my_chart_creator.make_chart('TIME-MEDIAN', 1, 'ExecutionTime')

    