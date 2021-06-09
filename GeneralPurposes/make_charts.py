# Python script used to create bar charts from previous analysis
import pandas as pd
import numpy as np
import os
import re

class ChartsCreator:
    '''
    Attributes:
        ouput_path:     Path of the folder in which the results will be stored
    '''
    def __init__(self, output_path):
        self.output_path = output_path
        self.analysis_colors = {
            1: 'FFFF00',
            2: '808000',
            3: '008000',
            4: '00FFFF',
            5: '0000FF',
            6: 'FF00FF',
            7: '800080',
            8: 'FF7F50'
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
            data_folder = tool + '_'
            data_folder += analysis_directory + '_'
            data_folder += str(n_repetitions) + '-repetitions'
            print(os.path.join(analysis_directory, data_folder, self.data_folder_by_tool[tool]))
            
    

        

if __name__ == "__main__":
    my_chart_creator = ChartsCreator('./charts')
    my_chart_creator.make_chart('TIME-MEDIAN', 1, 'ExecutionTime')

    