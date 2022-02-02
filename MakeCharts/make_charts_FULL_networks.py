# Python script used to create bar charts from previous analysis
import pandas as pd
import numpy as np
import os
import argparse
import json
import matplotlib.pyplot as plt

class ChartsCreator:
    '''
    Attributes:
        ouput_path:     Path of the folder in which the results will be stored
        file_format:    Format in which tha files will be saved (pdf, png, ...)
    '''
    def __init__(self, output_path, file_format):
        self.output_path = output_path
        self.file_format = '.' + file_format
        self.analysis_colors = {
            'N1': 'red',
            'N2': 'blue',
            'N3': 'green',
            'N4': 'orange',
            'N5': 'purple',
            'N6': 'brown',
            'N7': 'pink',
            'N8': 'olive',
            'N9': 'cyan'
        }
        self.data_folder_by_tool = {
            'ExecutionTime': 'execution_times',
            'Perf': 'perf_reports',
            'VTune': 'reports'
        }

    def make_chart(self, parameter_to_plot, measurements_unit, n_repetitions, tool, title=None, alloc_type=None):
        '''
        Args:
            parameter_to_plot:      List with the names of the parameters to use in the charts
            n_repetitions:          Number of repetitions used in the analysisfile_format
            tool:                   Name of the tool from which the results come from [ExecutionTime, Perf, VTune]
            title:                  Title of the charts
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

            if n_analysis not in results.keys():
                results[n_analysis] = {}

            # Get data folder path
            data_folder = tool + '_'
            data_folder += analysis_directory + '_'
            data_folder += str(n_repetitions) + '-repetitions'
            data_folder = os.path.join(analysis_directory, data_folder, self.data_folder_by_tool[tool])

            # Read the csv file in a DataFrame
            benchmarks_data_path = [file_ for file_ in os.listdir(data_folder) if file_.endswith('.csv')][0]
            benchmarks_data = pd.read_csv(os.path.join(data_folder, benchmarks_data_path))
        
            for i, row in benchmarks_data.iterrows():
                benchmarks_info = row[0].split('_')
                n_threads = benchmarks_info[2]
                layer_id  = benchmarks_info[6].replace('LAYER','')

                if layer_id not in results[n_analysis]:
                    results[n_analysis][layer_id] = {}
                
                results[n_analysis][layer_id][n_threads] = row[parameter_to_plot]

        # Normalize
        results_normalized = {}
        for analisys_name, analysis_dict in results.items():
            results_normalized[analisys_name] = {}
            for layer_name, layer_dict in analysis_dict.items():
                results_normalized[analisys_name][layer_name] = {}
                reference_value = layer_dict['1']   # The reference value the one for one thread
                for n_thread_name, n_thread_value in layer_dict.items():
                    results_normalized[analisys_name][layer_name][n_thread_name] = results[analisys_name][layer_name][n_thread_name] / reference_value
        results = results_normalized

        # Create the DataFrames
        results_df = {n_analysis_name: pd.DataFrame(n_analysis_dict) for n_analysis_name, n_analysis_dict in results.items()}

        # Prepare the plot
        N_ROWS = len(results.keys()) # One row for each analysis: N1, N2, ...
        N_COLS = 1
        fig, ax = plt.subplots(nrows=N_ROWS, ncols=N_COLS, figsize=(15,14))

        # Plot
        for i, (n_analysis,result_df) in enumerate(results_df.items()):
            result_df.T.plot(ax=ax[i], kind='bar', rot=0, legend=False)
            
            # Setup subplots
            ax[i].grid('y')
            ax[i].set_title('Order: '+n_analysis)
            ax[i].set_ylabel(measurements_unit)
            ax[i].set_xlabel('Layer ID')

        # Adjust subplots
        ax[0].legend(fontsize=10, ncol=16, loc='upper left', bbox_to_anchor=(0, 1.35))

        # Finalization 
        chart_name = parameter_to_plot + '_' + str(n_repetitions) + '-repetitions_' + tool + self.file_format
        plt.text(x=0.5, y=0.96, s=title, ha="center", transform=fig.transFigure)
        plt.text(x=0.5, y=0.95, s='Normalized with respect to 1-thread version', color='grey', ha="center", transform=fig.transFigure)
        plt.text(x=0.5, y=0.94, s='Cores utilization: '+alloc_type, color='grey', ha="center", transform=fig.transFigure)
        fig.subplots_adjust(top=0.9, hspace = .5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, chart_name))
        print(chart_name, ': Done')


if __name__ == "__main__":
    # Create a parser for arguments
    parser = argparse.ArgumentParser(description='Make charts.')
    parser.add_argument('--output-folder', '-o', type=str, default='./charts', help='destination folder. default: ./charts')
    parser.add_argument('--output-type', '-t', type=str, default='pdf', help='format of output charts [png, pdf]. default: èdf')
    parser.add_argument('--n-repetitions', '-n', type=int, default=5, help='Number of repetitions used in the benchmarks. default=5')
    args = parser.parse_args()

    my_chart_creator = ChartsCreator(args.output_folder, file_format=args.output_type)
    n_repetitions = args.n_repetitions

    my_chart_creator.make_chart('TIME-MEDIAN', 'Time [ms]', n_repetitions, 'ExecutionTime', title="TIME-MEDIAN AlexNet layers [1 to 16 threads]", alloc_type='DEFAULT')