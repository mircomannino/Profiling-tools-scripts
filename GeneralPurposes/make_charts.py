# Python script used to create bar charts from previous analysis
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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

    def make_chart(self, parameter_to_plot, measurement_unit, n_repetitions, tool, compute_best_order, min_is_best, log_scale, normalize):
        '''
        Args:
            parameter_to_plot:      Name of the parameter to use in the charts
            n_repetitions:          Number of repetitions used in the analysisfile_format
            tool:                   Name of the tool from which the results come from [ExecutionTime, Perf, VTune]
            compute_best_order:     Bool to say if you want the loop order that obtain best results
            min_is_best:            Bool needed only if compute_best_order id True
            log_scale:              Bool to say if the plot will be in logaritmic scale
            normalize:              Bool to say if the results will be normalized respect to the first analysis
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

            # Get data folder path
            data_folder = tool + '_'
            data_folder += analysis_directory + '_'
            data_folder += str(n_repetitions) + '-repetitions'
            data_folder = os.path.join(analysis_directory, data_folder, self.data_folder_by_tool[tool])

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
        chart_name = parameter_to_plot + '_' + str(n_repetitions) + '-repetitions_' + tool + ('_normalized_' if normalize else '') + self.file_format
        self.__plot_results(results, chart_name, parameter_to_plot, measurement_unit, log_scale, normalize)

        # Compute the best loop order
        if(compute_best_order):
            best_order_name = chart_name.replace(self.file_format, '')
            best_order_name += 'BEST_ORDER.txt'
            self.__compute_best_order(results, min_is_best, best_order_name, parameter_to_plot)

    def __plot_results(self, results: dict, chart_name, parameter_to_plot, measurement_unit, log_scale, normalize):
        '''
        The dict with results must have in the following format:
            results[benchmark_Naive_10_1_1_3] = {'N1': 0.3, 'N2': 0.3, 'N3': 1.3}
            results[benchmark_Naive_50_1_1_3] = {'N1': 1.2, 'N2': 1.1, 'N3': 2.0}
        '''
        # Normalize the results by Analysis N1
        results_normalized = {}
        if normalize:
            for dimension in results.keys():
                N1_value = float(results[dimension]['N1'])
                results_normalized[dimension] = {
                    n_analysis: (float(value)/N1_value) for (n_analysis, value) in results[dimension].items()
                }
        results = results_normalized if normalize else results

        # Order the name of dimensions (Order: Image size, Image depth, Kernel size, N Kernels)
        name_of_dimensions_ordered = [
            dimension for dimension in 
            sorted(list(results.keys()) ,key=lambda x: (int(x.split('_')[2]), int(x.split('_')[3])))
        ]

        # Order the results by analysis order and get some info for plotting
        results_ordered_by_analysis = {}
        name_of_dimensions = []

        for dimensions in name_of_dimensions_ordered:
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
        ax.grid(axis='y')
        if(log_scale):
            ax.set_yscale('log')
        # index = np.arange(n_groups)
        index = [n for n in range(len(name_of_dimensions))]
        bar_width = 0.09
        opacity = 0.8

        # Populate the plot
        offset = 0
        for n_analysis in results_ordered_by_analysis.keys():
            rects = (plt.bar(
                [(n + offset*bar_width) for n in index],
                results_ordered_by_analysis[n_analysis],
                bar_width,
                color = self.analysis_colors[n_analysis],
                label = n_analysis
            ))
            offset += 1

        # Save the plot
        plt.xlabel('Dimensions')
        plt.ylabel(measurement_unit)
        plt.title(chart_name.replace(self.file_format, ''))
        plt.xticks(
            [(n + bar_width) for n in index],
            [label.replace('benchmark_Naive_', '') for label in name_of_dimensions],
            rotation=90)
        plt.legend(loc='best', fontsize=6)
        plt.tight_layout()
        # plt.show()

        # Save the plot in a file with the appropriate name
        plt.savefig(os.path.join(self.output_path, chart_name))


    def __compute_best_order(self, results, min_is_best, best_order_name, paramater_to_plot):
        '''
        Find for each of the dimensions the best loop order for the parameter of interest
        '''
        best_orders = {}
        best_value = {}
        for dimensions in results.keys():
            if min_is_best:
                best_value[dimensions] = min(results[dimensions].items(), key = lambda x : x[1])[1]
            else:
                best_value[dimensions] = max(results[dimensions].items(), key = lambda x : x[1])[1]
            best_orders[dimensions] = [k for k, v in results[dimensions].items() if v==best_value[dimensions]]

        # Save on file
        with open(os.path.join(self.output_path, best_order_name), 'w+') as out_file:
            # Header
            out_file.write('BEST ORDER LOOPS - ')
            out_file.write(paramater_to_plot)
            out_file.write('\n')
            out_file.write('DIMENSIONS\t\t\t')
            out_file.write('BEST VALUE\t')
            out_file.write('LOOP ORDERS\t\n')
            out_file.write('-'*100)
            out_file.write('\n')
            for dimensions in best_orders:
                # Dimensions
                out_file.write(dimensions)
                out_file.write('\t')
                # Best value
                out_file.write(str(best_value[dimensions])) # value
                out_file.write('\t\t')
                # Best loop order
                for loop_order in sorted(best_orders[dimensions]):
                    # print(dimensions, ': ', loop_order)
                    out_file.write(loop_order) # N
                    out_file.write('\t')
                out_file.write('\n')






if __name__ == "__main__":
    my_chart_creator = ChartsCreator('./charts', file_format='png')
    n_repetitions = 1

    # Execution time
    my_chart_creator.make_chart('TIME-MEDIAN', 'Time (ms)', n_repetitions, 'ExecutionTime', compute_best_order=True, min_is_best=True, log_scale=False, normalize=True)
    
    # Perf
    my_chart_creator.make_chart('BRANCH-MISSES', '% of branches', n_repetitions, 'Perf', compute_best_order=True, min_is_best=True, log_scale=False, normalize=False)
    my_chart_creator.make_chart('CPI', 'CPI', n_repetitions, 'Perf', compute_best_order=True, min_is_best=True, log_scale=False, normalize=False)
    my_chart_creator.make_chart('L1-MISSES-COUNT', 'N. of Misses', n_repetitions, 'Perf', compute_best_order=True, min_is_best=True, log_scale=False, normalize=True)
    my_chart_creator.make_chart('LLC-MISSES-COUNT', 'N. of Misses', n_repetitions, 'Perf', compute_best_order=True, min_is_best=True, log_scale=True, normalize=True)

    # Vtune
    my_chart_creator.make_chart('CPI', 'CPI', n_repetitions, 'VTune', compute_best_order=True, min_is_best=True, log_scale=False, normalize=True)
    my_chart_creator.make_chart('SP_GFLOPS', 'SP_GFLOPS', n_repetitions, 'VTune', compute_best_order=True, min_is_best=False, log_scale=False, normalize=True)

    
