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
                dimensions = dimensions.replace('benchmark_Naive_', '')
                value = row[parameter_to_plot]
                # Store values
                if n_analysis not in results.keys():
                    results[n_analysis] = {}
                results[n_analysis][dimensions] = value

        # Normalize
        if(normalize):
            N1_values = {dim: val for (dim, val) in results['N1'].items()}
            for n_analysis in results.keys():
                for dimensions in results[n_analysis].keys():
                    results[n_analysis][dimensions] /= N1_values[dimensions]

        # Order the name of dimensions (Order: Image size, Image depth, Kernel size, N Kernels)
        results_ordered = {}
        for n_analysis in sorted(results.keys()):
            results_ordered[n_analysis] = {}
            for dimensions in sorted(list(results[n_analysis].keys()) ,key=lambda x: (int(x.split('_')[0]), int(x.split('_')[3]))):
                results_ordered[n_analysis][dimensions] = results[n_analysis][dimensions]
        results = results_ordered
        del results_ordered

        # Plot parameters
        plt.rcParams["figure.figsize"] = [20,9]
        font = {'family' : 'DejaVu Sans',
        # 'weight' : 'bold',
        'size'   : 30}
        plt.rc('font', **font)

        # Plot results
        result_df = pd.DataFrame(results)
        ax = result_df.plot.bar(width=0.9, alpha=0.6, edgecolor='black', linewidth=2)
        ax.grid(axis='y')
        # ax.axvline(x=4.5)
        plt.xticks(ha='right', rotation=30)
        if(log_scale):
            ax.set_yscale('log')

        chart_name = parameter_to_plot + '_' + str(n_repetitions) + '-repetitions_' + tool + ('_normalized_' if normalize else '') + self.file_format
        # plt.xlabel('x_y_z_v:  x = height and width of INPUT;  y = n. of channels of INPUT;  z = height and width of KERNEL; v = n. of KERNELS')
        plt.xlabel('Dimensions')
        plt.ylabel(measurement_unit)
        plt.title(chart_name.replace(self.file_format, ''))
        # plt.title('CPI (Cycles per instruction) - Changing the number of kernels')
        # plt.legend(loc='best', fontsize=20)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=20)
        plt.tight_layout()
        # plt.show()

        # Save the plot in a file with the appropriate name
        plt.savefig(os.path.join(self.output_path, chart_name))
        # self.__plot_results(results, chart_name, parameter_to_plot, measurement_unit, log_scale, normalize)


    def make_chart_double(self, parameters_to_plot, measurement_unit, n_repetitions, tools, compute_best_order, min_is_best, log_scale, normalize):
        '''
        Args:
            parameter_to_plot:      List with the names of the parameters to use in the charts
            n_repetitions:          Number of repetitions used in the analysisfile_format
            tools:                  List wirh the names of the tools from which the results come from [ExecutionTime, Perf, VTune]
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
        results_parameter_1 = {}
        results_parameter_2 = {}
        for analysis_directory in analysis_directories:
            # Get Number of the analysis
            n_analysis = analysis_directory.replace('analysis_', '')    # 'analysis_N1' ---> 'N1'

            ### Get data folder of parameter 1 ###
            # Get data folder path
            data_folder_parameter_1 = tools[0] + '_'
            data_folder_parameter_1 += analysis_directory + '_'
            data_folder_parameter_1 += str(n_repetitions) + '-repetitions'
            data_folder_parameter_1 = os.path.join(analysis_directory, data_folder_parameter_1, self.data_folder_by_tool[tools[0]])
            # Read the csv file in a DataFrame
            benchmarks_data_path_parameter_1 = [file_ for file_ in os.listdir(data_folder_parameter_1) if file_.endswith('.csv')][0]
            benchmarks_data_parameter_1 = pd.read_csv(os.path.join(data_folder_parameter_1, benchmarks_data_path_parameter_1))

            ### Get data folder of parameter 2 ###
            # Get data folder path
            data_folder_parameter_2 = tools[1] + '_'
            data_folder_parameter_2 += analysis_directory + '_'
            data_folder_parameter_2 += str(n_repetitions) + '-repetitions'
            data_folder_parameter_2 = os.path.join(analysis_directory, data_folder_parameter_2, self.data_folder_by_tool[tools[1]])
            # Read the csv file in a DataFrame
            benchmarks_data_path_parameter_2 = [file_ for file_ in os.listdir(data_folder_parameter_2) if file_.endswith('.csv')][0]
            benchmarks_data_parameter_2 = pd.read_csv(os.path.join(data_folder_parameter_2, benchmarks_data_path_parameter_2))

            # Get only the column of interest - Parameter 1
            for index, row in benchmarks_data_parameter_1.iterrows():
                # Get info from dataframe
                dimensions = row[0].replace('.txt', '')[:-4]     # benchmark_Naive_x_x_x_x_x_x.txt ---> benchmark_Naive_x_x_x_x
                dimensions = dimensions.replace('benchmark_Naive_', '')
                value = row[parameters_to_plot[0]]
                # Store values
                if n_analysis not in results_parameter_1.keys():
                    results_parameter_1[n_analysis] = {}
                results_parameter_1[n_analysis][dimensions] = value
            
            # Get only the column of interest - Parameter 2
            for index, row in benchmarks_data_parameter_2.iterrows():
                # Get info from dataframe
                dimensions = row[0].replace('.txt', '')[:-4]     # benchmark_Naive_x_x_x_x_x_x.txt ---> benchmark_Naive_x_x_x_x
                dimensions = dimensions.replace('benchmark_Naive_', '')
                value = row[parameters_to_plot[1]]
                # Store values
                if n_analysis not in results_parameter_2.keys():
                    results_parameter_2[n_analysis] = {}
                results_parameter_2[n_analysis][dimensions] = value
    

        if(normalize):
            # Normalize - Parameter 1
            N1_values = {dim: val for (dim, val) in results_parameter_1['N1'].items()}
            for n_analysis in results_parameter_1.keys():
                for dimensions in results_parameter_1[n_analysis].keys():
                    results_parameter_1[n_analysis][dimensions] /= N1_values[dimensions]
            # Normalize - Parameter 2
            N1_values = {dim: val for (dim, val) in results_parameter_2['N1'].items()}
            for n_analysis in results_parameter_2.keys():
                for dimensions in results_parameter_2[n_analysis].keys():
                    results_parameter_2[n_analysis][dimensions] /= N1_values[dimensions]

        # Order the name of dimensions (Order: Image size, Image depth, Kernel size, N Kernels)
        results_ordered_parameter_1 = {}
        results_ordered_parameter_2 = {}
        for n_analysis in sorted(results_parameter_1.keys()):
            results_ordered_parameter_1[n_analysis] = {}
            results_ordered_parameter_2[n_analysis] = {}
            for dimensions in sorted(list(results_parameter_1[n_analysis].keys()) ,key=lambda x: (int(x.split('_')[0]), int(x.split('_')[3]))):
                results_ordered_parameter_1[n_analysis][dimensions] = results_parameter_1[n_analysis][dimensions]
                results_ordered_parameter_2[n_analysis][dimensions] = results_parameter_2[n_analysis][dimensions]
        results_parameter_1 = results_ordered_parameter_1
        results_parameter_2 = results_ordered_parameter_2
        del results_ordered_parameter_1
        del results_ordered_parameter_2

        # Plot parameters
        plt.rcParams["figure.figsize"] = [20,9]
        plt.rcParams["figure.autolayout"] = True
        font = {'family' : 'DejaVu Sans',
        # 'weight' : 'bold',
        'size'   : 25}
        plt.rc('font', **font)

        # Merge two dataframes

        # Plot results
        result_df_parameter_1 = pd.DataFrame(results_parameter_1)
        result_df_parameter_2 = pd.DataFrame(results_parameter_2)
        ax = result_df_parameter_2.plot.bar(width=0.9, alpha=0.6, edgecolor='black', linewidth=2)
        result_df_parameter_1.plot.bar(width=0.9, ax=ax,  color='grey', alpha=0.2, align='center', edgecolor='black', linewidth=2)
        # ax.grid(axis='y')
        # ax.axvline(x=4.5)
        plt.xticks(ha='right', rotation=30)
        if(log_scale):
            ax.set_yscale('log')

        chart_name = parameters_to_plot[0] + '_' + parameters_to_plot[1] + '_' + str(n_repetitions) + '-repetitions_' + str(tools) + ('_normalized_' if normalize else '') + self.file_format
        # plt.xlabel('x_y_z_v:  x = height and width of INPUT;  y = n. of channels of INPUT;  z = height and width of KERNEL; v = n. of KERNELS')
        plt.xlabel('Dimensions')
        plt.ylabel(measurement_unit)
        plt.title(chart_name.replace(self.file_format, ''))
        # plt.title('CPI (Cycles per instruction) - Changing the number of kernels')
        # plt.legend(loc='best', fontsize=20)
        plt.legend(
            [str(n_analysis) for n_analysis in list(results_parameter_2.keys())] +  ['Exec Time'], 
            bbox_to_anchor=(1, 1), 
            loc='upper left', 
            fontsize=15
        )
        plt.tight_layout()
        # plt.show()

        # Save the plot in a file with the appropriate name
        plt.savefig(os.path.join(self.output_path, chart_name))
        # self.__plot_results(results, chart_name, parameter_to_plot, measurement_unit, log_scale, normalize)
    
    
    def make_chart_stacked(self, parameters_to_plot, measurements_unit, n_repetitions, tool):
        '''
        Args:
            parameter_to_plot:      List with the names of the parameters to use in the charts
            n_repetitions:          Number of repetitions used in the analysisfile_format
            tool:                   Name of the tool from which the results come from [ExecutionTime, Perf, VTune]
        '''
        # Get all the folders with analysis
        analysis_directories = os.listdir()
        analysis_directories = [analysis_directory for analysis_directory in analysis_directories if (
            os.path.isdir(analysis_directory) and analysis_directory.find('analysis_N')!=-1)]

        # Collect data from each analysis folder
        results = {}
        for parameter in parameters_to_plot:
            results[parameter] = {}
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
                dimensions = dimensions.replace('benchmark_Naive_', '')
                for parameter in parameters_to_plot:
                    value = row[parameter]
                    # Store values
                    if n_analysis not in results[parameter].keys():
                        results[parameter][n_analysis] = {}
                    results[parameter][n_analysis][dimensions] = value

        # Order the name of dimensions (Order: Image size, Image depth, Kernel size, N Kernels)
        for parameter in parameters_to_plot:
            results_ordered = {}
            for n_analysis in sorted(results[parameter].keys()):
                results_ordered[n_analysis] = {}
                for dimensions in sorted(list(results[parameter][n_analysis].keys()) ,key=lambda x: (int(x.split('_')[0]), int(x.split('_')[3]))):
                    results_ordered[n_analysis][dimensions] = results[parameter][n_analysis][dimensions]
            results[parameter] = results_ordered
            del results_ordered

        # Plot parameters
        plt.rcParams["figure.figsize"] = [20,9]
        font = {'family' : 'DejaVu Sans',
        # 'weight' : 'bold',
        'size'   : 30}
        plt.rc('font', **font)

        # Plot results
        result_dfs = [pd.DataFrame(parameter_dict) for parameter_name, parameter_dict in results.items()] 
        print(type(result_dfs))
        self.__plot_clustered_stacked(dfall=result_dfs, labels=['a', 'b', 'c'])
        # ax = plt.gca()
        # for result_df in result_dfs:
        #     result_df.plot(kind='bar', stacked=True, ax=ax, width=0.5, position=0)
        # # ax = result_df.plot.bar(width=0.9, alpha=0.6, edgecolor='black', linewidth=2, stacked=True)
        # ax.grid(axis='y')
        # # ax.axvline(x=4.5)
        # plt.xticks(ha='right', rotation=30)

        chart_name = str(parameters_to_plot) + '_' + str(n_repetitions) + '-repetitions_' + tool + self.file_format
        # # plt.xlabel('x_y_z_v:  x = height and width of INPUT;  y = n. of channels of INPUT;  z = height and width of KERNEL; v = n. of KERNELS')
        # plt.xlabel('Dimensions')
        # # plt.ylabel(measurement_unit)
        # plt.title(chart_name.replace(self.file_format, ''))
        # # plt.title('CPI (Cycles per instruction) - Changing the number of kernels')
        # # plt.legend(loc='best', fontsize=20)
        # plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=20)
        # plt.tight_layout()
        # # plt.show()

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
    n_repetitions = 20

    # # Execution time
    # my_chart_creator.make_chart('TIME-MEDIAN', 'Time (ms)', n_repetitions, 'ExecutionTime', compute_best_order=True, min_is_best=True, log_scale=False, normalize=True)

    # # # # Perf
    # my_chart_creator.make_chart('BRANCH-MISSES', '% of branches', n_repetitions, 'Perf', compute_best_order=True, min_is_best=True, log_scale=False, normalize=False)
    # my_chart_creator.make_chart('CPI', 'CPI', n_repetitions, 'Perf', compute_best_order=True, min_is_best=True, log_scale=False, normalize=False)
    # my_chart_creator.make_chart('L1-MISSES-COUNT', 'N. of Misses', n_repetitions, 'Perf', compute_best_order=True, min_is_best=True, log_scale=False, normalize=True)
    # my_chart_creator.make_chart('LLC-MISSES-COUNT', 'N. of Misses', n_repetitions, 'Perf', compute_best_order=True, min_is_best=True, log_scale=True, normalize=True)

    # # Vtune
    # my_chart_creator.make_chart('CPI', 'CPI', n_repetitions, 'VTune', compute_best_order=True, min_is_best=True, log_scale=False, normalize=False)
    # my_chart_creator.make_chart('SP_GFLOPS', 'SP_GFLOPS', n_repetitions, 'VTune', compute_best_order=True, min_is_best=False, log_scale=False, normalize=True)
    # my_chart_creator.make_chart('L1-BOUND', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=True, min_is_best=False, log_scale=False, normalize=False)
    # my_chart_creator.make_chart('L2-BOUND', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=True, min_is_best=False, log_scale=False, normalize=False)
    # my_chart_creator.make_chart('L3-BOUND', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=True, min_is_best=False, log_scale=False, normalize=False)
    # my_chart_creator.make_chart('VECTOR-CAPACITY-USAGE', 'Vector Capacity Usage', n_repetitions, 'VTune', compute_best_order=False, min_is_best=False, log_scale=False, normalize=False)
    # my_chart_creator.make_chart('MEMORY-BOUND', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=False, min_is_best=False, log_scale=True, normalize=True)
    my_chart_creator.make_chart('MEMORY-LATENCY', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=False, min_is_best=False, log_scale=False, normalize=True)

    # # Double charts
    # my_chart_creator.make_chart_double(
    #     ['TIME-MEDIAN', 'MEMORY-BOUND'], 'Time (ms)', n_repetitions, ['ExecutionTime', 'VTune'], compute_best_order=False, min_is_best=True, log_scale=False, normalize=True
    # )

    # Stacked charts
    # my_chart_creator.make_chart_stacked(['L1-BOUND', 'L2-BOUND', 'L3-BOUND'], '% of Clockticks', n_repetitions, 'VTune')
