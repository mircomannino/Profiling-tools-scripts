# Python script used to create bar charts from previous analysis
import pandas as pd
import numpy as np
import os
import argparse
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

    def make_chart(self, parameter_to_plot, measurement_unit, n_repetitions, tool, compute_best_order, min_is_best, log_scale, normalize, sub_plot=None, sub_title=None):
        '''
        Args:
            parameter_to_plot:      Name of the parameter to use in the charts
            n_repetitions:          Number of repetitions used in the analysisfile_format
            tool:                   Name of the tool from which the results come from [ExecutionTime, Perf, VTune]
            compute_best_order:     Bool to say if you want the loop order that obtain best results
            min_is_best:            Bool needed only if compute_best_order id True
            log_scale:              Bool to say if the plot will be in logaritmic scale
            normalize:              Bool to say if the results will be normalized respect to the first analysis
            sub_plot:               Axes used to plot subplots, default: None
            sub_title:              String with subtitle, default: None
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
                dimensions = dimensions.replace('benchmark_Compilers_', '')
                value = row[parameter_to_plot]
                # Store values
                if n_analysis not in results.keys():
                    results[n_analysis] = {}
                results[n_analysis][dimensions] = value

        # Normalize
        if(normalize):
            N1_values = {dim: (val if val>0 else 1.) for (dim, val) in results['N1'].items()}
            for n_analysis in results.keys():
                for dimensions in results[n_analysis].keys():
                    results[n_analysis][dimensions] /= N1_values[dimensions]

        # Order the name of dimensions (Order: Image size, Image depth, Kernel size, N Kernels)
        results_ordered = {}
        for n_analysis in sorted(results.keys()):
            results_ordered[n_analysis] = {}
            for dimensions in sorted(list(results[n_analysis].keys()) ,key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1]))):
                results_ordered[n_analysis][dimensions] = results[n_analysis][dimensions]
        results = results_ordered
        del results_ordered

        # Plot parameters
        if sub_plot == None:
            plt.rcParams["figure.figsize"] = [20,9]
            font = {'family' : 'DejaVu Sans',
            # 'weight' : 'bold',
            'size'   : 30}
            plt.rc('font', **font)

        # Title of the chart
        chart_name = parameter_to_plot + '_' + str(n_repetitions) + '-repetitions_' + tool + ('_normalized_' if normalize else '') + self.file_format


        # Plot results
        result_df = pd.DataFrame(results)
        ax = sub_plot or plt.gca()
        result_df.plot.bar(width=0.9, alpha=0.6, edgecolor='black', linewidth=2, ax=ax)
        ax.grid(axis='y')

        if (normalize):
            if(log_scale):
                ax.set_yscale('log')
                Y_LIM = 10**2
                ax.set_yticks(np.arange(10**(0), Y_LIM, 10))
            else:
                Y_LIM = 1.5
                ax.set_yticks(np.arange(0, Y_LIM, 0.5))
            ax.set_ylim(top=Y_LIM)
            for p in ax.patches:
                value = np.round(p.get_height(), decimals=2)
                if value <= Y_LIM:
                    value = ''
                ax.annotate(str(value), (p.get_x() * 1.0005, Y_LIM * 0.7), fontsize=9.5)


        # Only for subplots operations
        if sub_plot != None:
            ax.set_xlabel('Dimensions')
            ax.set_ylabel(measurement_unit)
            ax.set_title(chart_name.replace(self.file_format, '') + ' ' + sub_title)
            plt.subplots_adjust(hspace=1.2)
            ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=14)
            for tick in ax.get_xticklabels():
                tick.set_rotation(20)
            return ax, chart_name


        plt.xticks(ha='right', rotation=30)

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
        plt.clf() # Clear the figure
        print(chart_name, ': Done')


    def make_chart_double(self, parameters_to_plot, measurement_unit, n_repetitions, tools, compute_best_order, min_is_best, log_scale, normalize, sub_plot=None, sub_title=None):
        '''
        Args:
            parameter_to_plot:      List with the names of the parameters to use in the charts
            n_repetitions:          Number of repetitions used in the analysisfile_format
            tools:                  List wirh the names of the tools from which the results come from [ExecutionTime, Perf, VTune]
            compute_best_order:     Bool to say if you want the loop order that obtain best results
            min_is_best:            Bool needed only if compute_best_order id True
            log_scale:              Bool to say if the plot will be in logaritmic scale
            normalize:              Bool to say if the results will be normalized respect to the first analysis
            sub_plot:               Axes used to plot subplots, default: None
            sub_title:              String with subtitle, default: None
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
                dimensions = dimensions.replace('benchmark_Compilers_', '')
                value = row[parameters_to_plot[0]]
                # Store values
                if n_analysis not in results_parameter_1.keys():
                    results_parameter_1[n_analysis] = {}
                results_parameter_1[n_analysis][dimensions] = value

            # Get only the column of interest - Parameter 2
            for index, row in benchmarks_data_parameter_2.iterrows():
                # Get info from dataframe
                dimensions = row[0].replace('.txt', '')[:-4]     # benchmark_Naive_x_x_x_x_x_x.txt ---> benchmark_Naive_x_x_x_x
                dimensions = dimensions.replace('benchmark_Compilers_', '')
                value = row[parameters_to_plot[1]]
                # Store values
                if n_analysis not in results_parameter_2.keys():
                    results_parameter_2[n_analysis] = {}
                results_parameter_2[n_analysis][dimensions] = value


        if(normalize):
            # Normalize - Parameter 1
            N1_values = {dim: (val if val>0 else 1.) for (dim, val) in results_parameter_1['N1'].items()}
            for n_analysis in results_parameter_1.keys():
                for dimensions in results_parameter_1[n_analysis].keys():
                    results_parameter_1[n_analysis][dimensions] /= N1_values[dimensions]
            # Normalize - Parameter 2
            N1_values = {dim: (val if val>0 else 1.) for (dim, val) in results_parameter_2['N1'].items()}
            for n_analysis in results_parameter_2.keys():
                for dimensions in results_parameter_2[n_analysis].keys():
                    results_parameter_2[n_analysis][dimensions] /= N1_values[dimensions]

        # Order the name of dimensions (Order: Image size, Image depth, Kernel size, N Kernels)
        results_ordered_parameter_1 = {}
        results_ordered_parameter_2 = {}
        for n_analysis in sorted(results_parameter_1.keys()):
            results_ordered_parameter_1[n_analysis] = {}
            results_ordered_parameter_2[n_analysis] = {}
            for dimensions in sorted(list(results_parameter_1[n_analysis].keys()) ,key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1]))):
                results_ordered_parameter_1[n_analysis][dimensions] = results_parameter_1[n_analysis][dimensions]
                results_ordered_parameter_2[n_analysis][dimensions] = results_parameter_2[n_analysis][dimensions]
        results_parameter_1 = results_ordered_parameter_1
        results_parameter_2 = results_ordered_parameter_2
        del results_ordered_parameter_1
        del results_ordered_parameter_2

        # Plot parameters
        if sub_plot == None:
            plt.rcParams["figure.figsize"] = [20,9]
            plt.rcParams["figure.autolayout"] = True
            font = {'family' : 'DejaVu Sans',
            # 'weight' : 'bold',
            'size'   : 25}
            plt.rc('font', **font)

        # Make the Pandas DataFrame
        result_df_parameter_1 = pd.DataFrame(results_parameter_1)
        result_df_parameter_2 = pd.DataFrame(results_parameter_2)

        # Title of the chart
        chart_name = parameters_to_plot[0] + '_' + parameters_to_plot[1] + '_' + str(n_repetitions) + '-repetitions_' + str(tools) + ('_normalized_' if normalize else '') + self.file_format

        ax = sub_plot or plt.gca()
        result_df_parameter_2.plot.bar(width=0.9, alpha=0.6, edgecolor='black', linewidth=2, ax=ax)
        result_df_parameter_1.plot.bar(width=0.9, ax=ax,  color='grey', alpha=0.2, align='center', edgecolor='black', linewidth=2)
        ax.grid(axis='y')
        plt.xticks(ha='right', rotation=30)

        if (normalize):
            if(log_scale):
                ax.set_yscale('log')
                Y_LIM = 10**2
                ax.set_yticks(np.arange(10**(0), Y_LIM, 10))
            else:
                Y_LIM = 1.5
                ax.set_yticks(np.arange(0, Y_LIM, 0.5))
            ax.set_ylim(top=Y_LIM)
            for p in ax.patches:
                value = np.round(p.get_height(), decimals=2)
                if value <= Y_LIM:
                    value = ''
                ax.annotate(str(value), (p.get_x() * 1.0005, Y_LIM * 0.7), fontsize=9.5)


        if(log_scale):
            ax.set_yscale('log')

        # Only for subplots operations
        if sub_plot != None:
            ax.set_xlabel('Dimensions')
            ax.set_ylabel(measurement_unit)
            ax.set_title(chart_name.replace(self.file_format, '') + ' ' + sub_title)
            plt.subplots_adjust(hspace=1.2)
            ax.legend(
                [str(n_analysis) for n_analysis in list(results_parameter_2.keys())] +  ['Exec Time'],
                bbox_to_anchor=(1, 1),
                loc='upper left',
                fontsize=14
            )
            for tick in ax.get_xticklabels():
                tick.set_rotation(20)
            return ax, chart_name

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
        plt.clf() # Clear the figure
        print(chart_name, ': Done')


    def make_chart_stacked(self, parameters_to_plot, measurements_unit, n_repetitions, tool, title=None):
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
                dimensions = dimensions.replace('benchmark_Compilers_', '')
                if dimensions not in results.keys():
                    results[dimensions] = {}
                for parameter in parameters_to_plot:
                    value = row[parameter]
                    # Store values
                    if parameter not in results[dimensions].keys():
                        results[dimensions][parameter] = {}
                    results[dimensions][parameter][n_analysis] = value

        # Order the name of dimensions (Order: Image size, Image depth, Kernel size, N Kernels)
        results_ordered = {}
        for dimensions in results.keys():
            results_ordered[dimensions] = {}
            for parameter in parameters_to_plot:
                results_ordered[dimensions][parameter] = [v for (k,v) in sorted(list(results[dimensions][parameter].items()), key=lambda x: x[0])]
        results = results_ordered
        del results_ordered

        # Plot results
        result_dfs = {dimensions_name: pd.DataFrame(dimensions_dict, index=['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7'])
            for dimensions_name, dimensions_dict in results.items()}

        N_ROWS = 3
        N_COLS = 2
        fig, ax = plt.subplots(nrows=N_ROWS, ncols=N_COLS)

        positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

        for i, (dimensions_name, result_df) in enumerate(result_dfs.items()):
            result_df.plot(kind='bar', stacked=True, ax=ax[positions[i]], alpha=0.7)

            ax[positions[i]].set_title(label=('dimensions: ' + dimensions_name.replace('_',' ')))
            ax[positions[i]].legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=8)
            ax[positions[i]].set(xlabel='N. of order of loops', ylabel='% Clocktick')

        chart_name = str(parameters_to_plot) + '_' + str(n_repetitions) + '-repetitions_' + tool + self.file_format

        # fig.delaxes(ax[positions[-1]])

        plt.tight_layout()

        # Save the plot in a file with the appropriate name
        plt.savefig(os.path.join(self.output_path, chart_name))
        plt.clf() # Clear the figure
        print(chart_name, ': Done')

    def make_charts_of_different_folders(self, parameter_to_plot, measurement_unit, n_repetitions, tool, log_scale, normalize):
        '''
        Args:
            parameter_to_plot:      Name of the parameter to use in the charts
            n_repetitions:          Number of repetitions used in the analysisfile_format
            tool:                   Name of the tool from which the results come from [ExecutionTime, Perf, VTune]
            log_scale:              Bool to say if the plot will be in logaritmic scale
            normalize:              Bool to say if the results will be normalized respect to the first analysis
        '''
        # Get all directories path
        directories = {path.split('_')[-1]: path for path in sorted(os.listdir(), reverse=True) if (os.path.isdir(path) and path != self.output_path.replace('./',''))}

        # Setup subplot
        font = {'family' : 'DejaVu Sans',
            # 'weight' : 'bold',
            'size'   : 20}
        plt.rc('font', **font)
        fig, ax = plt.subplots(nrows=len(directories), figsize=(20, 15))

        # Make all the charts in each folder
        for i, (name, directory) in enumerate(directories.items()):
            os.chdir(directory)
            _, title = self.make_chart(parameter_to_plot, measurement_unit, n_repetitions, tool, '', '', log_scale, normalize, sub_plot=ax[i], sub_title=name)
            os.chdir('..')

        # Save the plot in a file with the appropriate name
        plt.savefig(os.path.join(self.output_path, title))
        plt.clf()
        print(str(parameter_to_plot), ': Done')

    def make_chart_double_from_different_folders(self, parameters_to_plot, measurement_unit, n_repetitions, tools, log_scale, normalize):
        '''
        Args:
            parameters_to_plot:      List with the names of the parameters to use in the charts
            n_repetitions:          Number of repetitions used in the analysisfile_format
            tools:                  List wirh the names of the tools from which the results come from [ExecutionTime, Perf, VTune]
            log_scale:              Bool to say if the plot will be in logaritmic scale
            normalize:              Bool to say if the results will be normalized respect to the first analysis
        '''

        # Get all directories path
        directories = {path.split('_')[-1]: path for path in sorted(os.listdir(), reverse=True) if (os.path.isdir(path) and path != self.output_path.replace('./',''))}

        # Setup subplot
        font = {'family' : 'DejaVu Sans',
            # 'weight' : 'bold',
            'size'   : 20}
        plt.rc('font', **font)
        fig, ax = plt.subplots(nrows=len(directories), figsize=(20, 15))

        # Make all the charts in each folder
        for i, (name, directory) in enumerate(directories.items()):
            os.chdir(directory)
            _, title = self.make_chart_double(parameters_to_plot, measurement_unit, n_repetitions, tools, '', '', log_scale, normalize, sub_plot=ax[i], sub_title=name)
            os.chdir('..')

        # Save the plot in a file with the appropriate name
        plt.savefig(os.path.join(self.output_path, title))
        print(str(parameters_to_plot), ': Done')





if __name__ == "__main__":
    # Create a parser for arguments
    parser = argparse.ArgumentParser(description='Make charts.')
    parser.add_argument('--output-folder', '-o', type=str, default='./charts', help='destination folder. default: ./charts')
    parser.add_argument('--output-type', '-t', type=str, default='png', help='format of output charts [png, pdf]. default: png')
    parser.add_argument('--n-repetitions', '-n', type=int, default=5, help='Number of repetitions used in the benchmarks. default=5')
    parser.add_argument('--single-folder', '-s', action='store_true', help='Use it when you plot a single folder')
    parser.add_argument('--multiple-folders', '-m', action='store_true', help='Use it when you polot from different folders')
    args = parser.parse_args()

    my_chart_creator = ChartsCreator(args.output_folder, file_format=args.output_type)
    n_repetitions = args.n_repetitions

    if args.single_folder:
        # Execution time
        my_chart_creator.make_chart('TIME-MEDIAN', 'Time (ms)', n_repetitions, 'ExecutionTime', compute_best_order=True, min_is_best=True, log_scale=False, normalize=True)

        # Perf
        my_chart_creator.make_chart('BRANCH-MISSES', '% of branches', n_repetitions, 'Perf', compute_best_order=True, min_is_best=True, log_scale=False, normalize=False)
        my_chart_creator.make_chart('BRANCH-MISSES', '% of branches', n_repetitions, 'Perf', compute_best_order=True, min_is_best=True, log_scale=False, normalize=True)
        my_chart_creator.make_chart('CPI', 'CPI', n_repetitions, 'Perf', compute_best_order=True, min_is_best=True, log_scale=False, normalize=False)
        my_chart_creator.make_chart('L1-MISSES-COUNT', 'N. of Misses', n_repetitions, 'Perf', compute_best_order=True, min_is_best=True, log_scale=False, normalize=True)
        my_chart_creator.make_chart('LLC-MISSES-COUNT', 'N. of Misses', n_repetitions, 'Perf', compute_best_order=True, min_is_best=True, log_scale=True, normalize=True)

        # Vtune
        my_chart_creator.make_chart('CPI', 'CPI', n_repetitions, 'VTune', compute_best_order=True, min_is_best=True, log_scale=False, normalize=False)
        my_chart_creator.make_chart('CPI', 'CPI', n_repetitions, 'VTune', compute_best_order=True, min_is_best=True, log_scale=False, normalize=True)
        my_chart_creator.make_chart('SP_GFLOPS', 'SP_GFLOPS', n_repetitions, 'VTune', compute_best_order=True, min_is_best=False, log_scale=False, normalize=True)
        my_chart_creator.make_chart('FRONT-END-BOUND', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=True, min_is_best=False, log_scale=True, normalize=False)
        my_chart_creator.make_chart('BACK-END-BOUND', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=True, min_is_best=False, log_scale=False, normalize=True)
        my_chart_creator.make_chart('L1-BOUND', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=True, min_is_best=False, log_scale=False, normalize=False)
        my_chart_creator.make_chart('L1-BOUND', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=True, min_is_best=False, log_scale=False, normalize=True)
        my_chart_creator.make_chart('L2-BOUND', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=True, min_is_best=False, log_scale=False, normalize=False)
        my_chart_creator.make_chart('L2-BOUND', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=True, min_is_best=False, log_scale=False, normalize=True)
        my_chart_creator.make_chart('L3-BOUND', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=True, min_is_best=False, log_scale=False, normalize=False)
        my_chart_creator.make_chart('L3-BOUND', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=True, min_is_best=False, log_scale=False, normalize=True)
        my_chart_creator.make_chart('VECTOR-CAPACITY-USAGE', 'Vector Capacity Usage', n_repetitions, 'VTune', compute_best_order=False, min_is_best=False, log_scale=False, normalize=False)
        my_chart_creator.make_chart('MEMORY-BOUND', '% of PipelineSlots', n_repetitions, 'VTune', compute_best_order=False, min_is_best=False, log_scale=False, normalize=True)
        my_chart_creator.make_chart('LLC-MISSES-COUNT', 'N. of Misses', n_repetitions, 'VTune', compute_best_order=True, min_is_best=True, log_scale=True, normalize=True)
        my_chart_creator.make_chart('CORE-BOUND', '% of PipelineSlots', n_repetitions, 'VTune', compute_best_order=False, min_is_best=False, log_scale=False, normalize=True)
        my_chart_creator.make_chart('MEMORY-LATENCY', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=False, min_is_best=False, log_scale=False, normalize=True)
        my_chart_creator.make_chart('LOAD-OPERATION-UTILIZATION', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=False, min_is_best=False, log_scale=False, normalize=True)
        my_chart_creator.make_chart('STORE-OPERATION-UTILIZATION', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=False, min_is_best=False, log_scale=False, normalize=True)
        my_chart_creator.make_chart('CYCLES-0-PORTS-UTILIZED', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=False, min_is_best=False, log_scale=False, normalize=True)
        my_chart_creator.make_chart('CYCLES-1-PORT-UTILIZED', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=False, min_is_best=False, log_scale=False, normalize=True)
        my_chart_creator.make_chart('CYCLES-2-PORTS-UTILIZED', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=False, min_is_best=False, log_scale=False, normalize=True)
        my_chart_creator.make_chart('CYCLES-3+-PORTS-UTILIZED', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=False, min_is_best=False, log_scale=False, normalize=True)
        my_chart_creator.make_chart('BAD-SPECULATION', '% of PipelineSlots', n_repetitions, 'VTune', compute_best_order=False, min_is_best=False, log_scale=False, normalize=True)
        my_chart_creator.make_chart('MACHINE-CLEARS', '% of PipelineSlots', n_repetitions, 'VTune', compute_best_order=False, min_is_best=False, log_scale=False, normalize=True)
        my_chart_creator.make_charts('RETIRING', '% of Clockticks', n_repetitions, 'VTune', compute_best_order=False, min_is_best=False, log_scale=False, normalize=True)

        # Double charts
        my_chart_creator.make_chart_double(
            ['TIME-MEDIAN', 'MEMORY-BOUND'], 'Time (ms)', n_repetitions, ['ExecutionTime', 'VTune'], compute_best_order=False, min_is_best=True, log_scale=False, normalize=True
        )

        # Stacked charts
        my_chart_creator.make_chart_stacked(['L1-BOUND', 'L2-BOUND', 'L3-BOUND'], '% of Clockticks', n_repetitions, 'VTune')

    if args.multiple_folders:
        # Execution time
        my_chart_creator.make_charts_of_different_folders('TIME-MEDIAN', 'Time (ms)', n_repetitions, 'ExecutionTime', log_scale=False, normalize=True)
        my_chart_creator.make_charts_of_different_folders('TIME-MEDIAN', 'Time (ms)', n_repetitions, 'ExecutionTime', log_scale=False, normalize=False)

        # Perf
        my_chart_creator.make_charts_of_different_folders('BRANCH-MISSES', '% of branches', n_repetitions, 'Perf', log_scale=False, normalize=False)
        my_chart_creator.make_charts_of_different_folders('CPI', 'CPI', n_repetitions, 'Perf', log_scale=False, normalize=False)
        my_chart_creator.make_charts_of_different_folders('L1-MISSES-COUNT', 'N. of Misses', n_repetitions, 'Perf', log_scale=False, normalize=True)
        my_chart_creator.make_charts_of_different_folders('LLC-MISSES-COUNT', 'N. of Misses', n_repetitions, 'Perf', log_scale=True, normalize=True)

        # Vtune
        my_chart_creator.make_charts_of_different_folders('CPI', 'CPI', n_repetitions, 'VTune', log_scale=False, normalize=False)
        my_chart_creator.make_charts_of_different_folders('CPI', 'CPI', n_repetitions, 'VTune', log_scale=False, normalize=True)
        my_chart_creator.make_charts_of_different_folders('SP_GFLOPS', 'SP_GFLOPS', n_repetitions, 'VTune', log_scale=False, normalize=True)
        my_chart_creator.make_charts_of_different_folders('L1-BOUND', '% of Clockticks', n_repetitions, 'VTune', log_scale=False, normalize=True)
        my_chart_creator.make_charts_of_different_folders('L2-BOUND', '% of Clockticks', n_repetitions, 'VTune', log_scale=False, normalize=True)
        my_chart_creator.make_charts_of_different_folders('L3-BOUND', '% of Clockticks', n_repetitions, 'VTune', log_scale=False, normalize=True)
        my_chart_creator.make_charts_of_different_folders('VECTOR-CAPACITY-USAGE', 'Vector Capacity Usage', n_repetitions, 'VTune', log_scale=False, normalize=False)
        my_chart_creator.make_charts_of_different_folders('MEMORY-BOUND', '% of PipelineSlots', n_repetitions, 'VTune', log_scale=False, normalize=True)
        my_chart_creator.make_charts_of_different_folders('CORE-BOUND', '% of PipelineSlots', n_repetitions, 'VTune', log_scale=False, normalize=True)
        my_chart_creator.make_charts_of_different_folders('MEMORY-LATENCY', '% of Clockticks', n_repetitions, 'VTune', log_scale=False, normalize=True)
        my_chart_creator.make_charts_of_different_folders('FRONT-END-BOUND', '% of Clockticks', n_repetitions, 'VTune', log_scale=True, normalize=False)
        my_chart_creator.make_charts_of_different_folders('BACK-END-BOUND', '% of Clockticks', n_repetitions, 'VTune', log_scale=False, normalize=True)
        my_chart_creator.make_charts_of_different_folders('RETIRING', '% of Clockticks', n_repetitions, 'VTune', log_scale=False, normalize=False)
        # my_chart_creator.make_charts_of_different_folders('FB-FILL', '% of Clockticks', n_repetitions, 'VTune', log_scale=True, normalize=True)

        # Double charts
        my_chart_creator.make_chart_double_from_different_folders(['TIME-MEDIAN', 'MEMORY-BOUND'], 'Time (ms)', n_repetitions, ['ExecutionTime', 'VTune'], log_scale=False, normalize=True)
