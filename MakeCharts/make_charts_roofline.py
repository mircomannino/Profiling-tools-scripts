# Python script used to create bar charts from previous analysis   
import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

##### Class for BlackYeti roofline ######
class BlackYeti:
    def __init__(self):
        # Initialize with the known bandwidth peaks
        L1 = { # Max bandwidth [GB/s]
            '1-cores': 425.28,
            '2-cores': 851.16,
            '4-cores': 1702.32,
            '6-cores': 2553.47,
            '8-cores': 3404.63
        }
        L2 = { # Max bandwidth [GB/s]
            '1-cores': 150.44,
            '2-cores': 300.87,
            '4-cores': 601.74,
            '6-cores': 902.47,
            '8-cores': 1203.63
        }
        L3 = { # Max bandwidth [GB/s]
            '1-cores': 67.85,
            '2-cores': 135.7,
            '4-cores': 271.39,
            '6-cores': 407.09,
            '8-cores': 542.79
        }
        DRAM = { # Max bandwidth [GB/s]
            '1-cores': 23.52,
            '2-cores': 27.35,
            '4-cores': 30.86,
            '6-cores': 31.23,
            '8-cores': 31.36
        }

        # Initialize max peaks at 0.16-0.17 FLOPs/Byte (OI)
        self.max_OI = {}
        self.max_OI['L1'] = {}
        self.max_OI['L2'] = {}
        self.max_OI['L3'] = {}
        self.max_OI['DRAM'] = {}
        self.OI = 0.167
        for n_cores in L1.keys():
            self.max_OI['L1'][n_cores] = L1[n_cores] * self.OI
            self.max_OI['L2'][n_cores] = L2[n_cores] * self.OI
            self.max_OI['L3'][n_cores] = L3[n_cores] * self.OI
            self.max_OI['DRAM'][n_cores] = DRAM[n_cores] * self.OI
        

    def L1_interpolation(self, n_cores):
        x_val = np.array([1, 2, 4, 6, 8]) # Number of cores
        y_val = np.array([v for k,v in self.max_OI['L1'].items()])
        return np.interp(n_cores, x_val, y_val)
    
    def L2_interpolation(self, n_cores):
        x_val = np.array([1, 2, 4, 6, 8]) # Number of cores
        y_val = np.array([v for k,v in self.max_OI['L2'].items()])
        return np.interp(n_cores, x_val, y_val)
    
    def L3_interpolation(self, n_cores):
        x_val = np.array([1, 2, 4, 6, 8]) # Number of cores
        y_val = np.array([v for k,v in self.max_OI['L3'].items()])
        return np.interp(n_cores, x_val, y_val)

    def DRAM_interpolation(self, n_cores):
        x_val = np.array([1, 2, 4, 6, 8]) # Number of cores
        y_val = np.array([v for k,v in self.max_OI['DRAM'].items()])
        return np.interp(n_cores, x_val, y_val)

##### FONT SIZES #####
FONTSIZE = {
    'REGULAR': 24,
    'TITLE': 30,
    'SUBTITLE': 27,
    'LEGEND': 23,
    'ANNOTATIONS': 18
}

##### CORE USAGE #####
CORES_USAGE =  {
    'PHYCORE1_THREAD1': {
        1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 8, 10: 8, 11: 8, 12: 8, 13: 8, 14: 8, 15: 8, 16: 8
    }
}

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
            'VTune': 'reports',
            'Roofline': 'roofline-reports'
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
        SELECTED_ANALYSIS=['N2', 'N3', 'N4']
        for analysis_directory in analysis_directories:
            # Get Number of the analysis
            n_analysis = analysis_directory.replace('analysis_', '')    # 'analysis_N1' ---> 'N1'

            if n_analysis not in SELECTED_ANALYSIS:
                continue

            if n_analysis not in results.keys():
                results[n_analysis] = {}

            # Get data folder path
            data_folder = tool + '_'
            data_folder += analysis_directory + '_'
            data_folder += str(n_repetitions) + '-repetitions'
            data_folder = os.path.join(analysis_directory, data_folder, self.data_folder_by_tool[tool])

            # Read the csv file in a DataFrame
            benchmarks_data_path = [file_ for file_ in os.listdir(data_folder) if file_.endswith('.csv') and file_.find('Roofline')!=-1][0]
            benchmarks_data = pd.read_csv(os.path.join(data_folder, benchmarks_data_path))

            for i, row in benchmarks_data.iterrows():
                benchmarks_info = row[0].split('_')
                n_threads = benchmarks_info[2]
                results[n_analysis][n_threads] = row[parameter_to_plot]
    

        # results_df = {n_analysis_name: pd.DataFrame(n_analysis_dict) for n_analysis_name, n_analysis_dict in results.items()}
        results_df = pd.DataFrame.from_dict(results)

        # Order by analysis
        results_df = {n_analysis_name: results_df[n_analysis_name] for n_analysis_name in sorted(results_df.keys())}

        # Prepare the plot
        N_ROWS = len(results.keys()) # One row for each analysis: N1, N2, ...
        N_COLS = 1
        fig, ax = plt.subplots(nrows=N_ROWS, ncols=N_COLS, figsize=(25,23))

        # Plot
        for i, (n_analysis_name,result_df) in enumerate(results_df.items()):
            result_df.T.plot(ax=ax[i], kind='bar', rot=0, legend=False, width=0.6, colormap='Pastel1', edgecolor='black', zorder=3, label=alloc_type+' Performance')

            # Setup subplots
            ax[i].grid('y')
            ax[i].set_yscale('log')
            for axis in [ax[i].yaxis]:
                axis.set_major_formatter(ScalarFormatter())
            ax[i].set_title('Order: '+n_analysis_name, fontsize=FONTSIZE['REGULAR'])
            ax[i].set_ylabel(measurements_unit, fontsize=FONTSIZE['REGULAR'])
            ax[i].set_xlabel('Number of threads', fontsize=FONTSIZE['REGULAR'])
            ax[i].tick_params(labelsize=FONTSIZE['REGULAR'])

            # Plot the rooflines
            blackyeti_perf = BlackYeti()
            n_threads = [ n_threads.keys() for n_analysis_name, n_threads in results.items() ][0]
            # Max Performance L1
            max_performance_L1 = {int(k):blackyeti_perf.L1_interpolation((CORES_USAGE[alloc_type][int(k)])) for k in n_threads}
            x_val = [(k-1) for k,v in max_performance_L1.items()]
            y_val = [v for k,v in max_performance_L1.items()]
            ax[i].plot(x_val, y_val, '+', c='blue', markeredgewidth=2, markersize=30, label='L1 peak', zorder=3)
            # # Max Performance L2
            max_performance_L2 = {int(k):blackyeti_perf.L2_interpolation((CORES_USAGE[alloc_type][int(k)])) for k in n_threads}
            x_val = [(k-1) for k,v in max_performance_L2.items()]
            y_val = [v for k,v in max_performance_L2.items()]
            ax[i].plot(x_val, y_val, '+', c='green', markeredgewidth=2, markersize=30, label='L2 peak', zorder=3)
            # # Max Performance L3
            max_performance_L3 = {int(k):blackyeti_perf.L3_interpolation((CORES_USAGE[alloc_type][int(k)])) for k in n_threads}
            x_val = [(k-1) for k,v in max_performance_L3.items()]
            y_val = [v for k,v in max_performance_L3.items()]
            ax[i].plot(x_val, y_val, '+', c='purple', markeredgewidth=2, markersize=30, label='L3 peak', zorder=3)
            # # Max Performance DRAM
            max_performance_DRAM = {int(k):blackyeti_perf.DRAM_interpolation((CORES_USAGE[alloc_type][int(k)])) for k in n_threads}
            x_val = [(k-1) for k,v in max_performance_DRAM.items()]
            y_val = [v for k,v in max_performance_DRAM.items()]
            ax[i].plot(x_val, y_val, '+', c='red', markeredgewidth=2, markersize=30, label='DRAM peak', zorder=3)

            # Annotate values 
            for p in ax[i].patches:
                value = int(np.round(p.get_height()))
                ax[i].annotate(str(value), (p.get_x(), p.get_height()/2), fontsize=FONTSIZE['ANNOTATIONS'])

    
        plt.tight_layout()

        # Adjust subplots
        plt.rcParams['legend.title_fontsize'] = FONTSIZE['LEGEND']
        if len(SELECTED_ANALYSIS) == 5:
            ax[0].legend(fontsize=FONTSIZE['LEGEND'], ncol=16//2, loc='upper left', bbox_to_anchor=(0, 1.68))
            fig.subplots_adjust(top=0.85, hspace = .6)
        if len(SELECTED_ANALYSIS) == 3:
            ax[0].legend(fontsize=FONTSIZE['LEGEND'], ncol=16//2, loc='upper left', bbox_to_anchor=(0, 1.32))
            fig.subplots_adjust(top=0.86, hspace = .3)

        # Finalization 
        chart_name = parameter_to_plot + '_' + str(n_repetitions) + '-repetitions_' + tool + '_' + alloc_type + self.file_format
        plt.text(x=0.5, y=0.98, s=title, ha="center", transform=fig.transFigure, fontsize=FONTSIZE['TITLE'])
        plt.text(x=0.5, y=0.955, s='Affinity type: '+alloc_type, color='darkgrey', ha="center", transform=fig.transFigure, fontsize=FONTSIZE['SUBTITLE'])
        
        plt.savefig(os.path.join(self.output_path, chart_name))
        print(chart_name, ': Done')


if __name__ == "__main__":
    # Create a parser for arguments
    parser = argparse.ArgumentParser(description='Make charts.')
    parser.add_argument('--output-folder', '-o', type=str, default='./charts', help='destination folder. default: ./charts')
    parser.add_argument('--output-type', '-t', type=str, default='pdf', help='format of output charts [png, pdf]. default: èdf')
    parser.add_argument('--n-repetitions', '-n', type=int, default=10, help='Number of repetitions used in the benchmarks. default=5')
    args = parser.parse_args()

    my_chart_creator = ChartsCreator(args.output_folder, file_format=args.output_type)
    n_repetitions = args.n_repetitions

    my_chart_creator.make_chart('GFLOPS', 'Performance [GLOPs/S]', n_repetitions, 'Roofline', title="Performance of AlexNet at 0.167 FLOPs/Byte", alloc_type='PHYCORE1_THREAD1')
