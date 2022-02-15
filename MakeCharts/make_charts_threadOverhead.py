# Script used to plot the overhead of the thread
import os 
import csv
import matplotlib.pyplot as plt
import pandas as pd
import copy

class ChartsCreator:
    def __init__(self, output_path) -> None:
        self.output_path = output_path
        self.results_dir = {
            'EMPTY': 'results_threadEmpty'
        }
    
    def make_chart_empty(self):
        csv_name = 'test_threadOverheadEmpty.csv'
        input_path = os.path.join(self.results_dir['EMPTY'], csv_name)

        # Use a DataFrame
        results = {}
        results['Average Time'] = {}
        results['Max Time'] = {}
        results['Min Time'] = {}
        standard_deviations = []
        with open(input_path, 'r') as input_file:
            csv_file = csv.reader(input_file)
            next(csv_file) # Skip first row
            for line in csv_file:
                n_thread = int(line[0].split('_')[2])
                mean = float(line[1])
                std_dev = float(line[2])
                median = float(line[3])
                minimum = float(line[4])
                maximum = float(line[5])
                # Store results for each times
                results['Average Time'][n_thread] = mean
                results['Max Time'][n_thread] = maximum
                results['Min Time'][n_thread] = minimum
                standard_deviations.append(std_dev)

        # Setup the plot
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8,10))

        results_normalized = copy.deepcopy(results)
        results_normalized_nthread = copy.deepcopy(results)

        # Standard plot
        results_df = pd.DataFrame.from_dict(results)
        results_df.plot(ax=ax[0], kind='line', marker='o', yerr=[standard_deviations, [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]], grid='y')
        ax[0].set(title='Total time overhead [1 to 8 threads]')

        # Normalized plot
        for time_type, time_val in results_normalized.items():
            thread_n1_val = time_val[1]
            for n_thread, value in time_val.items():
                results_normalized[time_type][n_thread] = results[time_type][n_thread] / thread_n1_val 
        results_normalized_df = pd.DataFrame.from_dict(results_normalized)
        results_normalized_df.plot(ax=ax[1], kind='line', marker='o', grid='y')
        ax[1].set(title='Normalized total time overhead [1 to 8 threads]')

        # Overhead for 1 thread 
        for time_type, time_val in results_normalized_nthread.items():
            for n_thread, value in time_val.items():
                results_normalized_nthread[time_type][n_thread] = results[time_type][n_thread] / n_thread   
        results_normalized_nthread_df = pd.DataFrame.from_dict(results_normalized_nthread)
        results_normalized_nthread_df.plot(ax=ax[2], kind='line', marker='o', grid='y')
        ax[2].set(title='Overhead per thread (time_1_thread = time_N_threads / N) [1 to 8 threads]')

        # Finalize the plot
        for i, axes in enumerate(ax):
            y_label = 'Overhead time [ms]' if i!=1 else 'Overhead time'
            axes.set(ylabel=y_label, xlabel='Number of threads')

        # Save the plot
        # plt.title('Thread overhead analysis')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'thread_overhead_empty.pdf'))



my_chart_creator = ChartsCreator('./charts')
my_chart_creator.make_chart_empty()