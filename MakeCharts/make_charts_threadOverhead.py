# Script used to plot the overhead of the thread
import os 
import pandas as pd

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
        results_df = pd.read_csv(input_path)

        print(results_df)


my_chart_creator = ChartsCreator('./charts')
my_chart_creator.make_chart_empty()
