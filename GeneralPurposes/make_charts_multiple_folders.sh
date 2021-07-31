#!/bin/bash

# Get python script
git clone
cp Profiling-tools-scripts/MakeCharts/make_charts.py ./

# Launch the python script
N_REPETITIONS=5
python3 ./make_charts.py --n-repetitions ${N_REPETITIONS} --multiple-folders

# Clean up
rm ./make_charts.py
rm -rf ./Profiling-tools-scripts    
