#!/bin/bash

# Download repository with aggregators scripts
rm -rf Profiling-tools-scripts
git clone https://github.com/mircomannino/Profiling-tools-scripts.git

ROOT_FOLDER=$(pwd)

# Thread overhead EMPTY
RESULT_FOLDER=${ROOT_FOLDER}/results_threadEmpty
if [ -d  "$RESULT_FOLDER" ]; then
    cd ${RESULT_FOLDER}
    CSV_NAME=$(basename `pwd`)
    # Copy the python script
    cp ${ROOT_FOLDER}/Profiling-tools-scripts/Aggregators/group_thread_overhead.py ./
    # Run the python script
    THREAD_TEST_OPTION="--empty"
    python3 ./group_thread_overhead.py ${THREAD_TEST_OPTION} -o ${CSV_NAME}
    # Clean the directory and back to ROOT_FOLDER
    rm ./group_thread_overhead.py
    cd ..
fi

# Remove folder with grouping scripts
# rm -rf Profiling-tools-scripts
