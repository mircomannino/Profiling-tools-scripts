#!/bin/bash

# Download repository with aggregators scripts
rm -rf Profiling-tools-scripts
git clone https://github.com/mircomannino/Profiling-tools-scripts.git

ROOT_FOLDER=$(pwd)
ALL_ROOFLINES_FOLDER=$(pwd)/roofline-models

for ANALYSIS_FOLDER in ./analysis_N*
do
    cd ${ANALYSIS_FOLDER}
    # Roofline
    mkdir -p ${ALL_ROOFLINES_FOLDER}
    for ROOFLINE_ANALYSIS_FOLDER in ./Roofline*
    do
        # echo ${ROOFLINE_ANALYSIS_FOLDER}
        cp ${ROOFLINE_ANALYSIS_FOLDER}/roofline-reports/*.html ${ALL_ROOFLINES_FOLDER}
        echo Copied ${ANALYSIS_FOLDER} HTML report in ${ALL_ROOFLINES_FOLDER}
    done

    cd ..
done

# Remove folder with grouping scripts
rm -rf Profiling-tools-scripts
