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
        # Copy HTML file in roofline-reports
        cp ${ROOFLINE_ANALYSIS_FOLDER}/roofline-reports/*.html ${ALL_ROOFLINES_FOLDER}
        echo Copied ${ANALYSIS_FOLDER} HTML report in ${ALL_ROOFLINES_FOLDER}
        # Group csv data
        cd ${ROOFLINE_ANALYSIS_FOLDER}
        CSV_NAME=$(basename `pwd`)
        # Copy the python script to group data
        cp ${ROOT_FOLDER}/Profiling-tools-scripts/Aggregators/group_data_roofline.py ./roofline-reports/group_data_roofline.py
        # Exectue the python script to group data
        cd ./roofline-reports
        python3 group_data_roofline.py -o ${CSV_NAME}
        # Delete python script
        rm group_data_roofline.py
        cd ../..
    done

    cd ..
done

# Remove folder with grouping scripts
rm -rf Profiling-tools-scripts
