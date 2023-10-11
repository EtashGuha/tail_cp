#!/bin/bash

# Define the absolute path to the 'cfgs' directory
cfgs_dir="cfgs"

# List of folder names
folders=("best")

# Specify the range of seeds you want to iterate over
seed_start=0
seed_end=5  # Change this to the desired end seed

# Loop through each folder
for folder in "${folders[@]}"; do
    # Get a list of all YAML files in the folder
    echo "$cfgs_dir/$folder"
    yaml_files=($(find "cfgs/$folder" -name "*.yaml"))

    # Loop through each YAML file
    for yaml_file in "${yaml_files[@]}"; do
        # Extract the dataset name from the file name (assuming a specific naming convention)
        dataset=$(basename "$yaml_file" .yaml)
        
        # Loop through the specified range of seeds
        screen_name="${dataset}"
        command="python log_final_results.py -c $yaml_file"
        echo "${command}"
        screen -dmS "$screen_name" bash -c "$command"
        sleep .5
    done
done