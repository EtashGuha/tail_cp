#!/bin/bash

# Define the absolute path to the 'cfgs' directory
cfgs_dir="cfgs"

# List of folder names
folders=("best")

# Loop through each folder
for folder in "${folders[@]}"; do
    # Get a list of all YAML files in the folder
    echo "$cfgs_dir/$folder"
    yaml_files=($(find "cfgs/$folder" -name "*.yaml"))

    # Loop through each YAML file
    for yaml_file in "${yaml_files[@]}"; do
        # Extract the dataset name from the file name (assuming a specific naming convention)
        dataset=$(basename "$yaml_file" .yaml)
        screen_name="${dataset}"
        command="python main.py -c $yaml_file"

        # Run the command in a detached screen session
        screen -dmS "$screen_name" bash -c "$command"

    done
done