#!/bin/bash

# List of folder names
folders=("original" "mle" "crossentropy" "no_entropy" "mle_entropy" "heteroskedastic")

# Activate the Conda environment
conda activate realground

# Loop through each folder
for folder in "${folders[@]}"; do
    # Get a list of all YAML files in the folder
    yaml_files=($(find "cfgs/$folder" -name "*.yaml"))

    # Loop through each YAML file
    for yaml_file in "${yaml_files[@]}"; do
        # Extract the dataset name from the file name (assuming a specific naming convention)
        dataset=$(basename "$yaml_file" .yaml)
        screen_name="${yaml_file}"
        command="python main.py -c $yaml_file"

        # Run the command in a detached screen session
        screen -dmS "$screen_name" bash -c "$command"

        # Optional: Add a sleep to stagger the screen creation
        sleep 2
    done
done





