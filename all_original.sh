#!/bin/bash

# Define the absolute path to the 'cfgs' directory
cfgs_dir="cfgs"

# List of folder names
folders=("")

# Specify the range of seeds you want to iterate over
seed_start=1
seed_end=10  # Change this to the desired end seed

# Loop through each folder
for folder in "${folders[@]}"; do
    # Get a list of all YAML files in the folder
    echo "$cfgs_dir/$folder"
    yaml_files=($(find "/home/etashguha/tail_cp/cfgs/$folder" -name "*.yaml"))

    # Loop through each YAML file
    for yaml_file in "${yaml_files[@]}"; do
        # Extract the dataset name from the file name (assuming a specific naming convention)
        dataset=$(basename "$yaml_file" .yaml)
        
        # Loop through the specified range of seeds
        for seed in $(seq $seed_start $seed_end); do
            screen_name="${dataset}_seed${seed}"
            command="python main.py -c $yaml_file --seed $seed"
            
            # Run the command in a detached screen session
            screen -dmS "$screen_name" bash -c "$command"
        done
    done
done