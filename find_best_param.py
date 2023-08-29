import subprocess
from itertools import product
import yaml
import os
# Define argument names and their possible values as lists
argument_values = {
    "lr": [1e-3, 1e-4, 1e-5, 1e-6],
    "ffn_hidden_dim": [256, 512, 1024, 2048, 3060],
    "ffn_num_layers": [2,3,4,5,6],
    "constraint_weights": ["1, 1", "1, 2", "1, 5", "1, 10", "1, 20"],
    "max_epochs": [500, 1000, 2000, 3000, 5000],
    "lq_norm_val": [1, .5, .25, .1],
    "range_size": [25, 50, 75, 100],
    "ffn_activation": ["relu"],
    "devices": [1], 
    "batch_size": [32],
    "loss_type": ["thomas_lq"],
    "dataset_name": ["concrete"],
    "model": ["mlp"],
    "model_path": [None]
}
# Create all possible argument combinations
argument_combinations = product(*(values for values in argument_values.values()))

# Path to the main.py script
main_script_path = "main.py"
os.mkdir("cfgs/search_concrete")
# Iterate over the argument combinations and run main.py
for index, combo in enumerate(argument_combinations):
    # Create the command to run main.py with the current arguments
    
    combo = list(combo)
    combo[-1] = f"search_concrete_{index}"
    yaml_filename = f"cfgs/search_concrete/search_concrete_{index}.yaml"

    with open(yaml_filename, "w") as yaml_file:
        # Create a dictionary with argument-value pairs
        arg_combo_dict = {arg: value for arg, value in zip(argument_values.keys(), combo)}
        
        # Write the dictionary to the YAML file
        yaml.dump(arg_combo_dict, yaml_file, default_flow_style=False)
    command = ["python", main_script_path, "-c", yaml_filename]
    # Run the command using subprocess
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {main_script_path} with arguments {command}: {e}")
