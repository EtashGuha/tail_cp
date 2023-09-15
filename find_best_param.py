import subprocess
from itertools import product
import yaml
import os
import multiprocessing
from tqdm import tqdm

num_processes = 20
# Define argument names and their possible values as lists
argument_values = {
    "cqr":[True],
    "dataset_name": ["bimodal", "bio", "blog", "community", "concrete", "diabetes", "log_normal", "meps_19", "meps_20", "meps_21", "parkinsons", "solar"],
    "model_path": [None]
}
# Create all possible argument combinations
argument_combinations = product(*(values for values in argument_values.values()))

# Path to the main.py script
main_script_path = "main.py"

folder = "cqr"
name_suffix = "cqr"
if not os.path.exists("cfgs/{}".format(folder)):
    os.mkdir("cfgs/{}".format(folder))
all_commands = []
for index, combo in enumerate(argument_combinations):
    # Create the command to run main.py with the current arguments
    combo = list(combo)
    model_name = "{}_{}".format(combo[-2], name_suffix)
    combo[-1] = model_name
    yaml_filename = "cfgs/{}/{}.yaml".format(folder, model_name)

    with open(yaml_filename, "w") as yaml_file:
        # Create a dictionary with argument-value pairs
        arg_combo_dict = {arg: value for arg, value in zip(argument_values.keys(), combo)}
        
        # Write the dictionary to the YAML file
        yaml.dump(arg_combo_dict, yaml_file, default_flow_style=False)
