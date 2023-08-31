import subprocess
from itertools import product
import yaml
import os
import multiprocessing
from tqdm import tqdm

num_processes = 20
# Define argument names and their possible values as lists
argument_values = {
    "lr": [1e-3,1e-4],
    "ffn_hidden_dim": [1024, 2048, 3060],
    "ffn_num_layers": [4,5,6],
    "constraint_weights": ["0, 1"],
    "max_epochs": [500, 2000, 5000],
    "lq_norm_val": [.5, .25],
    "range_size": [50, 100],
    "ffn_activation": ["relu"],
    "devices": [1], 
    "batch_size": [32],
    "loss_type": ["thomas_lq"],
    "dataset_name": ["concrete"],
    "model": ["mlp"],
    "early_stopping": [False],
    "model_path": [None]
}
# Create all possible argument combinations
argument_combinations = product(*(values for values in argument_values.values()))

# Path to the main.py script
main_script_path = "main.py"
os.mkdir("cfgs/search_concrete_aga2_ne")
all_commands = []
for index, combo in enumerate(argument_combinations):
    # Create the command to run main.py with the current arguments
    
    combo = list(combo)
    combo[-1] = f"search_concrete_aga2_ne_{index}"
    yaml_filename = f"cfgs/search_concrete_aga2_ne/search_concrete_aga2_ne_{index}.yaml"

    with open(yaml_filename, "w") as yaml_file:
        # Create a dictionary with argument-value pairs
        arg_combo_dict = {arg: value for arg, value in zip(argument_values.keys(), combo)}
        
        # Write the dictionary to the YAML file
        yaml.dump(arg_combo_dict, yaml_file, default_flow_style=False)
    command = ["python", main_script_path, "-c", yaml_filename]
    all_commands.append(command)

def run_yaml(command):
    try:
        subprocess.run(command, check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error running {main_script_path} with arguments {command}: {e}")

with multiprocessing.Pool(num_processes) as pool:
    for _ in tqdm(pool.imap_unordered(run_yaml, all_commands), total=len(all_commands), desc="Processing"):
        pass
