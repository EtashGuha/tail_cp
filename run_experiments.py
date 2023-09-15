import os
import subprocess
from multiprocessing import Pool
import argparse
from tqdm import tqdm

def run_command(folder_path, yaml_file):
    cmd = f'python main.py -c {folder_path}/{yaml_file}'
    subprocess.call(cmd, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run commands in parallel for YAML files in a folder")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing YAML files")
    args = parser.parse_args()

    folder_path = args.folder_path
    yaml_files = [f for f in os.listdir(folder_path) if f.endswith('.yaml')]

    # Number of parallel processes to run (adjust this as needed)
    num_processes = 4

    
    for yaml_file in tqdm(yaml_files):
        run_command(folder_path, yaml_file)
