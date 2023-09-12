import subprocess
from itertools import product
import yaml
import os
from create_argparser import get_parser_args
import multiprocessing
from tqdm import tqdm
import torch
import copy
from main import main
import matplotlib.pyplot as plt


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = get_parser_args()
    num_processes = 10

    param_name = args.ablation_parameter
    param_values = args.ablation_values
    if not os.path.exists("model_paths/{}".format(args.ablation_name)):
        os.mkdir("model_paths/{}".format(args.ablation_name))
    if not os.path.exists("images/{}".format(args.ablation_name)):
        os.mkdir("images/{}".format(args.ablation_name))

    all_args = []
    for param_val in param_values:
        new_args = copy.deepcopy(args)
        setattr(new_args, param_name, param_val)
        setattr(new_args, "model_path", "{}/{}_{}".format(args.ablation_name, args.ablation_parameter, param_val))
        all_args.append(new_args)
    
    results = []
    for arg in tqdm(all_args):
        results.append(main(arg))
    
    mean_coverages = [res[0] for res in results]
    std_coverages = [res[1] for res in results]
    mean_lengths = [res[2] for res in results]
    std_lengths = [res[3] for res in results]
    plt.clf()
    plt.plot(param_values, mean_lengths)
    plt.xlabel(r'$q$')
    plt.ylabel("length")
    plt.savefig("images/{}/mean_length.png".format(args.ablation_name))


    
