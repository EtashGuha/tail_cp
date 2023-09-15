
from data import get_input_and_range, get_loaders
import multiprocessing
from cp import percentile_excluding_index, calc_length_coverage
from tqdm import tqdm
import multiprocessing
from functools import partial
from sklearn.neighbors import KernelDensity
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle
import copy

def plot(args, i, all_label_scores, range_vals, cutoff):
    if not os.path.exists("images/{}".format(args.dataset_name)):
        os.mkdir("images/{}".format(args.dataset_name))
        os.mkdir("images/{}/lei".format(args.dataset_name))
    plt.clf()
    plt.plot(range_vals,all_label_scores)
    plt.plot([np.min(np.asarray(range_vals)), np.max(np.asarray(range_vals))], [cutoff, cutoff], label=r'Confidence Level $\alpha$')
    plt.savefig("images/{}/lei/{}.png".format(args.dataset_name, i))

def get_cal_data(X_train, y_train, X_cal, y_cal):
    h=.1
    cal_scores = []
    for i in tqdm(range(len(X_cal))):
        diff_Xi = np.exp(-1 * np.sum(np.square(X_cal[i] - X_train), axis=1)/(h*h))
        diff_yi = np.exp(-1 * np.square(y_cal[i].item() - y_train)/(h*h))
        cal_scores.append((np.sum(diff_Xi) + 1) * (np.sum(diff_yi) + 1))
    return cal_scores

def get_cov_len_fast(i, args, range_vals, cal_scores, X_train, y_train, X_val, y_val):
    h=.1
    diff_Xi = np.exp(-1 * np.sum(np.square(X_val[i] - X_train), axis=1)/(h*h))
    all_label_scores = []
    for r in range_vals:
        all_scores = copy.deepcopy(cal_scores)
        diff_yi = np.exp(-1 * np.square(r.item() - y_train)/(h*h))
        new_score = (np.sum(diff_Xi) + 1) * (np.sum(diff_yi) + 1)
        all_scores.append(new_score)
        sorted_indices = np.argsort(all_scores)
        relative_rank = np.where(sorted_indices == len(all_scores) - 1)[0].item()/(len(all_scores) - 1)
        all_label_scores.append(relative_rank)
    
    plot(args, i, all_label_scores, range_vals, .1)
    coverage, length = calc_length_coverage(all_label_scores, range_vals, .1, y_val[i].item())
    return coverage, length

def lei(args):
    input_size, range_vals = get_input_and_range(args)
    train_loader,  val_loader = get_loaders(args)

    X_train = train_loader.dataset.tensors[0].detach().numpy().astype('float16')
    y_train = train_loader.dataset.tensors[1].unsqueeze(-1).detach().numpy().astype('float16')
    X_val = val_loader.dataset.tensors[0].detach().numpy().astype('float16')
    y_val = val_loader.dataset.tensors[1].unsqueeze(-1).detach().numpy().astype('float16')

    h = .1
    cal_scores = get_cal_data(X_train, y_train, X_val, y_val)
    real_get_cov_len_fast = partial(get_cov_len_fast, args=args,range_vals =range_vals,cal_scores=cal_scores, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    lengths = []
    coverages = []
    
    num_processes = 10

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(real_get_cov_len_fast, list(range(len(X_val)))), total=(len(X_val))))
    coverages = [res[0] for res in results]
    lengths = [res[1] for res in results]

    
    if not os.path.exists("saved_results/{}".format(args.dataset_name)):
        os.mkdir("saved_results/{}".format(args.dataset_name))
    with open("saved_results/{}/lei.pkl".format(args.dataset_name), "wb") as f:
        pickle.dump((coverages, lengths), f)
    return np.mean(coverages).item(), np.std(coverages).item(), np.mean(lengths).item(), np.std(lengths).item()

            




