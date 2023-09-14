
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

def row_differences(matrix, initial_matrix=None):
    if initial_matrix is None:
        n, d = matrix.shape
        expanded_matrix = matrix[:, np.newaxis, :]  # Add a new axis to the matrix
        # Calculate the pairwise differences
        diff_matrix = expanded_matrix - expanded_matrix.transpose(1, 0, 2)
        return diff_matrix
    else:
        initial_matrix[-1, :, :] = matrix[-1] - matrix
        initial_matrix[:, -1, :] = matrix - matrix[-1]
        return initial_matrix

def nd_kernel(X, y, h=.1, initial_kernel_X=None, initial_kernel_Y=None):
    diff_X = row_differences(X)
    diff_Y = row_differences(y)
    kernel_X = np.sum(np.exp(-1 * np.sum(np.square(diff_X), axis=2)/(h*h)), axis=1)
    kernel_Y = np.sum(np.exp(-1 * np.sum(np.square(diff_Y), axis=2)/(h*h)), axis=1)
    scores = kernel_X * kernel_Y
    return scores


def nd_kernel_single(x, y, allX, ally):
    diff_x = np.sum(np.exp(-1 * np.sum(np.square(x - allX), dim=1)))
    diff_y = np.sum(np.exp(-1 * np.sum(np.square(y - ally), dim=1)))
    return (diff_x * diff_y).item()

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
    train_loader, cal_loader, val_loader = get_loaders(args)

    X_train = train_loader.dataset.tensors[0].detach().numpy().astype('float16')
    y_train = train_loader.dataset.tensors[1].unsqueeze(-1).detach().numpy().astype('float16')
    X_cal = cal_loader.dataset.tensors[0].detach().numpy().astype('float16')
    y_cal = cal_loader.dataset.tensors[1].unsqueeze(-1).detach().numpy().astype('float16')
    X_val = val_loader.dataset.tensors[0].detach().numpy().astype('float16')
    y_val = val_loader.dataset.tensors[1].unsqueeze(-1).detach().numpy().astype('float16')

    h = .1
    cal_scores = get_cal_data(X_train, y_train, X_cal, y_cal)
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

            




