
import torch
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
# warnings.filterwarnings("ignore", category=UserWarning)
def row_differences(matrix):
    n, d = matrix.shape
    expanded_matrix = matrix.unsqueeze(1)
    differences = expanded_matrix - expanded_matrix.transpose(0, 1)
    return differences

def nd_kernel(X, y):
    diff_X = row_differences(X)
    diff_Y = row_differences(y)
    kernel_X = torch.sum(torch.exp(-1 * torch.sum(torch.square(diff_X), dim=2)), dim=1)
    kernel_Y = torch.sum(torch.exp(-1 * torch.sum(torch.square(diff_Y), dim=2)), dim=1)
    scores = kernel_X * kernel_Y
    return scores

def nd_kernel_single(x, y, allX, ally):
    diff_x = torch.sum(torch.exp(-1 * torch.sum(torch.square(x - allX), dim=1)))
    diff_y = torch.sum(torch.exp(-1 * torch.sum(torch.square(y - ally), dim=1)))
    return (diff_x * diff_y).item()

# def nd_kernel_sci(X, y):
#     kde = KernelDensity(kernel='gaussian', bandwidth=.1, atol=1e-5).fit(X)
#     X_scores = np.exp(kde.score_samples(X))
#     kde = KernelDensity(kernel='gaussian', bandwidth=.1, atol=1e-5).fit(y)
#     y_scores = np.exp(kde.score_samples(y))
#     return torch.tensor(X_scores * y_scores)

# def nd_kernel_single_sci(x, y, kdex, kdey):
#     X_scores = np.exp(kdex.score_samples(x.reshape(1, -1)))
#     y_scores = np.exp(kdey.score_samples(y.reshape(1, -1)))
#     return torch.tensor(X_scores * y_scores)

def plot(args, i, all_label_scores, range_vals, cutoff):
    if not os.path.exists("images/{}".format(args.dataset_name)):
        os.mkdir("images/{}".format(args.dataset_name))
        os.mkdir("images/{}/lei".format(args.dataset_name))
    plt.clf()
    plt.plot(range_vals,all_label_scores)
    plt.plot([torch.min(range_vals), torch.max(range_vals)], [cutoff, cutoff], label=r'Confidence Level $\alpha$')
    plt.savefig("images/{}/lei/{}.png".format(args.dataset_name, i))

def get_cov_len(i, args, range_vals, X_train, y_train, X_val, y_val):
    all_label_scores  = []

    cutoff = .1
    for r in range_vals:
        print(r)
        X_aug = torch.cat((X_train, torch.tensor(X_val[i]).unsqueeze(0)))
        y_aug = torch.cat((y_train, torch.tensor([r]).unsqueeze(0)))
        kde_x = KernelDensity(kernel='gaussian', bandwidth=.1, rtol=1e-1).fit(X_aug)
        kde_y = KernelDensity(kernel='gaussian', bandwidth=.1, rtol=1e-1).fit(y_aug)
        X_scores = np.exp(kde_x.score_samples(X_aug))
        y_scores = np.exp(kde_y.score_samples(y_aug))
        all_scores = X_scores * y_scores
        sorted_indices = torch.argsort(torch.tensor(all_scores))
        relative_rank = torch.where(sorted_indices == len(X_aug)- 1)[0].item()/(len(X_aug) - 1)

        all_label_scores.append(relative_rank)
        
    plot(args, i, all_label_scores, range_vals, cutoff)
    coverage, length = calc_length_coverage(all_label_scores, range_vals, cutoff, y_val[i])
    return coverage, length

def lei(args):
    input_size, range_vals = get_input_and_range(args)
    train_loader, val_loader = get_loaders(args)

    X_train = train_loader.dataset.tensors[0]
    y_train = train_loader.dataset.tensors[1].unsqueeze(-1)
    X_val = val_loader.dataset.tensors[0]
    y_val = val_loader.dataset.tensors[1].unsqueeze(-1)

    # all_train_scores = nd_kernel(X_train, y_train)
    real_get_cov_len = partial(get_cov_len, args=args,range_vals =range_vals, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    lengths = []
    coverages = []
    
    num_processes = 2

    coverages = []
    lengths = []
    for i in range(len(X_val)):
        coverage, length = real_get_cov_len(i)
        coverages.append(coverage)
        lengths.append(length)
    # # Create a Pool of worker processes
    # with multiprocessing.Pool(processes=num_processes) as pool:
    #     # Use tqdm to create a progress bar
    #     results = list(tqdm(pool.imap(real_get_cov_len, list(range(len(X_val)))), total=(len(X_val))))
    # coverages = [res[0] for res in results]
    # lengths = [res[1] for res in results]

    return torch.mean(torch.tensor(coverages).float()).item(), torch.std(torch.tensor(coverages).float()).item(), torch.mean(torch.tensor(lengths).float()).item(), torch.std(torch.tensor(lengths).float()).item()

            




