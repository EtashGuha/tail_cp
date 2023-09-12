
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

# def nd_kernel_sci(X, y):
#     kde = KernelDensity(kernel='gaussian', bandwidth=.1, atol=1e-5).fit(X)
#     X_scores = np.exp(kde.score_samples(X))
#     kde = KernelDensity(kernel='gaussian', bandwidth=.1, atol=1e-5).fit(y)
#     y_scores = np.exp(kde.score_samples(y))
#     return np.tensor(X_scores * y_scores)

# def nd_kernel_single_sci(x, y, kdex, kdey):
#     X_scores = np.exp(kdex.score_samples(x.reshape(1, -1)))
#     y_scores = np.exp(kdey.score_samples(y.reshape(1, -1)))
#     return np.tensor(X_scores * y_scores)

def plot(args, i, all_label_scores, range_vals, cutoff):
    if not os.path.exists("images/{}".format(args.dataset_name)):
        os.mkdir("images/{}".format(args.dataset_name))
        os.mkdir("images/{}/lei".format(args.dataset_name))
    plt.clf()
    plt.plot(range_vals,all_label_scores)
    plt.plot([np.min(np.asarray(range_vals)), np.max(np.asarray(range_vals))], [cutoff, cutoff], label=r'Confidence Level $\alpha$')
    plt.savefig("images/{}/lei/{}.png".format(args.dataset_name, i))

def get_cov_len_fast(i, args, range_vals, X_train, y_train, X_val, y_val, scores_X, scores_y):
    h=.1
    diff_Xi = np.exp(-1 * np.sum(np.square(X_val[i] - X_train), axis=1)/(h*h))
    new_scores_X = scores_X + diff_Xi
    all_scores_X = np.concatenate((new_scores_X, np.asarray([np.sum(diff_Xi) + 1])))
    all_label_scores = []
    for r in range_vals:
        diff_yi = np.exp(-1 * np.square(r.item() - y_train)/(h*h))
        new_scores_y = scores_y + diff_yi.flatten()
        all_scores_y = np.concatenate((new_scores_y, np.asarray([np.sum(diff_yi) + 1])))
        all_scores = all_scores_y * all_scores_X
        sorted_indices = np.argsort(all_scores)
        relative_rank = np.where(sorted_indices == len(X_train))[0].item()/(len(X_train))
        
        all_label_scores.append(relative_rank)
        
    plot(args, i, all_label_scores, range_vals, .1)
    coverage, length = calc_length_coverage(all_label_scores, range_vals, .1, y_val[i].item())
    return coverage, length



def get_cov_len(i, args, range_vals, X_train, y_train, X_val, y_val):
    all_label_scores  = []

    cutoff = .1
    for r in range_vals:
        X_aug = np.concatenate((X_train, np.expand_dims(np.asarray(X_val[i]), 0)))
        y_aug = np.concatenate((y_train, np.expand_dims(np.asarray([r]), 0)))
        # kde_x = KernelDensity(kernel='gaussian', bandwidth=.1, rtol=1e-1).fit(X_aug)
        # kde_y = KernelDensity(kernel='gaussian', bandwidth=.1, rtol=1e-1).fit(y_aug)
        # X_scores = np.exp(kde_x.score_samples(X_aug))
        # y_scores = np.exp(kde_y.score_samples(y_aug))
        # all_scores = X_scores * y_scores
        all_scores = nd_kernel(X_aug, y_aug, h=.1)
        sorted_indices = np.argsort(all_scores)
        relative_rank = np.where(sorted_indices == len(X_aug)- 1)[0].item()/(len(X_aug) - 1)

        all_label_scores.append(relative_rank)
        
    plot(args, i, all_label_scores, range_vals, cutoff)
    coverage, length = calc_length_coverage(all_label_scores, range_vals, cutoff, y_val[i])
    return coverage, length

def lei(args):
    input_size, range_vals = get_input_and_range(args)
    train_loader, val_loader = get_loaders(args)

    X_train = train_loader.dataset.tensors[0].detach().numpy().astype('float16')
    y_train = train_loader.dataset.tensors[1].unsqueeze(-1).detach().numpy().astype('float16')
    X_val = val_loader.dataset.tensors[0].detach().numpy().astype('float16')
    y_val = val_loader.dataset.tensors[1].unsqueeze(-1).detach().numpy().astype('float16')

    # all_train_scores = nd_kernel(X_train, y_train)
    real_get_cov_len = partial(get_cov_len, args=args,range_vals =range_vals, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    h = .1
    diff_X = row_differences(X_train)
    diff_Y = row_differences(y_train)
    kernel_X = np.sum(np.exp(-1 * np.sum(np.square(diff_X), axis=2)/(h*h)), axis=1)
    kernel_Y = np.sum(np.exp(-1 * np.sum(np.square(diff_Y), axis=2)/(h*h)), axis=1)
    real_get_cov_len_fast = partial(get_cov_len_fast, args=args,range_vals =range_vals, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, scores_X=kernel_X, scores_y= kernel_Y)

    lengths = []
    coverages = []
    
    num_processes = 10

    # coverages = []
    # lengths = []
    # for i in tqdm(range(len(X_val))):
    #     coverage, length = real_get_cov_len_fast(i)
    #     coverages.append(coverage)
    #     lengths.append(length)
    # Create a Pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use tqdm to create a progress bar
        results = list(tqdm(pool.imap(real_get_cov_len_fast, list(range(len(X_val)))), total=(len(X_val))))
    coverages = [res[0] for res in results]
    lengths = [res[1] for res in results]

    return np.mean(coverages).item(), np.std(coverages).item(), np.mean(lengths).item(), np.std(lengths).item()

            




