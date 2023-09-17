import matplotlib.pyplot as plt
import os
from cp import percentile_excluding_index
import torch
from cp import calc_length_coverage, find_intervals_above_value_with_interpolation, get_all_scores
import seaborn as sns
import pickle
import numpy as np
def find_rank(value, value_list):
    sorted_list = sorted(value_list)
    rank = 1
    for item in sorted_list:
        if value > item:
            rank += 1
        else:
            break
    return rank

def calculate_ranks(scores,all_scores):
    ranks = [find_rank(score, all_scores) for score in scores]
    return np.asarray(ranks)/len(all_scores)


def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")
    # Set the font to be serif, rather than sans
    sns.set(font='serif', font_scale=1.5)
    sns.set_palette('muted')
    # Make the background white, and specify the
    # specific font family
    sns.set_style("whitegrid", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })

def plot_path(args, range_vals, X_val, y_val, model):
    # set_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    if not os.path.exists("images/{}".format(args.model_path)):
        os.mkdir("images/{}".format(args.model_path))
    
    scores, all_scores = get_all_scores(range_vals, X_val, y_val, model)

    alpha = args.alpha

    plt.scatter(X_val.detach().numpy(), y_val.detach().numpy(), label=r'(x_i, y_i)')
    for i in range(len(X_val)):
        percentile_val = percentile_excluding_index(all_scores, alpha)
        intervals = find_intervals_above_value_with_interpolation(range_vals, scores[i], percentile_val)
        for interval in intervals:
            plt.scatter([X_val[i].detach().numpy(), X_val[i].detach().numpy()], [interval[0].detach().numpy(), interval[1].detach().numpy()], color="orange")
    plt.legend()
    plt.savefig("images/{}/dcp.png".format(args.model_path))

        
def plot_prob(args, range_vals, X_val, y_val, model):
    # set_style()
    if not os.path.exists("images/{}".format(args.model_path)):
        os.mkdir("images/{}".format(args.model_path))
    if not os.path.exists("images/{}/right".format(args.model_path)):
        os.mkdir("images/{}/right".format(args.model_path))
    if not os.path.exists("images/{}/wrong".format(args.model_path)):
        os.mkdir("images/{}/wrong".format(args.model_path))
    if not os.path.exists("images/{}/pi".format(args.model_path)):
        os.mkdir("images/{}/pi".format(args.model_path))
        

    scores, all_scores = get_all_scores(range_vals, X_val, y_val, model)


    alpha = args.alpha
    for i in range(len(X_val[:25])):
        fig, ax = plt.subplots()
        sns.set_style("ticks", {
            "font.family": "serif",
            "font.serif": ["Times", "Palatino", "serif"]
        })
        sns.lineplot(
            ax=ax,
            x=range_vals.detach().numpy(),
            y=scores[i].detach().numpy(),
            label=r'$\mathbb{Q}(y \mid x_{n+1})$',
            color='black',
            linewidth=2.8,
            marker='o',
            markerfacecolor='white',
            markeredgecolor='black'        
        )
        ax.set(title=f"{args.model_path}", xlabel=r'$y$', ylabel=r'$\mathbb{P}(y \mid x_{n+1})$')
        percentile_val = percentile_excluding_index(all_scores, alpha)
        coverage, length = calc_length_coverage(scores[i], range_vals, percentile_val, y_val[i])
        ax.axhline(y=percentile_val.detach().numpy(), label=r'Confidence Level $\alpha$', color='#a8acb3', linestyle='--',)
        ax.axvline(x=y_val[i].detach().numpy(), label=r'Ground Truth $y_{n+1}$', color='#646566', linestyle=':',)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        if coverage == 1:
            fig.savefig("images/{}/right/{}.png".format(args.model_path, i))
        else:
            fig.savefig("images/{}/wrong/{}.png".format(args.model_path, i))
    
    for i in range(len(X_val[:25])):
        plt.clf()
        list_of_ranks = calculate_ranks(scores[i], all_scores)
        plt.plot(range_vals.detach().numpy(), list_of_ranks)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$\pi(z)$')
        plt.savefig("images/{}/pi/{}.png".format(args.model_path, i))

def plot_violin(args, coverages, lengths):
    
    with open("saved_results/{}/lei.pkl".format(args.dataset_name), "rb") as f:
        lei_coverages, lei_lengths = pickle.load(f)
    
    with open("saved_results/{}/ridge.pkl".format(args.dataset_name), "rb") as f:
        ridge_coverages, ridge_lengths = pickle.load(f)
    
    all_coverages = [coverages, lei_coverages, ridge_coverages]
    all_lengths = [torch.stack(lengths).detach().numpy(), torch.stack(lei_lengths).detach().numpy(), ridge_lengths]
    labels = ["Ours", "Lei", "Ridge" ]

    plt.clf()
    sns.set(style="whitegrid")  # Optional styling
    plt.figure(figsize=(8, 6))  # Optional figure size

    # Use the violinplot function to create the plot
    sns.violinplot(data=all_coverages, inner="box", palette="Set3")

    # Set labels and title
    plt.xticks(range(len(labels)), labels)
    plt.xlabel('Coverages')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig("images/{}/violin_coverage.png".format(args.model_path))

    plt.clf()
    sns.set(style="whitegrid")  # Optional styling
    plt.figure(figsize=(8, 6))  # Optional figure size

    # Use the violinplot function to create the plot
    sns.violinplot(data=all_lengths, inner="box", palette="Set3")

    # Set labels and title
    plt.xticks(range(len(labels)), labels)
    plt.xlabel('Lengths')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig("images/{}/violin_length.png".format(args.model_path))


