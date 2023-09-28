import matplotlib.pyplot as plt
import os
from cp import percentile_excluding_index
import torch
from cp import calc_length_coverage, find_intervals_above_value_with_interpolation, get_all_scores
import seaborn as sns
import pickle
import numpy as np
from data import get_train_val_data
from labellines import labelLine, labelLines

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

def plot_path(args, range_vals, X_val, y_val, model):
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
    plt.savefig("images/{}/dcp.pdf".format(args.model_path))

        
def plot_prob(args, range_vals, X_val, y_val, model, baselines=True):
    baseline_path = 'baselines' if baselines else 'no_baselines'
    plt.rcParams["mathtext.fontset"] = "cm"
    if not os.path.exists("images/"):
        os.mkdir("images/")
    if not os.path.exists(f"images/{args.model_path}"):
        os.mkdir(f"images/{args.model_path}")
    if not os.path.exists(f"images/{args.model_path}/{baseline_path}"):
        os.mkdir(f"images/{args.model_path}/{baseline_path}")
    if not os.path.exists(f"images/{args.model_path}/{baseline_path}/right"):
        os.mkdir(f"images/{args.model_path}/{baseline_path}/right")
    if not os.path.exists(f"images/{args.model_path}/{baseline_path}/wrong"):
        os.mkdir(f"images/{args.model_path}/{baseline_path}/wrong")
    if not os.path.exists(f"images/{args.model_path}/{baseline_path}/pi"):
        os.mkdir(f"images/{args.model_path}/{baseline_path}/pi")

    ## Load Baselines
    cqr_lower = None
    cqr_upper = None
    chr_probs = None
    lei_probs = None
    scores, all_scores = get_all_scores(range_vals, X_val, y_val, model)
    alpha = args.alpha
    for i in range(len(X_val[:25])):
        if baselines:
            # Load CQR
            if os.path.exists(f"saved_results/{args.dataset_name}/cqr_predictions.pkl"):
                with open(f"saved_results/{args.dataset_name}/cqr_predictions.pkl", "rb") as f:
                        cqr_lower, cqr_upper = pickle.load(f)

            # Load Lei
            if os.path.exists(f"saved_results/{args.dataset_name}/lei_probs.pkl"):
                with open(f"saved_results/{args.dataset_name}/lei_probs.pkl", "rb") as f:
                        lei_probs = pickle.load(f)[i]
            
            # Load CHR
            if os.path.exists(f"saved_results/{args.dataset_name}/chr_probs.pkl"):
                with open(f"saved_results/{args.dataset_name}/chr_probs.pkl", "rb") as f:
                    in_probs = pickle.load(f)[i]
                    if args.dataset_name == 'solar':
                        in_probs = np.delete(in_probs, np.arange(9, len(in_probs), 10))
                    subsequence_length = len(in_probs) // len(lei_probs)
                    reshaped_chr = in_probs.reshape(len(lei_probs), subsequence_length)
                    chr_probs = np.sum(reshaped_chr, axis=1)
        fig, ax = plt.subplots(constrained_layout=True)
        sns.set_style("ticks", {
            "font.family": "serif",
            "font.serif": ["Times", "Palatino", "serif"]
        })
        # ax.set_title(label=f"{args.model_path}", y=1.0, pad=28, fontdict={"family": "Times New Roman", "size": 15})
        ax.set_xlabel(r'$y$', fontdict={"family": "Times New Roman", "size": 20})
        ax.set_ylabel(r'$\mathbb{P}(y \mid x_{n+1})$', fontdict={"family": "Times New Roman", "size": 20})
        

        # Plot Ours
        if baselines:
            sns.lineplot(
                ax=ax,
                x=range_vals.detach().numpy(),
                y=scores[i].detach().numpy(),
                label=r'$\mathbb{Q}(y \mid x_{n+1})$',
                color='black',
                linewidth=2.3    
            )
            # Plot Lei
            if lei_probs is not None:
                sns.lineplot(
                    ax=ax,
                    x=range_vals.detach().numpy(),
                    y=lei_probs,
                    label='KDE',
                    color='black',
                    linewidth=1.5,
                    linestyle='dotted'
                )
                
            # Plot CHR
            if chr_probs is not None:
                sns.lineplot(
                    ax=ax,
                    x=range_vals.detach().numpy(),
                    y=chr_probs,
                    label='CHR',
                    color='black',
                    linestyle='--',
                    linewidth=1.7
                )
        else:
            sns.lineplot(
                ax=ax,
                x=range_vals.detach().numpy(),
                y=scores[i].detach().numpy(),
                label=r'$\mathbb{Q}(y \mid x_{n+1})$',
                color='black',
                linewidth=2.3,
                marker='o',
                markerfacecolor='white',
                markeredgecolor='black',
                markersize=5        
            )

        percentile_val = percentile_excluding_index(all_scores, alpha)
        coverage, length = calc_length_coverage(scores[i], range_vals, percentile_val, y_val[i])

        # Ground Truth and Confidence Level
        confidence_level = ax.axhline(y=percentile_val.detach().numpy(), label=r'Confidence Level $\alpha$', color='#cccccc', zorder=-1)
        ground_truth = ax.axvline(x=y_val[i].detach().numpy(), label=r'Ground Truth $y_{n+1}$', color='#cccccc', zorder=-1)

        # Add labels to the lines
        ax.text(
            x=1,
            y=percentile_val.detach().numpy(),
            s=r'Confidence Level $\alpha$',
            ha='right',
            va='bottom',
            transform=ax.get_yaxis_transform(),
            size=13,
            color="grey"
        )
        if baselines:
            ax.text(
                y_val[i].detach().numpy(),
                1.10,
                'Ground Truth',
                color='grey',
                ha='center',
                va='top',
                transform=ax.get_xaxis_transform(),
                size=12
            )
        else:
            ax.text(
                y_val[i].detach().numpy(),
                1.08,
                'Ground Truth',
                color='grey',
                ha='center',
                va='top',
                transform=ax.get_xaxis_transform(),
                size=12
            )
        # CQR Lower and Upper Predictions
        cqr_color = '#adb5bd'
        if cqr_lower is not None:
            ax.axvline(x=cqr_lower[i], label=r'CQR Lower', color=cqr_color, zorder=-1)
            ax.text(
                cqr_lower[i],
                1.05,
                'CQR Lower',
                color='grey',
                ha='center',
                va='top',
                size=12,
                transform=ax.get_xaxis_transform()
            )
        if cqr_upper is not None:
            ax.axvline(x=cqr_upper[i], label=r'CQR Upper', color=cqr_color, zorder=-1)
            ax.text(
                cqr_upper[i],
                1.05,
                'CQR Upper',
                color='grey',
                ha='center',
                va='top',
                size=12,
                transform=ax.get_xaxis_transform()
            )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Specify the lines to include in the legend
        lines_to_include = [r'$\mathbb{Q}(y \mid x_{n+1})$', 'KDE', 'CHR']
        
        # Create a custom legend with only the specified lines
        handles, labels = ax.get_legend_handles_labels()
        filtered_handles = [handle for handle, label in zip(handles, labels) if label in lines_to_include]
        filtered_labels = [label for label in labels if label in lines_to_include]
        ax.legend(filtered_handles, filtered_labels, fontsize=14)

        if coverage == 1:
            fig.savefig(f"images/{args.model_path}/{baseline_path}/right/{args.dataset_name}_{baseline_path}_{i}.pdf")
        else:
            fig.savefig(f"images/{args.model_path}/{baseline_path}/wrong/{args.dataset_name}_{baseline_path}_{i}.pdf")
        fig.clf()

    for i in range(len(X_val[:25])):
        plt.clf()
        list_of_ranks = calculate_ranks(scores[i], all_scores)
        plt.plot(range_vals.detach().numpy(), list_of_ranks)
        plt.xlabel(r'$z$')
        plt.ylabel(r'$\pi(z)$')
        plt.savefig(f"images/{args.model_path}/{baseline_path}/pi/{i}.pdf")
        plt.clf()

def plot_violin(args, coverages, lengths):
    with open("saved_results/{}/lei.pkl".format(args.dataset_name), "rb") as f:
        lei_coverages, lei_lengths = pickle.load(f)
    
    with open("saved_results/{}/ridge.pkl".format(args.dataset_name), "rb") as f:
        ridge_coverages, ridge_lengths = pickle.load(f)
    
    with open("saved_results/{}/cqr.pkl".format(args.dataset_name), "rb") as f:
        cqr_coverages, cqr_lengths = pickle.load(f)

    with open("saved_results/{}/cqr_nc.pkl".format(args.dataset_name), "rb") as f:
        cqr_nc_coverages, cqr_nc_lengths = pickle.load(f)
    
    with open("saved_results/{}/cb.pkl".format(args.dataset_name), "rb") as f:
        cb_coverages, cb_lengths = pickle.load(f)

    with open("saved_results/{}/chr.pkl".format(args.dataset_name), "rb") as f:
        chr_coverages, chr_lengths = pickle.load(f)
    
    sns.set_style("whitegrid", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })
    # sns.set_context("talk")
    # Mean cb_coverages
    cb_coverages_axis_means = [np.mean(cb_coverages[:, i]) for i in range(len(cb_coverages[0]))]
    cb_lengths_axis_means = [np.mean(cb_lengths[:, i]) for i in range(len(cb_coverages[0]))]

    all_coverages = [coverages, lei_coverages, ridge_coverages, cqr_coverages, cqr_nc_coverages, cb_coverages_axis_means, chr_coverages]
    all_lengths = [torch.stack(lengths).detach().numpy(), torch.stack(lei_lengths).detach().numpy(), np.array(ridge_lengths), cqr_lengths, chr_lengths, np.array(cb_lengths_axis_means)]
    labels = ["Ours", "KDE", "Lasso", "CQR", "CHR", "CB"]

    fig_lengths, (ax_lengths_ours, ax_lengths_lei, ax_lengths_ridge, ax_lengths_cqr, ax_lengths_chr, ax_lengths_cb) = plt.subplots(6, sharex=True, constrained_layout=True)
    all_length_axes = [ax_lengths_ours, ax_lengths_lei, ax_lengths_ridge, ax_lengths_cqr, ax_lengths_chr, ax_lengths_cb]

    for i in range(len(all_lengths)):
        if (np.count_nonzero(all_lengths[i] == all_lengths[i][0]) == len(all_lengths[i])):
            all_length_axes[i].axvline(x=all_lengths[i][0], color='black', linestyle='solid')
        else:
            sns.kdeplot(
                x=all_lengths[i],
                ax=all_length_axes[i],
                label=labels[i],
                color='black',
                linewidth=1.7,
                linestyle='solid',
                clip=(0, None)
            )
        all_length_axes[i].set_ylabel(labels[i], fontsize=12)
        all_length_axes[i].set_yticks([])
        all_length_axes[i].set_yticklabels([])
        all_length_axes[i].spines['top'].set_visible(False)
        all_length_axes[i].spines['left'].set_visible(False)
        all_length_axes[i].spines['right'].set_visible(False)
        all_length_axes[i].spines['bottom'].set_color('#cccccc')

    all_length_axes[-1].set_xlabel("Length", fontsize=12)
    all_length_axes[-1].text(0.0, -0.69, 'Best', transform=all_length_axes[-1].transAxes, ha='left', va='bottom', color='#757575')
    all_length_axes[-1].text(0.99, -0.69, 'Worst', transform=all_length_axes[-1].transAxes, ha='right', va='bottom', color='#757575', backgroundcolor='white', zorder=2)
    # all_length_axes[-1].text(0.06, -0.82, '←', transform=all_length_axes[-1].transAxes, ha='left', va='bottom', color='#adadad', fontsize=22)
    # all_length_axes[-1].text(0.91, -0.82, '→', transform=all_length_axes[-1].transAxes, ha='right', va='bottom', color='#adadad', fontsize=22)
    all_length_axes[-1].annotate('', xy=(0.05, -0.574), xycoords='axes fraction', xytext=(0.44, -0.574), 
            arrowprops=dict(arrowstyle="->", color='#adadad'))
    all_length_axes[-1].annotate('', xytext=(0.56, -0.574), xycoords='axes fraction', xy=(0.92, -0.574), 
            arrowprops=dict(arrowstyle="->", color='#adadad'))
    fig_lengths.savefig("images/{}/{}_kdeplot_lengths.pdf".format(args.model_path, args.dataset_name))
    fig_lengths.clf()
