import matplotlib.pyplot as plt
import os
from cp import percentile_excluding_index
import torch
from cp import calc_length_coverage
import pickle

import seaborn as sns
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


def plot_prob(args, range_vals, X_val, y_val, model, cqr_lower=None,cqr_upper=None):
    set_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    if not os.path.exists("images"):
        os.mkdir("images")
    if not os.path.exists("images/{}".format(args.model_path)):
        os.mkdir("images/{}".format(args.model_path))
        os.mkdir("images/{}/right".format(args.model_path))
        os.mkdir("images/{}/wrong".format(args.model_path))
    step_val = (max(range_vals) - min(range_vals))/len(range_vals)
    indices = ((y_val - min(range_vals))/step_val).to(torch.int)
    indices[indices == len(range_vals)] = indices[indices == len(range_vals)] - 1
    scores = torch.nn.functional.softmax(model(X_val), dim=1)
    all_scores = scores[torch.arange(len(X_val)), indices.long()]
    # cqr_scores the same way and then plot both
    ## Manually added cqr stuff
    # cqr_scores = None
    # with open('/Users/shloknatarajan/tail_cp/cqr_models/concrete-2023-09-02-17:31:34.pkl', 'rb') as f:
    #     cqr_model = pickle.load(f)
    #     cqr_scores = torch.nn.functional.softmax(cqr_model(X_val), dim=1)
    #     cqr_scores = cqr_scores[torch.arange(len(X_val)), indices.long()]

    alpha = args.alpha
    for i in range(len(X_val)):
        plt.clf()
        plt.plot(range_vals.detach().numpy(), scores[i].detach().numpy(), label="ours")
        plt.xlabel(r'$y$')
        plt.ylabel(r'$\mathbb{P}(y \mid x_{n+1})$')
        plt.grid(None)
        plt.tight_layout()

        percentile_val = percentile_excluding_index(all_scores, i, alpha)
        coverage, length = calc_length_coverage(scores[i], range_vals, percentile_val, y_val[i])

        if cqr_lower is not None:
            plt.plot([cqr_lower[i], cqr_lower[i]], [torch.min(scores).detach().numpy(), torch.max(scores).detach().numpy()], label="cqr lower", color="red")
        if cqr_upper is not None:
            plt.plot([cqr_upper[i], cqr_upper[i]], [torch.min(scores).detach().numpy(), torch.max(scores).detach().numpy()], label="cqr upper", color="red")
        plt.plot([torch.min(range_vals).detach().numpy(), torch.max(range_vals).detach().numpy()], [percentile_val.detach().numpy(), percentile_val.detach().numpy()], label=r'Confidence Level $\alpha$')
        plt.plot([y_val[i].detach().numpy(), y_val[i].detach().numpy()], [torch.min(scores).detach().numpy(), torch.max(scores).detach().numpy()], label=r'Ground Truth $y_{n+1}$')

        plt.legend()
        
        if coverage == 1:
            plt.savefig("images/{}/right/{}.png".format(args.model_path, i))
        else:
            plt.savefig("images/{}/wrong/{}.png".format(args.model_path, i))

