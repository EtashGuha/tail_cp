import numpy as np
from sklearn.model_selection import train_test_split
from interval import interval as intervals
from sklearn.linear_model import lasso_path
from data import get_input_and_range, get_loaders

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.linear_model import lasso_path
import seaborn as sns
import os
import pickle


def logcosh(coef, X, y, lambda_):

    loss = np.sum(np.log(np.cosh(y - X.dot(coef))))
    penalty = 0.5 * np.linalg.norm(coef, ord=2) ** 2

    return loss + lambda_ * penalty


def logcosh_reg(X, y, lambda_, coef=None):

    if coef is None:
        coef = np.zeros(X.shape[1])

    res = minimize(logcosh, x0=coef, args=(X, y, lambda_))
    coef = res.x

    if not res.success:
        print("boom")

    return coef


def linex(coef, X, y, lambda_, gamma=0.5):

    tmp = gamma * (y - X.dot(coef))
    loss = np.sum(np.exp(tmp) - tmp - 1)
    penalty = 0.5 * np.linalg.norm(coef, ord=2) ** 2

    return loss + lambda_ * penalty


def linex_reg(X, y, lambda_, gamma=0.5, coef=None):

    if coef is None:
        coef = np.zeros(X.shape[1])

    res = minimize(linex, x0=coef, args=(X, y, lambda_, gamma))
    coef = res.x

    if not res.success:
        print("boom")

    return coef


def cross_val(X, y, method="lasso"):
    """
        Perform a 5-fold cross-validation and return the mean square errors for
        different parameters lambdas.
    """

    n_samples, n_features = X.shape
    n_lambdas = 100
    # lambda_max = np.linalg.norm(X.T.dot(y), ord=np.inf)
    lambda_max = np.linalg.norm(X.T.dot(y))
    lambdas = lambda_max * np.logspace(0, -2, n_lambdas)

    KF = KFold(n_splits=5, shuffle=True)
    n_folds = KF.get_n_splits()
    errors = np.zeros((n_lambdas, n_folds))
    i_fold = 0

    for train_index, test_index in KF.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for l in range(n_lambdas):

            if method is "lasso":
                lmd = [lambdas[l] / X.shape[0]]
                res = lasso_path(X_train, y_train, alphas=lmd, eps=1e-12,
                                 max_iter=int(1e8))
                coef = res[1].ravel()

            elif method is "logcosh":
                coef = logcosh_reg(X_train, y_train, lambdas[l])

            elif method is "linex":
                coef = linex_reg(X_train, y_train, lambdas[l])

            y_pred = np.dot(X_test, coef)
            errors[l, i_fold] = np.mean((y_pred - y_test) ** 2)

        i_fold += 1

    i_best = np.argmin(np.mean(errors, axis=1))

    return lambdas[i_best]


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

def conf_pred(args, lambda_, method="lasso"):
    if os.path.exists("saved_results/{}/ridge.pkl".format(args.dataset_name)):
        with open("saved_results/{}/ridge.pkl".format(args.dataset_name), "rb") as f:
            coverages, lengths = pickle.load(f)
    else:
        input_size, range_vals = get_input_and_range(args)
        train_loader, val_loader = get_loaders(args)

        X_train = train_loader.dataset.tensors[0].detach().numpy().astype('float16')
        Y_train = train_loader.dataset.tensors[1].unsqueeze(-1).detach().numpy().astype('float16')
        X_val = val_loader.dataset.tensors[0].detach().numpy().astype('float16')
        y_val = val_loader.dataset.tensors[1].unsqueeze(-1).detach().numpy().astype('float16')


        # Training
        if method is "lasso":
            lmd = [lambda_ / X_train.shape[0]]
            res = lasso_path(X_train, Y_train, alphas=lmd, eps=1e-12)
            coef = res[1].ravel()

        elif method is "logcosh":
            coef = logcosh_reg(X_train, Y_train, lambda_)

        elif method is "linex":
            coef = linex_reg(X_train, Y_train, lambda_)

        # Ranking on the test
        mu = X_val.dot(coef)
        sorted_residual = np.sort(np.abs(y_val.flatten() - mu.flatten()))
        index = int((X_val.shape[0]) * (1 - args.alpha))
        quantile = sorted_residual[index]
        
        coverages = []
        lengths = []
        for i in range(len(X_val)):  
            mu = X_val[i].dot(coef)
            if mu - quantile <= y_val[i] <= mu + quantile:
                coverages.append(1)
            else:  
                coverages.append(0)
            lengths.append(2 * quantile)
        if not os.path.exists("saved_results/{}".format(args.dataset_name)):
            os.mkdir("saved_results/{}".format(args.dataset_name))
        with open("saved_results/{}/ridge.pkl".format(args.dataset_name), "wb") as f:
            pickle.dump((coverages, lengths), f)
    return np.mean(coverages), np.std(coverages), np.mean(lengths), np.std(lengths), np.std(coverages)/np.sqrt(len(coverages)), np.std(lengths)/np.sqrt(len(lengths))