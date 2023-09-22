import os
from os import path
import sys
import torch
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, '..')

from chr.chr.black_boxes import QNet, QRF
from chr.chr.methods import CHR
# from chr.others import CQR, CQR2, DistSplit, DCP
from chr.chr.utils import evaluate_predictions


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from data import get_train_val_data
import pickle
import pdb

def get_chr(args):
    if os.path.exists("saved_results/{}_{}/chr.pkl".format(args.dataset_name, args.seed)):
        with open("saved_results/{}_{}/chr.pkl".format(args.dataset_name, args.seed), "rb") as f:
            coverages, lengths = pickle.load(f)
    else:
        X_train, y_train, X_val, y_val = get_train_val_data(args)
        X_train = X_train.detach().numpy()
        X_val = X_val.detach().numpy()
        y_train = y_train.detach().numpy()
        y_val = y_val.detach().numpy()

        # Default arguments
        alpha = 0.1

        n_train = X_train.shape[0]
        n_features = X_train.shape[1]
        epochs = 2000
        lr = 0.0005
        batch_size = len(X_train)
        dropout = 0.1

        grid_quantiles = np.arange(0.01,1.0,0.01)
        bbox = QNet(grid_quantiles, n_features, no_crossing=True, batch_size=batch_size,
                                dropout=dropout, num_epochs=epochs, learning_rate=lr, calibrate=1,
                                verbose=1)

        bbox.fit(X_train, y_train)
        method = CHR(bbox, ymin=np.min(y_train), ymax=np.max(y_train), y_steps=1000, randomize=True)


        method.calibrate(X_val, y_val, alpha)
        # Compute prediction on test data
        pred, histograms = method.predict(X_val)

        # Evaluate results
        res, coverages, lengths = evaluate_predictions(pred, y_val, X=X_val)
        if not os.path.exists("saved_results/{}_{}".format(args.dataset_name, args.seed)):
            os.mkdir("saved_results/{}_{}".format(args.dataset_name, args.seed))
        with open("saved_results/{}_{}/chr.pkl".format(args.dataset_name, args.seed), "wb") as f:
            pickle.dump((coverages, lengths), f)
        with open("saved_results/{}_{}/chr_preds.pkl".format(args.dataset_name, args.seed), "wb") as f:
            pickle.dump((pred), f)
        with open("saved_results/{}_{}/chr_probs.pkl".format(args.dataset_name, args.seed), "wb") as f:
            pickle.dump((histograms), f)
    return np.mean(coverages).item(), np.std(coverages).item(), np.mean(lengths).item(), np.std(lengths).item(), np.std(coverages)/np.sqrt(len(coverages)), np.std(lengths)/np.sqrt(len(lengths))
