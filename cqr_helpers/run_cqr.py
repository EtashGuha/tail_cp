# CQR Imports
import torch
import random
import numpy as np
import cqr_helpers.helper as helper
from cqr_helpers.nonconformist.cp import IcpRegressor
from cqr_helpers.nonconformist.base import RegressorAdapter
from cqr_helpers.nonconformist.nc import QuantileRegErrFunc
from cqr_helpers.nonconformist.nc import RegressorNc

def run_cqr(X_train, y_train, X_val, y_val):
    nn_learn_func = torch.optim.Adam
    epochs = 1000
    lr = 0.0005
    hidden_size = 64
    batch_size = 64
    dropout = 0.1
    wd = 1e-6
    quantiles_net = [0.1, 0.9]
    cv_test_ratio = 0.05
    cv_random_state = 1

    X_train_cqr = np.asarray(X_train)
    y_train_cqr = np.asarray(y_train)
    X_val_cqr = np.asarray(X_val)
    y_val_cqr = np.asarray(y_val)
    n_train = X_train.shape[0]
    in_shape = X_train.shape[1]

    print("Size: train (%d, %d), test (%d, %d)" % (X_train.shape[0], X_train.shape[1], X_val.shape[0], X_val.shape[1]))
    significance = 0.1
    # divide the data into proper training set and calibration set

    
    ## This code takes the place of run_icp from icp.helper
    cqr_model = helper.AllQNet_RegressorAdapter(model=None,
                                            fit_params=None,
                                            in_shape = in_shape,
                                            hidden_size = hidden_size,
                                            quantiles = quantiles_net,
                                            learn_func = nn_learn_func,
                                            epochs = epochs,
                                            batch_size=batch_size,
                                            dropout=dropout,
                                            lr=lr,
                                            wd=wd,
                                            test_ratio=cv_test_ratio,
                                            random_state=cv_random_state,
                                            use_rearrangement=False)
    nc = RegressorNc(cqr_model, QuantileRegErrFunc())
    condition=None
    icp = IcpRegressor(nc,condition=condition)


    # Fit the ICP using the proper training set
    icp.fit(X_train_cqr, y_train_cqr)

    # Calibrate the ICP using the calibration set
    icp.calibrate(X_val_cqr, y_val_cqr)

    predictions = icp.predict(X_val_cqr, significance=significance)
    cqr_lower = predictions[:,0]
    cqr_upper = predictions[:,1]
    
    # Clipping the output ranges to what's been seen in the train data
    max_y = max(y_train_cqr)
    min_y = min(y_train_cqr)
    cqr_lower_clipped = np.array([max(min_y, y) for y in cqr_lower])
    cqr_upper_clipped = np.array([min(max_y, y) for y in cqr_upper])

    avg_coverage, std_coverage, avg_length, std_length = helper.compute_coverage(y_val_cqr,cqr_lower_clipped,cqr_upper_clipped,significance,"CQR Net")
    print(f"CQR Coverage: {avg_coverage} +- {std_coverage} Length: {avg_length} +- {std_length}")
    return avg_coverage, std_coverage, avg_length, std_length, cqr_upper_clipped, cqr_lower_clipped