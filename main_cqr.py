import torch
import matplotlib.pyplot as plt
from data import get_loaders, get_input_and_range, get_val_data
from create_argparser import get_parser_args
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from models.model import GenModule
import os
from sheets import log_results
from cp import get_cp
from plotter import plot_prob
from models.callbacks import get_callbacks
# CQR Imports
import random
import numpy as np
import helper
from nonconformist.cp import IcpRegressor
from nonconformist.base import RegressorAdapter
from nonconformist.nc import QuantileRegErrFunc
from nonconformist.nc import RegressorNc
import pickle

torch.autograd.set_detect_anomaly(True)
def get_model(args):
    input_size, range_vals = get_input_and_range(args)

    model = GenModule(args, input_size, range_vals)

    total_path = "model_paths/{}.pth".format(args.model_path)
    if os.path.exists(total_path):
        model.load_state_dict(torch.load(total_path))
    else:
        
        train_loader, val_loader = get_loaders(args)
        logger = TensorBoardLogger("tb_logs", name=args.model_path)
        callbacks = get_callbacks(args)
        trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator="cpu", logger=logger, callbacks=callbacks)
        trainer.fit(model, train_loader, val_loader)
        torch.save(model.state_dict(), total_path)

    model.eval()
    return model

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = get_parser_args()
    dataset_name = args.dataset_name
    print(f"Dataset: {dataset_name}")
    model = get_model(args) 


    X_train, y_train, X_val, y_val = get_val_data(args)
    input_size, range_vals = get_input_and_range(args)
    mean_coverage, std_coverage, mean_length, std_length = get_cp(args, range_vals, X_val, y_val, model)
    print(f"CP Coverage: {mean_coverage} Length: {mean_length}")

    #### CQR stuff
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

    idx = np.random.permutation(n_train)
    n_half = int(np.floor(n_train/2))
    idx_train, idx_cal = idx[:n_half], idx[n_half:2*n_half]

    print("Size: train (%d, %d), test (%d, %d)" % (X_train.shape[0], X_train.shape[1], X_val.shape[0], X_val.shape[1]))

    dataset_name_vec = []
    method_vec = []
    coverage_vec = []
    length_vec = []
    seed_vec = []
    significance = 0.1
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    coverage_cp_qnet=0
    length_cp_qnet=0
    # divide the data into proper training set and calibration set

    

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
    y_lower, y_upper = helper.run_icp(nc, X_train_cqr, y_train_cqr, X_val_cqr, idx_train, idx_cal, significance=0.1)
    condition=None
    icp = IcpRegressor(nc,condition=condition)

    # Fit the ICP using the proper training set
    icp.fit(X_train_cqr[idx_train,:], y_train_cqr[idx_train])

    # Calibrate the ICP using the calibration set
    icp.calibrate(X_train_cqr[idx_cal,:], y_train_cqr[idx_cal])

    predictions = icp.predict(X_val_cqr, significance=significance)
    cqr_lower = predictions[:,0]
    cqr_upper = predictions[:,1]

    # helper.plot_func_data(y_val_cqr,y_lower,y_upper, f"CQR Net: {dataset_name}")
    coverage_cp_qnet, length_cp_qnet = helper.compute_coverage(y_val_cqr,cqr_lower,cqr_upper,significance,"CQR Net")
    print(f"CQR Coverage: {coverage_cp_qnet} Length: {length_cp_qnet}")


    dataset_name_vec.append(dataset_name)
    method_vec.append('CQR Net')
    coverage_vec.append(coverage_cp_qnet)
    length_vec.append(length_cp_qnet)
    seed_vec.append(seed)
    # predictions = None
    # with open('concrete-predictions.pkl', 'rb') as infile:
    #     predictions = pickle.load(infile)
    # print(predictions)
    # cqr_lower = predictions['y_lower']
    # cqr_upper = predictions['y_upper']
    #### End CQR
    plot_prob(args, range_vals, X_val, y_val, model, cqr_lower, cqr_upper)

    log_results((args.dataset_name, args.model_path, mean_coverage, std_coverage, mean_length, std_length))