import torch
import matplotlib.pyplot as plt
from data import get_loaders, get_input_and_range, get_train_val_data
from create_argparser import get_parser_args
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from models.model import GenModule
import os
from sheets import log_results
from cp import get_cp, get_cp_lists
from plotter import plot_prob, plot_path, plot_violin
from models.callbacks import get_callbacks
from baselines.lei import lei
from baselines.ridge import conf_pred
import random
import numpy as np
from cqr_helpers.run_cqr import run_cqr

# Conformal Bayes imports
import jax.numpy as jnp
from conformal_bayes.run_sparsereg_conformal import fit_mcmc_laplace, run_cb

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
        trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=args.devices, logger=logger, callbacks=callbacks)
        trainer.fit(model, train_loader, val_loader)
        torch.save(model.state_dict(), total_path)
    model.eval()
    return model

# Looking for args.post_path: root directory/conformal_bayes/post_samples + a folder
# The folder contains the 3 distributions beta_post.npy, intercept_post.npy, b_post.npy
# args.post_save_path will be used for the new folder directory to save 
def get_posterior(args, X_train, y_train):
    beta_post = None
    intercept_post = None
    b_post = None
    sigma_post = None
    folder_path = f"conformal_bayes/post_samples/{args.post_path}"
    if os.path.exists(folder_path):
        beta_post = jnp.load(f"{folder_path}/beta_post.npy")
        intercept_post = jnp.load(f"{folder_path}/intercept_post.npy")
        b_post = jnp.load(f"{folder_path}/b_post.npy")
        sigma_post = jnp.load(f"{folder_path}/sigma_post.npy")
    else:
        beta_post, intercept_post, b_post, sigma_post = fit_mcmc_laplace(X_train, y_train, args.seed)
        if (args.post_save_path):
            total_save_path = f"conformal_bayes/post_samples/{args.post_save_path}"
            os.mkdir(total_save_path)
            np.save(f"{total_save_path}/beta_post", beta_post)
            np.save(f"{total_save_path}/intercept_post", intercept_post)
            np.save(f"{total_save_path}/b_post", b_post)
            np.save(f"{total_save_path}/sigma_post", sigma_post)
    return beta_post, intercept_post, b_post, sigma_post

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    X_train, y_train, X_val, y_val = get_train_val_data(args)

    input_size, range_vals = get_input_and_range(args)
    if args.cb:
        beta_post, intercept_post, b_post, sigma_post = get_posterior(args, X_train, y_train)
        mean_coverage, std_coverage, mean_length, std_length = run_cb(X_train, y_train, X_val, y_val, beta_post, intercept_post, b_post, sigma_post)
        log_results((args.dataset_name, args.model_path, mean_coverage, std_coverage, mean_length, std_length))
    if args.cqr:
        mean_coverage, std_coverage, mean_length, std_length = run_cqr(args)
        log_results((args.dataset_name, args.model_path, mean_coverage, std_coverage, mean_length, std_length))
    if args.lei:
        mean_coverage, std_coverage, mean_length, std_length = lei(args)
        log_results((args.dataset_name, args.model_path, mean_coverage, std_coverage, mean_length, std_length))
    elif args.ridge:  
        mean_coverage, std_coverage, mean_length, std_length = conf_pred(args, lambda_=.1)
        log_results((args.dataset_name, args.model_path, mean_coverage, std_coverage, mean_length, std_length))
    elif args.plot_dcp:
        model = get_model(args) 
        mean_coverage, std_coverage, mean_length, std_length = get_cp(args, range_vals, X_val, y_val, model)
        plot_path(args, range_vals, X_val, y_val, model)
        plot_prob(args, range_vals, X_val, y_val, model)
    else:  
        model = get_model(args) 
        coverages, lengths = get_cp_lists(args, range_vals, X_val, y_val, model)
        mean_coverage, std_coverage, mean_length, std_length = get_cp(args, range_vals,  X_val, y_val, model)
        plot_prob(args, range_vals, X_val, y_val, model)
        log_results((args.dataset_name, args.model_path, mean_coverage, std_coverage, mean_length, std_length))
        plot_violin(args, coverages, lengths)
        
    return mean_coverage, std_coverage, mean_length, std_length

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    for random_state_train_test_id in range(1):
        args = get_parser_args()
        setattr(args, "seed", random_state_train_test_id)
        seed_everything(random_state_train_test_id)
        main(args)
    