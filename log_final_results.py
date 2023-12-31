from main import main
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
import jax.numpy as jnp
from conformal_bayes_code.run_sparsereg_conformal import fit_mcmc_laplace, run_cb, get_posterior
from sheets import log_results
from chr.run_chr import get_chr

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
        if torch.cuda.is_available():
            trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator="gpu", devices=[int(args.devices)], logger=logger, callbacks=callbacks)
        else:
            trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator="cpu", logger=logger, callbacks=callbacks)

        trainer.fit(model, train_loader, val_loader)
        torch.save(model.state_dict(), total_path)
    model.eval()
    return model

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    mean_coverages = []
    mean_lengths  = []
    torch.set_float32_matmul_precision('medium')
    args = get_parser_args()
    original_model_path = args.model_path
    for seed in range(5):
        setattr(args, "model_path", "{}_s{}".format(original_model_path, seed))
        setattr(args, "seed", seed)
        seed_everything(args.seed)
        mean_coverage, std_coverage, mean_length, std_length = main(args)
        mean_coverages.append(mean_coverage)
        mean_lengths.append(mean_length)

    final_mean_coverage = np.mean(mean_coverages)
    final_se_coverage = np.std(mean_coverages)/np.sqrt(len(mean_coverages))
    final_mean_length = np.mean(mean_lengths)
    final_se_length = np.std(mean_lengths)/np.sqrt(len(mean_lengths))
    log_results((args.dataset_name, "{}_allseeds".format(original_model_path), final_mean_coverage, final_se_coverage, final_mean_length, final_se_length))
        