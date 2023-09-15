import torch
import matplotlib.pyplot as plt
from data import get_loaders, get_input_and_range, get_train_val_data
from create_argparser import get_parser_args
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from models.model import GenModule
import random
import numpy as np
import os
# from sheets import log_results
from cp import get_cp
from plotter import plot_prob
from models.callbacks import get_callbacks
import pickle

from cqr_helpers.run_cqr import run_cqr

torch.autograd.set_detect_anomaly(True)
def get_model(args):
    # logging.basicConfig(filename='results.log', encoding='utf-8', level=logging.INFO)
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
    torch.set_float32_matmul_precision('medium')
    dataset_name = args.dataset_name
    print(f"Dataset: {dataset_name}")
    model = get_model(args) 
    X_train, y_train, X_val, y_val = get_train_val_data(args)
    input_size, range_vals = get_input_and_range(args)
    mean_coverage, std_coverage, mean_length, std_length = get_cp(args, range_vals, X_val, y_val,model)
    print(f"Dataset: {dataset_name}")
    coverage_log = f"CP Coverage: {mean_coverage} +- {std_coverage} Length: {mean_length} +- {std_length}"
    print(coverage_log)
    cqr_avg_coverage, cqr_std_coverage, cqr_avg_length, cqr_std_length, cqr_lower, cqr_upper = run_cqr(X_train, y_train, X_val, y_val)
    # plot_prob(args, range_vals, X_val, y_val, model, cqr_lower, cqr_upper)
    

if __name__ == '__main__':
    args = get_parser_args()
    random_state_train_test_id = 0
    setattr(args, "seed", random_state_train_test_id)
    seed_everything(random_state_train_test_id)
    main(args)

    



