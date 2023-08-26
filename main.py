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
def get_model(args):
    input_size, range_vals = get_input_and_range(args)

    model = GenModule(args, input_size, range_vals)

    total_path = "model_paths/{}.pth".format(args.model_path)
    if os.path.exists(total_path):
        model.load_state_dict(torch.load(total_path))
    else:
        train_loader, val_loader = get_loaders(args)
        logger = TensorBoardLogger("tb_logs", name=args.model_path)
        trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=args.devices, logger=logger)
        trainer.fit(model, train_loader, val_loader)
        torch.save(model.state_dict(), total_path)

    model.eval()
    return model

if __name__ == '__main__':
    args = get_parser_args()
    model = get_model(args) 
    
    X_val, y_val = get_val_data(args)
    input_size, range_vals = get_input_and_range(args)
    mean_coverage, std_coverage, mean_length, std_length = get_cp(args, range_vals, X_val, y_val, model)
    plot_prob(args, range_vals, X_val, y_val, model)
    log_results((args.dataset_name, args.model_path, mean_coverage, std_coverage, mean_length, std_length))