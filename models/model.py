import torch
import pytorch_lightning as pl
from torch import optim
from models.transformer import Transformer
from models.mlp import MLPModel
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, MultiStepLR

# Define a LightningModule
class GenModule(pl.LightningModule):
    def __init__(self, args, input_size, range_vals):
        super(GenModule, self).__init__()

        if args.model == "mlp":
            model_class = MLPModel
        elif args.model == "transformer":
            model_class = Transformer
        self.loss_type = args.loss_type
        self.model = model_class(args, input_size, len(range_vals))
        self.smax = torch.nn.Softmax(dim=1)
        self.register_buffer("range_vals",range_vals)
        self.annealing = args.annealing
        if self.annealing:
            self.initial_temperature = 1
            self.annealing_epochs = args.annealing_epochs
        if self.loss_type == "cross_entropy_quantile":
            self.register_buffer("alpha", torch.tensor(args.alpha))

        self.q = args.lq_norm_val
        self.arguments = args
        self.K = args.num_moments
        if args.constraint_weights is None:
            self.register_buffer("constraint_weights", torch.ones(self.K + 1)/(self.K+1))
        else:
            self.register_buffer("constraint_weights",torch.tensor(args.constraint_weights))
            
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('train_loss', loss)
        return loss

    def compute_loss(self, batch):
        x, y = batch
        pre_probs = self(x)
        probs = self.smax(pre_probs)

        if self.annealing:
            self.constraint_weights[0] = max(0, (self.initial_temperature * (1 - self.current_epoch / self.annealing_epochs)))
        log_vals = torch.log(probs)
        log_vals[probs == 0] = 0
        neg_entropy = torch.sum(torch.sum(probs * log_vals, dim=1))
        all_losses = [neg_entropy]
        if torch.isnan(neg_entropy).any():
            breakpoint()
        if self.loss_type == "moment":
            for moment_index in range(self.K):
                all_losses.append(torch.sum(torch.square(torch.sum(probs * torch.pow(self.range_vals, moment_index + 1), axis=1) - torch.pow(y, moment_index+1))))
        elif self.loss_type == "thomas":
            all_losses.append(torch.sum(probs * torch.square(self.range_vals.view(1, -1).expand(len(y), -1) - y.view(-1, 1)))) 
        elif self.loss_type == "thomas_lq":
            all_losses.append(torch.sum(probs * torch.pow(torch.abs(self.range_vals.view(1, -1).expand(len(y), -1) - y.view(-1, 1)), self.q)) )
        elif self.loss_type == "cross_entropy":
            step_val = (max(self.range_vals) - min(self.range_vals))/(len(self.range_vals) - 1)
            indeces = torch.round((y - min(self.range_vals))/step_val)
            all_scores = torch.nn.functional.cross_entropy(pre_probs, indeces.long())
            all_losses.append(all_scores)

        elif self.loss_type == "cross_entropy_quantile":
            step_val = (max(self.range_vals) - min(self.range_vals))/(len(self.range_vals) - 1)
            indices_up = torch.ceil((y - min(self.range_vals))/step_val)
            indices_down = torch.floor((y - min(self.range_vals))/step_val)
            how_much_each_direction = (y - min(self.range_vals))/step_val - indices_down

            weight_up = how_much_each_direction
            weight_down = 1 - how_much_each_direction
            all_scores = -1 * torch.quantile(probs[torch.arange(len(probs)), indices_up.long()] * weight_up + probs[torch.arange(len(probs)), indices_down.long()] * weight_down, self.alpha)
            all_losses.append(all_scores)

        loss = torch.sum(torch.stack(all_losses) * self.constraint_weights)/len(batch)
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.compute_loss(batch)
        self.log('val_loss', loss.item())
        return loss
    
    def configure_optimizers(self):
        if self.arguments.optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.arguments.lr, weight_decay=self.arguments.weight_decay)
        elif self.arguments.optimizer =="adamw":
            optimizer = optim.AdamW(self.parameters(), lr=self.arguments.lr, weight_decay=self.arguments.weight_decay)
        elif self.arguments.optimizer == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.arguments.lr, weight_decay=self.arguments.weight_decay)

        if self.arguments.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.arguments.max_epochs,
            )
        elif self.arguments.lr_scheduler == "cosine_warmup":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                self.arguments.lr_warmup_epochs,
                self.arguments.max_epochs,
            )
        elif self.arguments.lr_scheduler == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=1,
                end_factor=self.arguments.lr_drop,
                total_iters=self.arguments.max_epochs,
            )
        elif self.arguments.lr_scheduler == "step":
            scheduler = MultiStepLR(
                optimizer,
                self.arguments.lr_steps,
                gamma=self.arguments.lr_drop,
            )
        
        return [optimizer],[scheduler]
