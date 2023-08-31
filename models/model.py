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

        self.q = args.lq_norm_val
        self.arguments = args
        self.K = args.num_moments
        if args.constraint_weights is None:
            self.register_buffer("constraint_weights", torch.ones(self.K + 1)/(self.K+1))
        else:
            self.register_buffer("constraint_weights",torch.tensor(args.constraint_weights))
        self.register_buffer("zero_vec", torch.tensor(0))
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('train_loss', loss)
        return loss

    def compute_loss(self, batch):
        x, y = batch
        probs = self.smax(self(x))

        if self.annealing:
            self.constraint_weights[0] = max(0, (self.initial_temperature * (1 - self.current_epoch / self.annealing_epochs)))
        if self.constraint_weights[0] == 0:
            all_losses = [self.zero_vec]
        else:
            neg_entropy = torch.sum(torch.sum(probs * torch.log(probs), dim=1))
            all_losses = [neg_entropy]
        
        if self.loss_type == "moment":
            for moment_index in range(self.K):
                all_losses.append(torch.sum(torch.square(torch.sum(probs * torch.pow(self.range_vals, moment_index + 1), axis=1) - torch.pow(y, moment_index+1))))
        elif self.loss_type == "thomas":
            all_losses.append(torch.sum(probs * torch.square(self.range_vals.view(1, -1).expand(len(y), -1) - y.view(-1, 1)))) 
        elif self.loss_type == "thomas_lq":
            all_losses.append(torch.sum(probs * torch.pow(torch.abs(self.range_vals.view(1, -1).expand(len(y), -1) - y.view(-1, 1)), self.q)) )
        elif self.loss_type == "cross_entropy":
            
            all_losses.append(torch.sum(probs * torch.pow(torch.abs(self.range_vals.view(1, -1).expand(len(y), -1) - y.view(-1, 1)), self.q)) )
        loss = torch.sum(torch.stack(all_losses) * self.constraint_weights)/len(batch)
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.compute_loss(batch)
        self.log('val_loss', loss.item())
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.arguments.lr)
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
