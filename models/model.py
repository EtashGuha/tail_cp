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

        self.model = model_class(args, input_size, len(range_vals))
        self.smax = torch.nn.Softmax(dim=1)
        self.register_buffer("range_vals",range_vals)
        self.arguments = args
        self.K = args.num_moments
        if args.constraint_weights is None:
            self.register_buffer("constraint_weights", torch.ones(self.K + 1)/(self.K+1))
        else:
            self.register_buffer("constraint_weights",torch.tensor(args.constraint_weights))
        assert len(self.constraint_weights == self.K + 1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('train_loss', loss)
        return loss

    def compute_loss(self, batch):
        x, y = batch
        probs = self.smax(self(x))
        neg_entropy = torch.sum(torch.sum(probs * torch.log2(probs), dim=1))
        moment_losses = [neg_entropy]
        
        for moment_index in range(self.K):
            moment_losses.append(torch.sum(torch.square(torch.sum(probs * torch.pow(self.range_vals, moment_index + 1), axis=1) - torch.pow(y, moment_index+1))))
        
        loss = torch.sum(torch.stack(moment_losses) * self.constraint_weights)/len(batch)
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
