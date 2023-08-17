import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import copy
from pytorch_lightning.callbacks import Callback
from transformer import Transformer
# Load the diabetes dataset
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger("tb_logs", name="my_model")

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

class MetricTracker(Callback):

  def __init__(self):
    self.collection = []

  def on_validation_epoch_end(self, trainer, module):
    elogs = trainer.logged_metrics # access it here
    self.collection.append(copy.deepcopy(elogs))
    # do whatever is needed

# Normalize both data and labels using StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
y_normalized = scaler.fit_transform(y.reshape(-1, 1)).flatten()  # Reshape y for scaler

# Split the normalized data and labels into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

# Create a DataLoader for training and validation data
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define the MLP model
class MLPModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Sequential(
          nn.Linear(input_size, 128),
          nn.LayerNorm(128),
          nn.Dropout(.25),
          nn.Linear(128, 1000),
          nn.LayerNorm(1000),
          nn.ReLU(),
          nn.Dropout(.25),
          nn.Linear(1000, 1000),
          nn.LayerNorm(1000),
          nn.ReLU(),
          nn.Dropout(.25),
          nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        
        return self.fc1(x)

# Define a LightningModule
class MLPModule(pl.LightningModule):
    def __init__(self, input_size, range_vals, K=5, constraint_weights=None):
        super(MLPModule, self).__init__()
        self.model = MLPModel(input_size, len(range_vals))
        self.smax = torch.nn.Softmax(dim=1)
        self.range_vals = range_vals
        self.K = K
        if constraint_weights is None:
            self.constraint_weights = (torch.ones(self.K + 1)/(self.K+1)).cuda()
        else:
            self.constraint_weights = torch.tensor(constraint_weights).cuda()
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
            try:
                moment_losses.append(torch.sum(torch.square(torch.sum(probs * torch.pow(self.range_vals, moment_index + 1), axis=1) - torch.pow(y, moment_index+1))))
            except:
                breakpoint()
        loss = torch.sum(torch.stack(moment_losses) * self.constraint_weights)/len(batch)
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.compute_loss(batch)
        self.log('val_loss', loss.item())
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        return [optimizer],[scheduler]

# Initialize the LightningModule
input_size = X_train.shape[1]
range_vals = torch.linspace(torch.min(y_train), torch.max(y_train), 50).to(device)
model = MLPModule(input_size, range_vals, constraint_weights=[1,1/2, 1/4, 1/16, 1/32, 1/64])

# Initialize the Trainer

trainer = pl.Trainer(max_epochs=1000, gpus=1, logger=logger)
# trainer.fit(model, train_loader, val_loader)
# torch.save(model.state_dict(), "model_state_dict.pth")
model.load_state_dict(torch.load("model_state_dict.pth"))

model.to(device)
model.eval()

step_val = (torch.max(y_train) - torch.min(y_train))/50
indices = ((y_val - torch.min(y_train))/step_val).to(torch.int)
def plot_index(idx):
    plt.clf() 
    plt.plot(range_vals.detach().cpu(), torch.nn.functional.softmax(model(X_val), dim=1)[idx].detach().cpu())
    plt.savefig("plsagain_{}.png".format(idx))

scores = torch.nn.functional.softmax(model(X_val), dim=1)
all_scores = scores[torch.arange(len(X_val)), indices.long()]

def percentile_excluding_index(vector, i, percentile):
    # Remove the value at the i-th index
    modified_vector = torch.cat((vector[:i], vector[i+1:]))

    # Calculate the percentile on the modified vector
    percentile_value = torch.quantile(modified_vector, percentile)
    
    return percentile_value
alpha = .1
lengths = []
coverage = []
for i in range(len(X_val)):
    percentile_val = percentile_excluding_index(all_scores, i, alpha)
    cp_vals = range_vals[torch.where(scores[i] > percentile_val)]
    top_range, bottom_range = max(cp_vals), min(cp_vals)
    if top_range >= y_val[i] and y_val[i] >= bottom_range:
        coverage.append(1)
    else:
        coverage.append(0)
    lengths.append(top_range - bottom_range)
breakpoint()